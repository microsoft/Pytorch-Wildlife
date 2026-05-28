# installer/probe_gpu_quality.ps1 — dot-sourceable layer-2 quality probe (Windows).
#                                    GPU runtime sidecar DLL surface +
#                                    cuDNN ≥9.10 floor + driver-version sanity.
#
# Purpose
#   Layer-2 of the Sparrow Engine install-time selector on Windows. Runs ONLY after
#   layer-1 (`probe.ps1`) has returned `gpu`. Reports three follow-up signals:
#     1. Are all hard-required GPU runtime sidecar DLLs reachable? ORT 1.25.1's
#        ORT CUDA provider links against six CUDA libs (verified via readelf
#        against the published Linux wheel, 2026-05-27); Windows equivalents
#        are `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`,
#        `curand64_10.dll`, `cufft64_11.dll`, `cudnn64_9.dll`. The
#        sparrow-engine GPU image-decode path additionally loads
#        `nvjpeg64_12.dll` via dlopen — missing nvJPEG fails at first
#        inference with `SparrowEngineError::NvjpegUnavailable`.
#     2. Is cuDNN ≥9.10 reachable? Windows ships cuDNN as `cudnn64_9.dll`
#        plus version-stamped sidecar files (e.g. `cudnn_9.10.2.21.dll`); the
#        FileVersion property of `cudnn64_9.dll` is the most reliable signal.
#     3. Is the GPU compute-capability ≥sm_80? FP16 production cells need
#        Ampere Tensor Cores; pre-Ampere falls back to FP32 at 2-3× latency.
#
# Usage
#   Dot-source form (preferred):
#       . .\installer\probe_gpu_quality.ps1
#       probe_gpu_quality
#       switch ($env:SPARROW_ENGINE_GPU_QUALITY) {
#           'ok'         { }                          # silent install
#           'sm_warn'    { Write-Warning '...' }
#           'cudnn_warn' { Write-Warning '...' }
#           'cudnn_err'  { exit 11 }
#       }
#
#   The `cudnn_err` verdict is the hard-fail bucket for ANY missing GPU
#   runtime prerequisite (cuDNN, cuBLAS, cuFFT, nvJPEG, etc.); the verdict
#   name is preserved for back-compat with `sparrow-engine-install.ps1`'s
#   switch-handling. The verdict REASON string carries which specific DLL is
#   the actual culprit.
#
#   Direct invocation:
#       powershell -NoProfile -ExecutionPolicy Bypass -File .\installer\probe_gpu_quality.ps1
#
# Env vars set
#   SPARROW_ENGINE_GPU_QUALITY         ok | sm_warn | cudnn_warn | cudnn_err
#   SPARROW_ENGINE_GPU_QUALITY_REASON  short string explaining the verdict
#
# Script-scope alias variables also set:
#   $script:SparrowEngineGpuQuality
#   $script:SparrowEngineGpuQualityReason
#
# Exit codes
#   This script always exits 0. The wrapper translates `cudnn_err` into
#   exit 11 per `final_design.md § 2.10`.
#
# Design source
#   docs/design/phase4.1-install-selector/final_design.md § 2.4
#   docs/design/phase4.1-install-selector/round_02/scripts-architect_proposal.md § 1.2.1 + § 3.2
#
# DLL surface ground truth (2026-05-27, RP-20):
#   ORT 1.25.1 CUDA provider DT_NEEDED CUDA libs:
#     cublasLt64_12.dll, cublas64_12.dll, curand64_10.dll, cufft64_11.dll,
#     cudart64_12.dll, cudnn64_9.dll
#   Engine-side nvJPEG (dlopen at first JpegDecoder::new):
#     nvjpeg64_12.dll
#
# cuDNN ≥9.10 floor citation (verified 2026-05-08):
#   - sparrow-engine/scripts/ort-env.sh:167-168 — "cuDNN: we require 9.10+ for SpeciesNet
#     on sm_89 (cuDNN 9.8 has a Conv engine bug with asymmetric padding —
#     'No valid engine configs for ConvFwd_'). PyTorch/TF bundle 9.8."
#   - docs/lessons.md:29
#   - docs/tech_report/06_gotchas_and_constraints.md:17-25

# ---------------------------------------------------------------------------
# Helper: search for a CUDA runtime sidecar DLL by exact basename.
#
# Looks in the canonical Windows locations a GPU install drops DLLs:
#   1. %SystemRoot%\System32 (cuDNN MSI installer drops here; CUDA Runtime
#      apt-style installs land here on some setups).
#   2. %CUDA_PATH%\bin (standalone CUDA Toolkit installer).
#   3. %CUDA_PATH_V*_*%\bin (side-by-side Toolkit installs — NVIDIA writes
#      version-suffixed env vars for each major.minor).
#
# Returns the first matching absolute path, or $null if not found.
# ---------------------------------------------------------------------------
function Search-RequiredDll {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$DllName)
    $candidates = @()
    if ($env:SystemRoot) {
        $candidates += (Join-Path $env:SystemRoot "System32\$DllName")
    }
    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH "bin\$DllName")
    }
    Get-ChildItem env: -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^CUDA_PATH_V\d+_\d+$' } |
        ForEach-Object { $candidates += (Join-Path $_.Value "bin\$DllName") }
    foreach ($p in $candidates) {
        if ($p -and (Test-Path -LiteralPath $p -PathType Leaf)) {
            return $p
        }
    }
    return $null
}

function probe_gpu_quality {
    [CmdletBinding()]
    param()

    $env:SPARROW_ENGINE_GPU_QUALITY = ''
    $env:SPARROW_ENGINE_GPU_QUALITY_REASON = ''

    # 0. Hard-required GPU runtime sidecar DLLs (RP-20). Collect ALL missing
    #    DLLs so the user fixes them in one cycle.
    $requiredDlls = @(
        @{ Name = 'cudart64_12.dll';   PipPkg = 'nvidia-cuda-runtime-cu12' },
        @{ Name = 'cublas64_12.dll';   PipPkg = 'nvidia-cublas-cu12' },
        @{ Name = 'cublasLt64_12.dll'; PipPkg = 'nvidia-cublas-cu12' },
        @{ Name = 'curand64_10.dll';   PipPkg = 'nvidia-curand-cu12' },
        @{ Name = 'cufft64_11.dll';    PipPkg = 'nvidia-cufft-cu12' },
        @{ Name = 'nvjpeg64_12.dll';   PipPkg = 'nvidia-nvjpeg-cu12' }
    )
    $missing = @()
    foreach ($req in $requiredDlls) {
        $found = Search-RequiredDll -DllName $req.Name
        if (-not $found) {
            $missing += $req
        }
    }

    if ($missing.Count -gt 0) {
        $lines = $missing | ForEach-Object {
            "    - $($_.Name)  (pip pkg: $($_.PipPkg))"
        }
        $hint = ($lines -join "`n")
        $env:SPARROW_ENGINE_GPU_QUALITY = 'cudnn_err'
        $env:SPARROW_ENGINE_GPU_QUALITY_REASON = @"
GPU runtime sidecar DLL(s) missing — spe-gpu will fail at first inference with a LoadLibraryW error.
$hint

Install via ONE of (see docs/user-manual.md §2.5):
  Option A — CUDA Toolkit installer: download CUDA 12.6+ from https://developer.nvidia.com/cuda-downloads and cuDNN 9.10+ from https://developer.nvidia.com/cudnn
  Option B — Python sidecar wheels (no admin): uv pip install --target $env:USERPROFILE\.sparrow-engine\cuda-sidecars nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12
"@
        $script:SparrowEngineGpuQuality = $env:SPARROW_ENGINE_GPU_QUALITY
        $script:SparrowEngineGpuQualityReason = $env:SPARROW_ENGINE_GPU_QUALITY_REASON
        Write-Output 'fail'
        return
    }

    # 1. cuDNN check — locate cudnn64_9.dll in the canonical Windows locations
    #    (System32 from the cuDNN MSI installer, %CUDA_PATH%\bin from the
    #    standalone CUDA toolkit installer). Read FileVersion to compare
    #    against the 9.10.0 floor.
    $cudnnDll = Search-RequiredDll -DllName 'cudnn64_9.dll'

    if (-not $cudnnDll) {
        $env:SPARROW_ENGINE_GPU_QUALITY = 'cudnn_err'
        $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "cuDNN 9.x DLL (cudnn64_9.dll) not found in System32 or %CUDA_PATH%\bin; the GPU flavor will fail at first inference. Install cuDNN 9.10+ from https://developer.nvidia.com/cudnn"
    } else {
        # FileVersion is e.g. "9.10.2.21" or "9.8.0.87".
        $ver = (Get-Item $cudnnDll).VersionInfo.FileVersion
        if (-not $ver) {
            # FileVersion property missing — degrade to cudnn_warn so we don't
            # block install on a metadata edge case, but flag it.
            $env:SPARROW_ENGINE_GPU_QUALITY = 'cudnn_warn'
            $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "cuDNN DLL found at $cudnnDll but FileVersion metadata missing; cannot verify >= 9.10 floor"
        } else {
            $parts = $ver -split '\.'
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if (($major -gt 9) -or ($major -eq 9 -and $minor -ge 10)) {
                $env:SPARROW_ENGINE_GPU_QUALITY = 'ok'
                $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "cuDNN $ver found at $cudnnDll (>= 9.10.0 floor)"
            } else {
                $env:SPARROW_ENGINE_GPU_QUALITY = 'cudnn_warn'
                $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "cuDNN $ver found at $cudnnDll, < 9.10.0 floor; SpeciesNet on sm_89 will fail (known 9.8 ConvFwd asymmetric-padding bug). Install cuDNN 9.10+ from https://developer.nvidia.com/cudnn"
            }
        }
    }

    # cudnn_err is hard-fail by policy. Set verdict + return early.
    if ($env:SPARROW_ENGINE_GPU_QUALITY -eq 'cudnn_err') {
        $script:SparrowEngineGpuQuality = $env:SPARROW_ENGINE_GPU_QUALITY
        $script:SparrowEngineGpuQualityReason = $env:SPARROW_ENGINE_GPU_QUALITY_REASON
        Write-Output 'fail'
        return
    }

    # 2. Compute-capability check via nvidia-smi.exe.
    $nvsmi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -LiteralPath $nvsmi -PathType Leaf) {
        $outFile = Join-Path $env:TEMP "bongo_gpuq_cc_$([guid]::NewGuid()).txt"
        try {
            $proc = Start-Process -FilePath $nvsmi `
                -ArgumentList '--query-gpu=compute_cap','--format=csv,noheader' `
                -NoNewWindow -PassThru `
                -RedirectStandardOutput $outFile
            if ($proc.WaitForExit(5000) -and $proc.ExitCode -eq 0) {
                $first = Get-Content $outFile -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($first) {
                    $cc = $first.Trim() -replace '\.', ''
                    if ([int]::TryParse($cc, [ref]$null) -and ([int]$cc -lt 80)) {
                        if ($env:SPARROW_ENGINE_GPU_QUALITY -eq 'ok') {
                            $env:SPARROW_ENGINE_GPU_QUALITY = 'sm_warn'
                            $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "$($env:SPARROW_ENGINE_GPU_QUALITY_REASON); compute_cap=$cc (< sm_80) — FP16 production cells fall back to FP32, ~2-3x slower"
                        } else {
                            $env:SPARROW_ENGINE_GPU_QUALITY_REASON = "$($env:SPARROW_ENGINE_GPU_QUALITY_REASON); compute_cap=$cc (< sm_80)"
                        }
                    }
                }
            } else {
                try { $proc.Kill() } catch {}
            }
        } catch {
            # nvidia-smi.exe invocation failed — leave quality verdict at the
            # cuDNN-derived state. Compute-cap check is informational, not gating.
        } finally {
            Remove-Item $outFile -ErrorAction SilentlyContinue
        }
    }

    $script:SparrowEngineGpuQuality = $env:SPARROW_ENGINE_GPU_QUALITY
    $script:SparrowEngineGpuQualityReason = $env:SPARROW_ENGINE_GPU_QUALITY_REASON

    # Map the 4-state quality verdict onto a 3-state stdout signal: pass | warn | fail.
    switch ($env:SPARROW_ENGINE_GPU_QUALITY) {
        'ok'         { Write-Output 'pass' }
        'sm_warn'    { Write-Output 'warn' }
        'cudnn_warn' { Write-Output 'warn' }
        'cudnn_err'  { Write-Output 'fail' }
        default      { Write-Output 'warn' }
    }
}

# Direct-invocation block — fires only when this file is executed (not dot-sourced).
if ($MyInvocation.InvocationName -ne '.') {
    probe_gpu_quality
}
