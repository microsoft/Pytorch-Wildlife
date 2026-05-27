# installer/probe.ps1 — dot-sourceable CUDA-detection probe (Windows).
#
# Purpose
#   Layer-1 of the Sparrow Engine install-time selector on Windows: decide
#   whether the host should prefer the CPU CLI binary (`spe`) or the GPU CLI
#   binary (`spe-gpu`). Sets POSIX-style env vars that the wrapper (`installer/sparrow-engine-install.ps1`)
#   reads, plus PowerShell-internal PascalCase aliases for PS-native callers.
#
# Usage
#   Dot-source form (preferred — wrapper integration):
#       . .\installer\probe.ps1
#       probe_cuda                                     # function name matches probe.sh
#       Write-Host $env:SPARROW_ENGINE_DETECTED_FLAVOR          # cpu | gpu
#       Write-Host $env:SPARROW_ENGINE_DETECTED_PROBE_REASON    # diagnostic string
#
#   Direct invocation (one-shot, e.g. `--probe-only` or smoke test):
#       powershell -NoProfile -ExecutionPolicy Bypass -File .\installer\probe.ps1
#       $env:SPARROW_ENGINE_INSTALL_FLAVOR='gpu'; powershell ... -File probe.ps1   # honors override
#
# Env vars set
#   SPARROW_ENGINE_DETECTED_FLAVOR        cpu | gpu
#   SPARROW_ENGINE_DETECTED_PROBE_REASON  short string explaining the decision
#
# Script-scope alias variables also set (PowerShell-internal):
#   $script:SparrowEngineDetectedFlavor
#   $script:SparrowEngineDetectedProbeReason
#
# Exit codes
#   This script always exits 0 (probe never blocks). The wrapper consumes the
#   env vars and emits its own exit codes per `final_design.md § 2.10`.
#
# Design source
#   docs/design/phase4.1-install-selector/final_design.md § 2.3
#   docs/design/phase4.1-install-selector/round_01/scripts-architect_proposal.md § 3.2 (canonical pseudocode)
#   docs/design/phase4.1-install-selector/round_02/scripts-architect_proposal.md § 1.3 + § 3.1 (T-5 contract)
#
# Notes on portability
#   - Targets PowerShell 5.1 (ships with Windows 10 1607+ / Server 2016+); does
#     not require PowerShell 7.
#   - Read-only API only: Test-Path, Get-CimInstance, Process.WaitForExit. No
#     elevation required.
#   - PowerShell 5.1 `Start-Process -Wait` has no timeout; `Wait-Job -Timeout`
#     does not kill the child process (PowerShell #15555, #10501). The .NET
#     BCL primitive `Process.WaitForExit(int)` is used to bound nvidia-smi.exe
#     at 5s and reliably kill on overrun.

function probe_cuda {
    [CmdletBinding()]
    param()

    # 1. Override path — highest priority (wrapper resolves CLI flags first).
    if ($env:SPARROW_ENGINE_INSTALL_FLAVOR) {
        $f = $env:SPARROW_ENGINE_INSTALL_FLAVOR.ToLower()
        if ($f -eq 'cpu' -or $f -eq 'gpu') {
            $env:SPARROW_ENGINE_DETECTED_FLAVOR = $f
            $env:SPARROW_ENGINE_DETECTED_PROBE_REASON = "SPARROW_ENGINE_INSTALL_FLAVOR=$f (env override)"
            $script:SparrowEngineDetectedFlavor = $f
            $script:SparrowEngineDetectedProbeReason = $env:SPARROW_ENGINE_DETECTED_PROBE_REASON
            Write-Output $f
            return
        } else {
            Write-Warning "SPARROW_ENGINE_INSTALL_FLAVOR='$($env:SPARROW_ENGINE_INSTALL_FLAVOR)' not in {cpu, gpu}; ignoring."
        }
    }

    $flavor = $null
    $reason = $null

    # 2. nvidia-smi.exe primary probe — bounded 5s timeout.
    $nvsmi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -LiteralPath $nvsmi -PathType Leaf) {
        $outFile = Join-Path $env:TEMP "bongo_nvsmi_out_$([guid]::NewGuid()).txt"
        $errFile = Join-Path $env:TEMP "bongo_nvsmi_err_$([guid]::NewGuid()).txt"
        try {
            $proc = Start-Process -FilePath $nvsmi -ArgumentList '-L' `
                -NoNewWindow -PassThru `
                -RedirectStandardOutput $outFile `
                -RedirectStandardError  $errFile
            if (-not $proc.WaitForExit(5000)) {
                try { $proc.Kill() } catch {}
                $reason = 'nvidia-smi.exe timed out (>5s); falling back to WMI probe'
                # Fall through to WMI for a second chance.
            } elseif ($proc.ExitCode -eq 0) {
                $out = Get-Content $outFile -ErrorAction SilentlyContinue
                if ($out -and ($out | Select-Object -First 1) -match 'GPU \d+:') {
                    $flavor = 'gpu'
                    $reason = "nvidia-smi.exe reports: $($out | Select-Object -First 1)"
                }
            } else {
                $reason = "nvidia-smi.exe exit $($proc.ExitCode); falling back to WMI probe"
            }
        } catch {
            $reason = "nvidia-smi.exe invocation error: $($_.Exception.Message); falling back to WMI probe"
        } finally {
            Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
        }
    }

    # 3. WMI fallback — degraded GPU detection. Catches systems where the
    #    NVIDIA driver was uninstalled but the adapter is still in WMI, or
    #    `nvidia-smi.exe` was deleted/quarantined.
    if (-not $flavor) {
        try {
            $vids = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
                    Where-Object { $_.Name -match 'NVIDIA' }
            if ($vids) {
                $nvcuda = Join-Path $env:SystemRoot 'System32\nvcuda.dll'
                if (Test-Path -LiteralPath $nvcuda -PathType Leaf) {
                    $flavor = 'gpu'
                    $reason = "WMI reports NVIDIA adapter ($($vids[0].Name)) and nvcuda.dll present; nvidia-smi.exe absent or broken — install may fail at first inference, override with --flavor cpu if so"
                }
            }
        } catch {
            # WMI not available (very rare on Win10/11). Fall through to default.
        }
    }

    # 4. Default: CPU.
    if (-not $flavor) {
        $flavor = 'cpu'
        if (-not $reason) {
            $reason = 'no NVIDIA driver detected (no nvidia-smi.exe, no NVIDIA video adapter via WMI)'
        }
    }

    $env:SPARROW_ENGINE_DETECTED_FLAVOR = $flavor
    $env:SPARROW_ENGINE_DETECTED_PROBE_REASON = $reason
    $script:SparrowEngineDetectedFlavor = $flavor
    $script:SparrowEngineDetectedProbeReason = $reason
    Write-Output $flavor
}

# Direct-invocation block — fires only when this file is executed (not dot-sourced).
# `$MyInvocation.InvocationName` is the script path on direct execution; the
# unary-call form `& script.ps1` also lands here. Dot-sourcing (`. ./script.ps1`)
# leaves `$MyInvocation.InvocationName` as `.` so the call is suppressed.
if ($MyInvocation.InvocationName -ne '.') {
    probe_cuda
}
