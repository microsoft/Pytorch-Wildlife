<#
.SYNOPSIS
    sparrow-engine-install.ps1 — Windows install wrapper (PowerShell mirror of sparrow-engine-install.sh).

.DESCRIPTION
    Probes hardware once, then installs the matching flavor (CPU or GPU) via the
    selected mode (-Pip, -Cli, -Docker). Mirrors sparrow-engine-install.sh exit codes
    0..14 per docs/design/phase4.1-install-selector/final_design.md § 2.10.

    Defense-in-depth (truncation safety): the entire body is wrapped in
    Invoke-Main; the LAST line is `Invoke-Main @PSBoundParameters; exit
    $LASTEXITCODE`. A truncated download leaves Invoke-Main undefined so
    PowerShell parse-fails before any partial work runs.

.NOTES
    Targets Windows PowerShell 5.1 and PowerShell 7+. No PS6+ exclusives
    (Select-Object -SkipLast etc.) so the script runs on stock Windows.
#>

[CmdletBinding()]
param(
    # Mode (default: auto-detect; if probe yields nothing, falls through to cli).
    [switch]$Pip,
    [switch]$Cli,
    [switch]$Docker,

    # Flavor (cpu|gpu|auto). auto = invoke probe; ignore $env:SPARROW_ENGINE_INSTALL_FLAVOR.
    [ValidateSet('cpu','gpu','auto')]
    [string]$Flavor = 'auto',

    # Bypass flags.
    [switch]$Reinstall,           # same-flavor force-overwrite
    [switch]$Reprobe,             # cross-flavor switch + driver-upgrade re-probe
    [switch]$Uninstall,           # remove install per state file
    [switch]$ForceRcOverwrite,    # force replace sparrow-engine block in $PROFILE

    # Convenience.
    [switch]$ProbeOnly,           # print verdict + reason; do not install
    [switch]$DryRun,              # print actions; no network/install/state-write
    [switch]$Yes,                 # suppress [y/N] prompts
    [string]$Version,             # pin a specific sparrow-engine release
    [int]$Retries = 3,
    [switch]$Help
)

# ----- Constants ---------------------------------------------------------

$Script:SparrowEngineVersion       = if ($Version) { $Version } elseif ($env:SPARROW_ENGINE_VERSION) { $env:SPARROW_ENGINE_VERSION } else { '0.1.17' }
$Script:InstallRoot        = Join-Path $env:LOCALAPPDATA 'Programs\sparrow-engine'         # %LocalAppData%\Programs\sparrow-engine
$Script:UserSparrowEngineDir       = Join-Path $env:USERPROFILE '.sparrow-engine'                  # state file + RC sentinel home
$Script:StateFile          = Join-Path $Script:UserSparrowEngineDir 'installed.json'
$Script:SentinelStart      = '# >>> sparrow-engine >>>'
$Script:SentinelEnd        = '# <<< sparrow-engine <<<'
# Default release base = public GH Releases asset URL (Phase E B-02 fix; was
# `file:///%TEMP%/sparrow-engine-release/v{ver}/` dev placeholder). Honors
# `$env:SPARROW_ENGINE_RELEASE_BASE` for staging mirrors / internal proxies.
$Script:DefaultReleaseBase = "https://github.com/microsoft/Pytorch-Wildlife/releases/download/v$Script:SparrowEngineVersion/"
# Helper-script base = immutable raw-tag path. Helper scripts (probe.ps1,
# probe_gpu_quality.ps1) live in the tagged source tree, NOT as release
# assets. E-R2-1 fix. Override via `$env:SPARROW_ENGINE_HELPER_BASE`.
$Script:DefaultHelperBase  = "https://raw.githubusercontent.com/microsoft/Pytorch-Wildlife/refs/tags/v$Script:SparrowEngineVersion/installer/"
# Helper-script cache dir for piped install (B-01) — used when invoked via
# `iex (irm <url>)` and no probe.ps1 / probe_gpu_quality.ps1 exists on disk
# next to the wrapper.
$Script:HelperCacheDir     = Join-Path $env:LOCALAPPDATA ("sparrow-engine\cache\v" + $Script:SparrowEngineVersion)

# ----- Logging ------------------------------------------------------------

function Write-Info  { param([string]$Msg) Write-Host "  $Msg" }
function Write-Step  { param([string]$Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Warn2 { param([string]$Msg) Write-Host "warn: $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "error: $Msg" -ForegroundColor Red }
function Die {
    param([int]$Code, [string]$Msg)
    Write-Err $Msg
    exit $Code
}

# ----- Probe integration -------------------------------------------------

function Get-ScriptDir {
    if ($PSScriptRoot) { return $PSScriptRoot }
    if ($MyInvocation.MyCommand.Path) {
        return (Split-Path -Parent $MyInvocation.MyCommand.Path)
    }
    return $null
}

# Resolve a helper script (probe.ps1, probe_gpu_quality.ps1) — either
# co-located on disk next to this wrapper (disk install) OR fetched once
# from the release URL into $Script:HelperCacheDir (piped install via
# `iex (irm <url>)`). Phase E B-01 fix: piped invocation previously failed
# because $PSScriptRoot is empty when invoked as a script block.
# The fetch path uses Get-ReleaseBase which honors $env:SPARROW_ENGINE_RELEASE_BASE.
# Returns the absolute resolved path; throws via Die on failure.
function Resolve-Helper {
    param([Parameter(Mandatory)][string]$Name)

    $scriptDir = Get-ScriptDir
    if ($scriptDir) {
        $localPath = Join-Path $scriptDir $Name
        if (Test-Path -LiteralPath $localPath) { return $localPath }
    }

    $cachePath = Join-Path $Script:HelperCacheDir $Name
    if (Test-Path -LiteralPath $cachePath) { return $cachePath }

    $base = Get-HelperBase
    $url = "$base$Name"
    if (-not (Test-Path -LiteralPath $Script:HelperCacheDir)) {
        New-Item -ItemType Directory -Force -Path $Script:HelperCacheDir | Out-Null
    }
    $tmpPath = "$cachePath.tmp"
    Write-Info "fetching $Name from $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $tmpPath -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop | Out-Null
    } catch {
        if (Test-Path -LiteralPath $tmpPath) { Remove-Item -LiteralPath $tmpPath -Force -ErrorAction SilentlyContinue }
        Die 4 "failed to fetch $Name from $url (piped install fallback; download install.ps1 + probe.ps1 + probe_gpu_quality.ps1 from the same tag and run from disk if your network blocks raw.githubusercontent.com): $($_.Exception.Message)"
    }
    Move-Item -LiteralPath $tmpPath -Destination $cachePath -Force
    return $cachePath
}

function Invoke-Probe {
    # Dot-source installer/probe.ps1 (sibling file or fetched cache copy).
    # probe_cuda sets $env:SPARROW_ENGINE_DETECTED_FLAVOR + reason and
    # writes 'cpu' or 'gpu' to the pipeline.
    $probe = Resolve-Helper -Name 'probe.ps1'
    . $probe
    if (-not (Get-Command -Name probe_cuda -ErrorAction SilentlyContinue)) {
        Die 8 "probe.ps1 did not define probe_cuda; cannot resolve flavor."
    }
    return (probe_cuda)
}

function Resolve-Flavor {
    param([string]$Requested)
    if ($Requested -eq 'cpu' -or $Requested -eq 'gpu') {
        return $Requested
    }
    # 'auto' explicitly means: ignore env override, force probe (final_design § 2.3).
    Remove-Item Env:\SPARROW_ENGINE_INSTALL_FLAVOR -ErrorAction SilentlyContinue
    $verdict = Invoke-Probe
    if ($verdict -ne 'cpu' -and $verdict -ne 'gpu') {
        Die 1 "probe returned unexpected verdict: '$verdict'."
    }
    return $verdict
}

# ----- Layer-2 GPU quality probe (cuDNN >=9.10 floor) --------------------
#
# Dot-sources installer/probe_gpu_quality.ps1 (sibling). probe_gpu_quality
# sets $env:SPARROW_ENGINE_GPU_QUALITY (4-state: ok|sm_warn|cudnn_warn|cudnn_err) +
# $env:SPARROW_ENGINE_GPU_QUALITY_REASON. Dispatch per `final_design.md § 2.4 + § 2.10`:
#   - ok          → silent continue
#   - sm_warn     → log warn + continue (FP16 falls back to FP32; not blocking)
#   - cudnn_warn  → log warn + continue (cuDNN reachable but version unknown
#                    or below floor; SpeciesNet may fail at first inference)
#   - cudnn_err   → die 11 (cuDNN <9.10 or absent; install would fail)
# Unknown verdict → log warn + continue (defensive fallback).
#
# Citation chain: sparrow-engine/scripts/ort-env.sh:167-168, docs/lessons.md:29,
# docs/tech_report/06_gotchas_and_constraints.md:17-25.
function Test-GpuQuality {
    param([string]$ResolvedFlavor)
    if ($ResolvedFlavor -ne 'gpu') {
        return
    }
    $probeQ = Resolve-Helper -Name 'probe_gpu_quality.ps1'
    . $probeQ
    if (-not (Get-Command -Name probe_gpu_quality -ErrorAction SilentlyContinue)) {
        Die 8 "probe_gpu_quality.ps1 did not define probe_gpu_quality."
    }
    # probe_gpu_quality writes pass|warn|fail to stdout; authoritative verdict
    # is in $env:SPARROW_ENGINE_GPU_QUALITY (4-state). Discard stdout.
    probe_gpu_quality | Out-Null
    switch ($env:SPARROW_ENGINE_GPU_QUALITY) {
        'ok'         { Write-Info "GPU quality: $($env:SPARROW_ENGINE_GPU_QUALITY_REASON)" }
        'sm_warn'    { Write-Warn2 "GPU quality: $($env:SPARROW_ENGINE_GPU_QUALITY_REASON)" }
        'cudnn_warn' { Write-Warn2 "GPU quality: $($env:SPARROW_ENGINE_GPU_QUALITY_REASON)" }
        'cudnn_err'  { Die 11 $env:SPARROW_ENGINE_GPU_QUALITY_REASON }
        default      { Write-Warn2 "GPU quality: unknown verdict (SPARROW_ENGINE_GPU_QUALITY=$($env:SPARROW_ENGINE_GPU_QUALITY))" }
    }
}

# ----- Mode auto-detect --------------------------------------------------

function Resolve-Mode {
    $modes = @()
    if ($Pip)    { $modes += 'pip' }
    if ($Cli)    { $modes += 'cli' }
    if ($Docker) { $modes += 'docker' }
    if ($modes.Count -gt 1) { Die 1 ("only one of -Pip / -Cli / -Docker; got: " + ($modes -join ', ')) }
    if ($modes.Count -eq 1) { return $modes[0] }
    # auto: prefer cli (binary distribution most general on Windows).
    return 'cli'
}

# ----- Release URL -------------------------------------------------------

function Get-ReleaseBase {
    # Returns the URL prefix. Defaults to the public GH Releases asset URL
    # (Phase E B-02 fix). Operator override via $env:SPARROW_ENGINE_RELEASE_BASE.
    $base = if ($env:SPARROW_ENGINE_RELEASE_BASE) { $env:SPARROW_ENGINE_RELEASE_BASE } else { $Script:DefaultReleaseBase }
    if (-not $base.EndsWith('/')) { $base += '/' }
    return $base
}

function Get-HelperBase {
    # Returns the URL prefix for helper scripts (probe.ps1, probe_gpu_quality.ps1).
    # Distinct from Get-ReleaseBase: helpers live in the tagged source tree,
    # not as release assets. E-R2-1 fix. Override via $env:SPARROW_ENGINE_HELPER_BASE.
    $base = if ($env:SPARROW_ENGINE_HELPER_BASE) { $env:SPARROW_ENGINE_HELPER_BASE } else { $Script:DefaultHelperBase }
    if (-not $base.EndsWith('/')) { $base += '/' }
    return $base
}

function Get-OsArchTuple {
    $arch = switch ($env:PROCESSOR_ARCHITECTURE) {
        'AMD64' { 'x86_64' }
        'ARM64' { 'aarch64' }
        default { Die 10 "unsupported PROCESSOR_ARCHITECTURE='$($env:PROCESSOR_ARCHITECTURE)'." }
    }
    return @{ Os = 'windows'; Arch = $arch }
}

# ----- State file --------------------------------------------------------

function Read-State {
    if (-not (Test-Path -LiteralPath $Script:StateFile)) { return $null }
    try {
        return (Get-Content -Raw -LiteralPath $Script:StateFile | ConvertFrom-Json)
    } catch {
        Write-Warn2 "installed.json unreadable: $($_.Exception.Message)"
        return $null
    }
}

function Write-State {
    param([string]$ResolvedFlavor, [string]$ResolvedMode)
    if ($DryRun) { Write-Info "[dry-run] would write state -> $Script:StateFile"; return }
    if (-not (Test-Path -LiteralPath $Script:UserSparrowEngineDir)) {
        New-Item -ItemType Directory -Force -Path $Script:UserSparrowEngineDir | Out-Null
    }
    $state = [pscustomobject]@{
        version       = $Script:SparrowEngineVersion
        flavor        = $ResolvedFlavor
        mode          = $ResolvedMode
        install_root  = $Script:InstallRoot
        installed_at  = (Get-Date).ToString('o')
    }
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Script:StateFile -Encoding UTF8
}

# ----- Cross-flavor refusal (T-3) ----------------------------------------

function Test-CrossFlavor {
    param([string]$ResolvedFlavor)
    $st = Read-State
    if (-not $st) { return }
    if ($st.flavor -ne $ResolvedFlavor -and -not $Reprobe) {
        Write-Err ("Sparrow Engine {0} flavor is already installed at {1} (version {2})." -f $st.flavor, $st.install_root, $st.version)
        Write-Err "The strict-flavor invariant (MT-4.1-2) does NOT support side-by-side install of CPU and GPU flavors."
        Write-Err ""
        Write-Err "To switch:"
        Write-Err "  sparrow-engine-install.ps1 -Uninstall ; sparrow-engine-install.ps1 -Flavor $ResolvedFlavor"
        Write-Err "  OR equivalently in one step:"
        Write-Err "  sparrow-engine-install.ps1 -Reprobe"
        exit 12
    }
    if ($st.flavor -ne $ResolvedFlavor -and $Reprobe -and -not $Yes) {
        # Non-interactive guard: `iwr ... | iex` and similar piped invocations
        # have no usable stdin. Without -Yes we cannot prompt; treat as user
        # abort (exit 2) per final_design.md § 2.10.
        if ([Console]::IsInputRedirected) {
            Die 2 "cross-flavor switch requires confirmation; pass -Yes to non-interactive sessions (iwr | iex)."
        }
        $msg = "Switch from $($st.flavor) to $ResolvedFlavor? [y/N] "
        try {
            $resp = Read-Host -Prompt $msg
        } catch {
            # PSHost without a console (e.g., scheduled task) throws here.
            Die 2 "cross-flavor switch requires confirmation; this session has no usable console. Pass -Yes to suppress the prompt."
        }
        if ($resp -notmatch '^[Yy]') {
            Write-Info "aborted by user."
            exit 0
        }
    }
    # Cross-flavor switch confirmed (either via -Yes or the prompt above).
    # Mirror sparrow-engine-install.sh::do_uninstall_silent — uninstall the existing
    # flavor before installing the new one, so we don't end up with a half-
    # mixed install. Skipped when state was already same-flavor (line above).
    if ($st.flavor -ne $ResolvedFlavor -and $Reprobe) {
        Write-Info "switching $($st.flavor) -> $ResolvedFlavor (reprobe); uninstalling existing first."
        Invoke-Uninstall
    }
    if ($st.flavor -eq $ResolvedFlavor -and $st.version -eq $Script:SparrowEngineVersion -and -not $Reinstall) {
        Write-Info ("Sparrow Engine {0} {1} already installed at {2}." -f $st.flavor, $st.version, $st.install_root)
        Write-Info "Pass -Reinstall to force same-flavor overwrite."
        exit 0
    }
}

# ----- Download with retry + sha256 verify -------------------------------

function Invoke-DownloadWithRetry {
    param([string]$Url, [string]$OutFile)
    $delay = 1
    for ($i = 0; $i -lt $Retries; $i++) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
            return
        } catch {
            $resp = $_.Exception.Response
            $code = if ($resp) { [int]$resp.StatusCode } else { 0 }
            # Permanent-failure HTTP codes — abort immediately, no retry.
            if ($code -in 401,403,404,410,451) {
                Die 4 "$Url returned $code — permanent, no retry."
            }
            if ($i -lt ($Retries - 1)) {
                Write-Warn2 ("retry {0}/{1} in {2}s ({3})" -f ($i + 1), $Retries, $delay, $_.Exception.Message)
                Start-Sleep -Seconds $delay
                $delay = $delay * 2
            }
        }
    }
    Die 4 "download of $Url failed after $Retries attempts."
}

function Test-Sha256 {
    param([string]$File, [string]$SidecarFile)
    $expectedLine = (Get-Content -LiteralPath $SidecarFile -ErrorAction Stop | Select-Object -First 1)
    $expected = ($expectedLine -split '\s+')[0]
    $actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLower()
    if ($expected.ToLower() -ne $actual) {
        Die 6 "sha256 mismatch on ${File}: expected $expected, got $actual."
    }
}

# ----- Mode dispatch -----------------------------------------------------

function Install-PipFlavor {
    param([string]$ResolvedFlavor)
    Write-Step "Installing Python wheel ($ResolvedFlavor)"
    $pkg = if ($ResolvedFlavor -eq 'gpu') { 'sparrow-engine-gpu' } else { 'sparrow-engine' }

    # Python >=3.11 floor (mirror sparrow-engine-install.sh:344-354 / CLAUDE.md
    # PyO3 0.25 invariant). Probe sys.version_info via subprocess; FileVersion
    # on python.exe is unreliable across 5.1/7+ and Windows-Store Python.
    $pythonCmd = Get-Command -Name python  -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { $pythonCmd = Get-Command -Name python3 -ErrorAction SilentlyContinue }
    if (-not $pythonCmd) {
        Die 8 "python (or python3) not found on PATH (required for -Pip mode)."
    }
    $pyExe = $pythonCmd.Path
    & $pyExe -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 5)" 2>$null
    $pyExit = $LASTEXITCODE
    if ($pyExit -eq 5) {
        $pyVer = & $pyExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        Die 5 "python $pyVer is too old; Sparrow Engine requires Python >=3.11."
    } elseif ($pyExit -ne 0) {
        Die 5 "python version check failed (exit $pyExit); Sparrow Engine requires Python >=3.11."
    }

    $hasUv  = [bool](Get-Command -Name uv  -ErrorAction SilentlyContinue)
    $hasPip = [bool](Get-Command -Name pip -ErrorAction SilentlyContinue)
    if (-not ($hasUv -or $hasPip)) {
        Die 8 "neither 'pip' nor 'uv' found on PATH; install Python >=3.11 + pip first."
    }
    $spec = "$pkg==$($Script:SparrowEngineVersion)"
    if ($DryRun) { Write-Info "[dry-run] would install $spec via $(if ($hasUv) { 'uv pip' } else { 'pip' })"; return }
    if ($hasUv) {
        Write-Info "uv pip install $spec"
        & uv pip install $spec
    } else {
        Write-Info "pip install $spec"
        & pip install $spec
    }
    if ($LASTEXITCODE -ne 0) { Die 1 "pip install exited $LASTEXITCODE." }
}

function Install-CliFlavor {
    param([string]$ResolvedFlavor)
    Write-Step "Installing CLI binary ($ResolvedFlavor)"
    $oa = Get-OsArchTuple
    $base = Get-ReleaseBase
    $tarball = "sparrow-engine-$ResolvedFlavor-$Script:SparrowEngineVersion-$($oa.Os)-$($oa.Arch).zip"
    $url = "$base$tarball"
    $sha = "$url.sha256"

    if ($DryRun) { Write-Info "[dry-run] would download $url + verify sha256 + extract to $Script:InstallRoot"; return }
    $staging = Join-Path $env:TEMP ("sparrow-engine-stage-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    try {
        $tarballPath = Join-Path $staging $tarball
        $shaPath = "$tarballPath.sha256"
        Invoke-DownloadWithRetry -Url $url -OutFile $tarballPath
        Invoke-DownloadWithRetry -Url $sha -OutFile $shaPath
        Test-Sha256 -File $tarballPath -SidecarFile $shaPath
        if (Test-Path -LiteralPath $Script:InstallRoot) {
            $bak = "$Script:InstallRoot.bak"
            if (Test-Path -LiteralPath $bak) { Remove-Item -Recurse -Force -LiteralPath $bak }
            Rename-Item -LiteralPath $Script:InstallRoot -NewName ([IO.Path]::GetFileName($bak))
        }
        New-Item -ItemType Directory -Force -Path $Script:InstallRoot | Out-Null
        Expand-Archive -LiteralPath $tarballPath -DestinationPath $Script:InstallRoot -Force
    } finally {
        if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
    }
}

function Install-DockerFlavor {
    param([string]$ResolvedFlavor)
    Write-Step "Pulling Docker image ($ResolvedFlavor)"
    if (-not (Get-Command -Name docker -ErrorAction SilentlyContinue)) {
        Die 8 "'docker' not found on PATH; install Docker Desktop first."
    }
    $tag = if ($ResolvedFlavor -eq 'cpu') { 'zhongqimiao/sparrow-engine-server:latest' } else { 'zhongqimiao/sparrow-engine-server-gpu:latest' }
    if ($DryRun) { Write-Info "[dry-run] would: docker pull $tag"; return }
    & docker pull $tag
    if ($LASTEXITCODE -ne 0) { Die 1 "docker pull exited $LASTEXITCODE." }
    Write-Info "Pulled $tag. To run: docker run --rm $tag --help"
}

# ----- $PROFILE rc-file edit (sentinel block) ----------------------------

function Update-ProfileRc {
    param([string]$ResolvedMode)
    if ($env:SPARROW_ENGINE_NO_MODIFY_PATH -eq '1') { Write-Info "SPARROW_ENGINE_NO_MODIFY_PATH=1 — skipping `$PROFILE edit."; return }
    if ($ResolvedMode -ne 'cli') { return }   # only CLI mode adds binaries to PATH
    $profilePath = $PROFILE
    if (-not (Test-Path -LiteralPath $profilePath)) {
        if ($DryRun) { Write-Info "[dry-run] would create `$PROFILE at $profilePath"; return }
        $profileDir = Split-Path -Parent $profilePath
        if (-not (Test-Path -LiteralPath $profileDir)) { New-Item -ItemType Directory -Force -Path $profileDir | Out-Null }
        New-Item -ItemType File -Force -Path $profilePath | Out-Null
    }
    $existing = Get-Content -Raw -LiteralPath $profilePath -ErrorAction SilentlyContinue
    if ($null -eq $existing) { $existing = '' }
    $startIdx = $existing.IndexOf($Script:SentinelStart)
    $endIdx   = $existing.IndexOf($Script:SentinelEnd)
    $newBlock = @"
$Script:SentinelStart
`$env:Path = "$Script:InstallRoot\bin;" + `$env:Path
$Script:SentinelEnd
"@
    if ($startIdx -ge 0 -and $endIdx -gt $startIdx) {
        $blockText = $existing.Substring($startIdx, ($endIdx - $startIdx) + $Script:SentinelEnd.Length)
        # Manual-edit detection: anything between the sentinels other than the canonical PATH line is "manual".
        # PS 5.1-compatible array-slice approach (Select-Object -SkipLast is PS6+).
        $blockLines = $blockText -split "`n"
        if ($blockLines.Count -gt 2) {
            $inner = ($blockLines[1..($blockLines.Count - 2)] -join "`n")
        } else {
            $inner = ''
        }
        $expected = "`$env:Path = `"$($Script:InstallRoot)\bin;`" + `$env:Path"
        if ($inner.Trim() -ne $expected.Trim() -and -not $ForceRcOverwrite) {
            Write-Err "manual edits detected inside sparrow-engine block in `$PROFILE ($profilePath); aborting rc edit."
            Write-Err "Edit `$PROFILE manually or re-run with -ForceRcOverwrite."
            exit 13
        }
        $newContent = $existing.Substring(0, $startIdx) + $newBlock + $existing.Substring($endIdx + $Script:SentinelEnd.Length)
    } else {
        $separator = if ($existing.Length -gt 0 -and -not $existing.EndsWith("`n")) { "`n" } else { '' }
        $newContent = $existing + $separator + $newBlock + "`n"
    }
    if ($DryRun) { Write-Info "[dry-run] would update `$PROFILE block at $profilePath"; return }
    Set-Content -LiteralPath $profilePath -Value $newContent -Encoding UTF8 -NoNewline
    Write-Info "updated `$PROFILE at $profilePath (re-open shell or `". `$PROFILE`" to pick up PATH)"
}

function Remove-ProfileRc {
    if (-not (Test-Path -LiteralPath $PROFILE)) { return }
    $existing = Get-Content -Raw -LiteralPath $PROFILE
    $startIdx = $existing.IndexOf($Script:SentinelStart)
    $endIdx   = $existing.IndexOf($Script:SentinelEnd)
    if ($startIdx -lt 0 -or $endIdx -le $startIdx) { return }
    if ($DryRun) { Write-Info "[dry-run] would strip sparrow-engine block from `$PROFILE"; return }
    $stripped = $existing.Substring(0, $startIdx) + $existing.Substring($endIdx + $Script:SentinelEnd.Length)
    Set-Content -LiteralPath $PROFILE -Value $stripped.TrimEnd() -Encoding UTF8
}

# ----- Uninstall ---------------------------------------------------------

function Invoke-Uninstall {
    Write-Step "Uninstalling Sparrow Engine"
    $st = Read-State
    if ($DryRun) {
        Write-Info "[dry-run] would remove $Script:InstallRoot + state file + `$PROFILE sparrow-engine block"
        return
    }
    if (Test-Path -LiteralPath $Script:InstallRoot) {
        Remove-Item -Recurse -Force -LiteralPath $Script:InstallRoot
    }
    if (Test-Path -LiteralPath $Script:StateFile) {
        Remove-Item -Force -LiteralPath $Script:StateFile
    }
    Remove-ProfileRc
    if ($st) { Write-Info ("removed Sparrow Engine {0} ({1})" -f $st.flavor, $st.version) } else { Write-Info "removed Sparrow Engine (no state file found; best-effort cleanup)" }
}

# ----- Main --------------------------------------------------------------

function Invoke-Main {
    if ($Help) {
        Write-Host "Usage: sparrow-engine-install.ps1 [-Pip|-Cli|-Docker] [-Flavor cpu|gpu|auto]"
        Write-Host "       [-Reinstall|-Reprobe|-Uninstall] [-ForceRcOverwrite]"
        Write-Host "       [-ProbeOnly] [-DryRun] [-Yes] [-Version X.Y.Z] [-Retries N]"
        Write-Host ""
        Write-Host "See docs/install.md for full reference."
        return
    }

    if ($Uninstall) { Invoke-Uninstall; return }

    $resolvedFlavor = Resolve-Flavor -Requested $Flavor
    Write-Info "resolved flavor: $resolvedFlavor"

    if ($ProbeOnly) {
        Write-Host $resolvedFlavor
        return
    }

    # Layer-2 quality probe FIRST — only fires when ResolvedFlavor='gpu'.
    # Dispatches $env:SPARROW_ENGINE_GPU_QUALITY: ok → silent; sm_warn / cudnn_warn →
    # log + continue; cudnn_err → Die 11. Must run BEFORE Test-CrossFlavor:
    # that function calls Invoke-Uninstall on cross-flavor `-Reprobe` (line
    # 251), destroying the existing install. If the GPU quality gate fails
    # AFTER the destructive uninstall, the user is left with NO install.
    # (impl-af R3 Inq F-R3-1 fix; mirror of sparrow-engine-install.sh.)
    Test-GpuQuality -ResolvedFlavor $resolvedFlavor

    Test-CrossFlavor -ResolvedFlavor $resolvedFlavor

    $resolvedMode = Resolve-Mode
    Write-Info "resolved mode: $resolvedMode"

    switch ($resolvedMode) {
        'pip'    { Install-PipFlavor    -ResolvedFlavor $resolvedFlavor }
        'cli'    { Install-CliFlavor    -ResolvedFlavor $resolvedFlavor }
        'docker' { Install-DockerFlavor -ResolvedFlavor $resolvedFlavor }
        default  { Die 1 "unknown mode: $resolvedMode" }
    }

    Write-State -ResolvedFlavor $resolvedFlavor -ResolvedMode $resolvedMode
    Update-ProfileRc -ResolvedMode $resolvedMode
    Write-Step "Done. Flavor: $resolvedFlavor; Mode: $resolvedMode."
}

# ----- Entrypoint --------------------------------------------------------

$ErrorActionPreference = 'Stop'

# SIGINT (Ctrl-C) -> exit 2 (final_design § 2.10).
# PowerShell traps Ctrl-C as a TerminatingError; we wrap Invoke-Main in
# try/catch and translate the [System.Management.Automation.PipelineStoppedException]
# / OperationCanceledException paths to exit 2.
try {
    Invoke-Main
    exit 0
} catch [System.Management.Automation.PipelineStoppedException] {
    Write-Err "user aborted (Ctrl-C)."
    exit 2
} catch [System.OperationCanceledException] {
    Write-Err "user aborted (Ctrl-C)."
    exit 2
} catch {
    # Unhandled errors propagate as exit 1 unless a prior `exit <N>` already fired.
    Write-Err $_.Exception.Message
    exit 1
}
