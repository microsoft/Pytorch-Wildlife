# Installing Sparrow Engine

> **Banner**: URLs marked `https://sparrow-engine.example/...` are RFC-2606
> placeholder hostnames pending public hosting per
> `docs/release_dev_plan.md § R1` + `§ R3`. Today's supported lead form
> is `bash installer/sparrow-engine-install.sh` from a local clone (CWD = repo
> root); the `curl ... | sh` and `iwr ... | iex` one-liners are
> documented for post-R3 once GH Releases publish the canonical URLs.
> Running them today fails because the wrapper resolves `probe.sh`
> relative to `dirname "$0"`, and `$0` becomes `bash` (not the script
> path) under stdin-piped invocation.
>
> Source of truth for the install design: `docs/design/phase4.1-install-selector/final_design.md`.

---

## Quickstart

One probe, one flavor, one tarball-or-wheel-or-image. The wrapper detects
NVIDIA hardware once, then installs the matching CPU or GPU build into
`~/.sparrow_engine/`.

| Platform | Command (today, supported) | Command (post-R3, deferred) | Result |
|---|---|---|---|
| Linux x86_64 | `bash installer/sparrow-engine-install.sh` | `curl --proto '=https' --tlsv1.2 -LsSf https://sparrow-engine.example/install.sh \| sh` ¹ | CLI tarball into `~/.sparrow_engine/bin`; pip wheel into the active Python environment |
| macOS arm64 (Apple Silicon) | `bash installer/sparrow-engine-install.sh` | same `curl ... \| sh` form ¹ | CPU flavor only (Apple Silicon = no NVIDIA) |
| macOS x86_64 (Intel) | `bash installer/sparrow-engine-install.sh` | same `curl ... \| sh` form ¹ | CPU flavor only (Intel + eGPU is not auto-detected) |
| Windows x86_64 (PowerShell) | `installer\sparrow-engine-install.ps1` | `iwr https://sparrow-engine.example/install.ps1 -useb \| iex` ¹ | Same probe; binary path `%USERPROFILE%\.sparrow_engine\bin` |

¹ Post-R3 form depends on GH Releases publishing the canonical
`sparrow-engine.example` URL (per `docs/release_dev_plan.md § R3`) AND the
wrapper being self-contained (probe.sh inlined or fetched separately).
Today the stdin-piped form fails: `dirname "$0"` resolves to `.` and
`probe.sh` is not co-located. Use the today-form above; see
Troubleshooting → "stdin-piped install (`curl | sh`)" for the full
explanation.

`<!-- TODO: replace sparrow-engine.example with canonical sparrow-engine URL when public hosting fires per release_dev_plan.md § R3 -->`

Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
for the HTTP-server image instead of CLI + wheel. Pass `--cli` or
`--pip` to install only one consumer.

### Inspect before run

```bash
curl -LsSf https://sparrow-engine.example/install.sh | less    # post-R3
```

### Version-pinned install URL

```bash
curl -LsSf https://sparrow-engine.example/0.1.0/install.sh | sh    # post-R3
```

### Disable PATH modification

```bash
SPARROW_ENGINE_NO_MODIFY_PATH=1 curl -LsSf https://sparrow-engine.example/install.sh | sh    # post-R3
```

---

## What the wrapper does

1. Parses CLI args (`--flavor`, `--cli`, `--pip`, `--docker`, `--reinstall`, `--reprobe`, `--uninstall`, `--dry-run`, `--probe-only`). Run `sparrow-engine-install.sh --help` for the full list.
2. Runs the layer-1 hardware probe (`installer/probe.sh`): checks `nvidia-smi`,
   `libcuda.so.1`, `/dev/nvidia0`, and (on WSL2) `/dev/dxg`. Returns `cpu` or `gpu`.
3. If layer-1 returned `gpu`, runs the layer-2 quality probe
   (`installer/probe_gpu_quality.sh`): verifies cuDNN ≥9.10 + driver
   version. Returns `pass`, `warn`, or `fail`.
4. Picks the install mode (`pip`, `cli`, `docker`) — auto-detected from CWD
   / `PATH` / installed packages, or supplied explicitly.
5. Downloads the matching tarball (CLI), wheel (pip), or image (docker)
   from `$SPARROW_ENGINE_RELEASE_BASE` and verifies sha256.
6. Extracts into `~/.sparrow_engine/` (CLI) or installs via `pip` (pip) or
   `docker pull` (docker).
7. Edits the user's rc-file via the conda-style sentinel block
   `# >>> sparrow_engine >>>` ... `# <<< sparrow_engine <<<` (idempotent).
8. Writes state to `~/.sparrow_engine/installed.json`.

The body wraps in `main()` with `main "$@"` as the literal last line so
that a truncated `curl | sh` download cannot execute a partial script
(rustup-init.sh + ollama install.sh pattern).

---

## Hardware requirements

### CPU flavor

| Item | Requirement | Source |
|---|---|---|
| OS | Linux x86_64 (glibc ≥2.35), macOS ≥12 arm64 or x86_64, Windows 10+ x86_64 | `final_design.md § 2.5` |
| Python | ≥3.11 (for `--pip` mode) | `sparrow-engine-python` PyO3 0.25 floor |
| Disk | ~100 MB free for tarball install | `final_design.md § 2.6` |

### GPU flavor (NVIDIA only)

| Item | Requirement | Source |
|---|---|---|
| GPU | NVIDIA, compute capability ≥7.5 (Turing or later) | ORT CUDA EP minimum |
| NVIDIA driver | ≥550.x (CUDA 12.6 runtime) | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base image |
| CUDA runtime | 12.6 | Phase 3.8 Phase C lock |
| cuDNN | **≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89 — see Troubleshooting) | `docs/lessons.md:29`, `sparrow-engine/scripts/ort-env.sh:167-168`, `docs/tech_report/06_gotchas_and_constraints.md:17-25` |
| Disk | ~3 GB free (Docker GPU image), ~300 MB (CLI tarball with bundled ORT) | `final_design.md § 2.6` |

**macOS GPU**: not supported. macOS arm64 and x86_64 install CPU only.
Intel + eGPU is rare and not auto-detected; pass `--flavor gpu` to force
(install will fail at first inference).

---

## What gets installed

| Mode | flavor=cpu | flavor=gpu |
|---|---|---|
| `pip` | `pip install sparrow-engine` (depends on `onnxruntime>=1.25.1`) | `pip install sparrow-engine-gpu` (depends on `onnxruntime-gpu>=1.25.1`) |
| `cli` | `sparrow-engine-cpu-{ver}-{os}-{arch}.tar.gz` (~30 MB compressed) — extracts `bin/spe` + bundled `lib/libonnxruntime.so.1` + `lib/libsparrow_engine.so` + `share/sparrow-engine/{LICENSE,README.md,manifest-schema.toml,wheels/<flavor>.whl}` | `sparrow-engine-gpu-{ver}-{os}-{arch}.tar.gz` (~270 MB compressed) — same layout, with bundled GPU ORT (~200 MB) |
| `docker` | `docker pull ghcr.io/microsoft.example/sparrow-engine:cpu` (167 MB image) | `docker pull ghcr.io/microsoft.example/sparrow-engine:gpu` (3.67 GB image; bundles cuDNN + CUDA) |

`<!-- TODO: replace ghcr.io/microsoft.example with canonical org when public hosting fires per release_dev_plan.md § R3 -->`

The strict-flavor invariant (MT-4.1-2, commit `d2e2202`) holds: a host
runs ONE flavor. The wrapper refuses cross-flavor install (exit 12)
unless `--reprobe` is passed. See `Switching flavors` below.

### CPU wheel vs GPU wheel

Both wheels are imported as `import sparrow_engine` (not `import sparrow_engine_gpu`). The
distribution name differs (`sparrow-engine` vs `sparrow-engine-gpu`), the import name does
not. The wheels do NOT bundle ORT or CUDA libraries — `pip` resolves
them from PyPI (CPU: `onnxruntime`; GPU: `onnxruntime-gpu`).

---

## Flavor selection (override the probe)

The probe is the default. To override:

```bash
# Skip probe; install CPU.
bash installer/sparrow-engine-install.sh --flavor cpu

# Skip probe; install GPU.
bash installer/sparrow-engine-install.sh --flavor gpu

# Run probe (default; same as omitting --flavor).
bash installer/sparrow-engine-install.sh --flavor auto
```

Or via env var:

```bash
SPARROW_ENGINE_INSTALL_FLAVOR=gpu bash installer/sparrow-engine-install.sh
```

`--flavor` takes precedence over `SPARROW_ENGINE_INSTALL_FLAVOR`. `--flavor auto`
explicitly runs the probe even if `SPARROW_ENGINE_INSTALL_FLAVOR` is set.

---

## Per-consumer install paths

### CLI binary

```bash
bash installer/sparrow-engine-install.sh --cli
```

Extracts the tarball to `~/.sparrow_engine/`. Rc-file gets `export PATH="$HOME/.sparrow_engine/bin:$PATH"`
inside the conda-style sentinel block. Verify:

```bash
$ spe --help
$ spe device
cuda:0    # or `cpu` for CPU flavor
```

### Python wheel

```bash
bash installer/sparrow-engine-install.sh --pip
```

Installs the wheel into the active Python environment (the venv / conda
env / system site-packages currently sourced) by invoking `uv pip install`
(if `uv` is on PATH) or `pip install`. CPU flavor installs the `sparrow-engine`
distribution; GPU flavor installs `sparrow-engine-gpu`; both import as
`import sparrow_engine`. The wrapper does NOT create a venv under `~/.sparrow_engine/` —
the active env at the time of invocation is the destination. Verify:

```bash
$ python -c "import sparrow_engine; print(sparrow_engine.active_device())"
cuda:0    # or `cpu`
```

CPU wheel + GPU wheel CANNOT coexist in the same Python environment
(`Provides-Dist` + `Conflicts-Dist` are advisory in pip ≥22 —
operator-discipline, not mechanical guardrail). The wrapper's
cross-flavor refusal (exit 12) prevents accidental mixing on a single
host.

**Note on `--uninstall`**: `bash installer/sparrow-engine-install.sh --uninstall`
removes `~/.sparrow_engine/` (CLI tarball + state file + rc-file blocks) but does
NOT call `pip uninstall`. To remove the wheel from the active Python
env, run `pip uninstall sparrow-engine` (or `sparrow-engine-gpu`) explicitly.

### Docker image

```bash
bash installer/sparrow-engine-install.sh --docker
```

The wrapper PRINTS the recommended `docker pull` + `docker run`
commands; it does NOT execute Docker on the user's behalf. The user is
responsible for running the daemon.

GPU image requires NVIDIA Container Toolkit + `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`-compatible host.

---

## Air-gapped / offline install

> **Status (2026-05-08)**: automated air-gap flags (`--offline`,
> `SPARROW_ENGINE_OFFLINE_TARBALL`, `SPARROW_ENGINE_LOCAL_ORT_GPU_WHEEL`) are NOT yet
> implemented (TODO; tracked at `docs/ideas.md`). The implementation
> is deferred until public-release work fires per
> `docs/release_dev_plan.md § R3` (the same gate that publishes the
> canonical release URLs); this matches the dev-first / release-last
> ordering captured in `feedback_dev_first_release_last.md`.
> The manual workflows below are the supported path today.

### CPU air-gapped (manual tarball path)

```bash
# On networked host: fetch tarball + sha256 sidecar (post-R3 URLs).
curl -LsSf https://sparrow-engine.example/0.1.0/sparrow-engine-cpu-0.1.0-linux-x86_64.tar.gz \
    -o sparrow-engine-cpu.tar.gz
curl -LsSf https://sparrow-engine.example/0.1.0/sparrow-engine-cpu-0.1.0-linux-x86_64.tar.gz.sha256 \
    -o sparrow-engine-cpu.tar.gz.sha256
# sneakernet both files.

# On air-gapped host: verify, then install via SPARROW_ENGINE_RELEASE_BASE pointing
# at a local file:// URL that contains the staged artifacts.
sha256sum -c sparrow-engine-cpu.tar.gz.sha256
mkdir -p /tmp/sparrow-engine-release/v0.1.0
cp sparrow-engine-cpu.tar.gz sparrow-engine-cpu.tar.gz.sha256 /tmp/sparrow-engine-release/v0.1.0/
SPARROW_ENGINE_RELEASE_BASE=file:///tmp/sparrow-engine-release/v0.1.0/ \
    bash installer/sparrow-engine-install.sh --cli --flavor cpu
```

The vendored wheel inside `share/sparrow-engine/wheels/` handles the Python
`--pip` path without PyPI once the CLI tarball is extracted (manual
`pip install $HOME/.sparrow_engine/share/sparrow-engine/wheels/<flavor>.whl`).

### GPU air-gapped (docker save/load — recommended; M-1)

The genuine GPU air-gap path is `docker save`. The bundled ORT in the
GPU tarball is NOT sufficient for true air-gap because cuDNN + CUDA
libraries must come from the system or a containing image; `docker save`
of `sparrow-engine:gpu` includes all three (cuDNN + CUDA + ORT) via the
`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` base image.

```bash
# On networked host:
docker pull ghcr.io/microsoft.example/sparrow-engine:gpu                            # post-R3
docker save ghcr.io/microsoft.example/sparrow-engine:gpu | gzip > sparrow-engine-gpu-image-0.1.0.tar.gz
# sneakernet (~3.67 GB compressed)

# On air-gapped host (must already have NVIDIA driver + Container Toolkit):
gunzip -c sparrow-engine-gpu-image-0.1.0.tar.gz | docker load
docker run --gpus all ghcr.io/microsoft.example/sparrow-engine:gpu --help
```

### GPU air-gapped (manual tarball path; advanced)

For HPC users who prefer not to use Docker. The user is responsible for
supplying cuDNN ≥9.10 + CUDA 12.6 from the system:

```bash
# On networked host: download the GPU tarball + sidecar (~270 MB; bundles ORT).
curl -LsSf https://sparrow-engine.example/0.1.0/sparrow-engine-gpu-0.1.0-linux-x86_64.tar.gz \
    -o sparrow-engine-gpu.tar.gz
curl -LsSf https://sparrow-engine.example/0.1.0/sparrow-engine-gpu-0.1.0-linux-x86_64.tar.gz.sha256 \
    -o sparrow-engine-gpu.tar.gz.sha256
# sneakernet

# On air-gapped host (must already have cuDNN ≥9.10 + CUDA 12.6):
sha256sum -c sparrow-engine-gpu.tar.gz.sha256
mkdir -p /tmp/sparrow-engine-release/v0.1.0
cp sparrow-engine-gpu.tar.gz sparrow-engine-gpu.tar.gz.sha256 /tmp/sparrow-engine-release/v0.1.0/
SPARROW_ENGINE_RELEASE_BASE=file:///tmp/sparrow-engine-release/v0.1.0/ \
    bash installer/sparrow-engine-install.sh --cli --flavor gpu
```

---

## Switching flavors after install

Two flags handle the two scenarios:

| Flag | Use case | Semantics |
|---|---|---|
| `--reinstall` | Same-flavor force-overwrite (corrupt install recovery) | Wrapper deletes the old install via the `--uninstall` flow then runs the install path; same-fs atomic rename; backup at `$PREFIX/.bak-<prev-version>` cleaned on success. Exit 0. |
| `--reprobe` | Cross-flavor switch (e.g., user added GPU after CPU install) **OR** driver-upgrade re-probe | Wrapper re-runs the hardware probe, then calls `--uninstall`, then runs the install path on the newly-probed flavor. `[y/N]` confirmation unless `-y`. Exit 0. |

### Recommended workflow

```bash
# Two-step (recommended; explicit)
bash installer/sparrow-engine-install.sh --uninstall
bash installer/sparrow-engine-install.sh --flavor=gpu

# One-step shortcut
bash installer/sparrow-engine-install.sh --reprobe                  # cross-flavor switch
bash installer/sparrow-engine-install.sh --reprobe --flavor=gpu     # explicit pin (skip probe)

# Same-flavor force-overwrite (corrupt install recovery)
bash installer/sparrow-engine-install.sh --reinstall
```

### Cross-flavor refusal (default)

When the user invokes the wrapper with a different `--flavor` than the
existing install AND has not passed `--reprobe`, the wrapper refuses
with **exit 12**:

```
$ bash installer/sparrow-engine-install.sh --flavor=gpu
ERROR: sparrow-engine CPU flavor is already installed at ~/.sparrow_engine (version 0.1.0).
The strict-flavor invariant (MT-4.1-2, commit d2e2202) does NOT support
side-by-side install of CPU and GPU flavors.

To switch flavors, choose one of:

  1. Two-step (explicit):
       bash installer/sparrow-engine-install.sh --uninstall
       bash installer/sparrow-engine-install.sh --flavor=gpu

  2. One-step (after [y/N] confirm):
       bash installer/sparrow-engine-install.sh --reprobe

Same-flavor force-overwrite uses --reinstall:
       bash installer/sparrow-engine-install.sh --reinstall

Exit 12.
```

---

## Re-probing after driver upgrade

If you installed CPU on a server that later gained an NVIDIA driver +
cuDNN, the wrapper does NOT auto-detect. Sticky-by-default; `--reprobe`
is the explicit escape hatch:

```bash
bash installer/sparrow-engine-install.sh --reprobe
```

This re-runs the layer-1 + layer-2 probes, asks `[y/N]` to confirm the
new flavor, and then uninstalls + reinstalls. Use `-y` to skip the
prompt.

---

## Error message catalog (exit codes 0–14)

| Code | Meaning | Typical cause |
|---|---|---|
| 0 | Success | Install completed |
| 1 | Generic error (`set -eu` / `\|\| exit 1` propagation; argv parse failures fall through here) | Catch-all |
| 2 | User aborted (Ctrl-C) | User pressed Ctrl-C during install |
| 3 | Probe disagreement (override conflicts with hardware) | `--flavor gpu` on a host without NVIDIA hardware |
| 4 | Network failure (after retries) | `curl` retried 3× with 1s/2s/4s exponential backoff; permanent 4xx (401/403/404/410/451) aborts |
| 5 | Python too old (<3.11) | `--pip` mode found Python <3.11 |
| 6 | sha256 verification failed | Tarball / wheel sha256 mismatch (transit corruption or tampering) |
| 7 | Disk space insufficient | `~/.sparrow_engine/` partition lacks free space for download + extract |
| 8 | Required tool missing (curl/tar/docker/pip) | Tool absent from `PATH` |
| 9 | Platform/flavor combination not supported (e.g., GPU on macOS) | macOS + `--flavor gpu` |
| 10 | OS not supported | Wrapper detected an OS outside Linux x86_64, macOS, Windows x86_64 |
| 11 | cuDNN < 9.10 (driver layer-2 probe failure) | cuDNN 9.8 detected → BLOCKING. See Troubleshooting → cuDNN 9.8 detected |
| 12 | Cross-flavor install attempted without `--reprobe` | User ran `--flavor=gpu` while CPU is installed |
| 13 | Manual rc-file edit detected without `--force-rc-overwrite` | User edited content inside the `# >>> sparrow_engine >>>` block |
| 14 | `SPARROW_ENGINE_RELEASE_BASE` unset and no public hosting yet | Pre-R3; user did not set `SPARROW_ENGINE_RELEASE_BASE` to a `file://` or local URL |

---

## Troubleshooting

### stdin-piped install (`curl | sh` / `iwr | iex`) fails

The wrapper resolves `probe.sh` relative to `dirname "$0"`. When invoked
via `curl ... | sh`, `$0` is `bash` (not the script path) and `probe.sh`
is not at `./probe.sh`, so the wrapper exits with `probe.sh not found`
(exit 1). The PowerShell wrapper has the same shape with `iwr | iex`.

**Workaround today**: use the local form. Clone the Sparrow Engine repo (or
download the `installer/` directory) and run
`bash installer/sparrow-engine-install.sh` from the repo root.

**Future**: post-R3 (per `docs/release_dev_plan.md § R3`), the curl|sh
one-liner is expected to work either by (a) inlining `probe.sh` +
`probe_gpu_quality.sh` into the wrapper as heredocs, or (b) fetching
them inline from the published release URL. Tracked at `docs/ideas.md`.

### Probe failed (no flavor detected; or wrong flavor)

The layer-1 probe checks (in order): `nvidia-smi`, `libcuda.so.1` via
`ldconfig`, `/dev/nvidia0`, and (WSL2 only) `/dev/dxg`. If all fail, it
returns `cpu`. To diagnose:

```bash
bash installer/sparrow-engine-install.sh --probe-only
# prints the resolved flavor and the SPARROW_ENGINE_DETECTED_PROBE_REASON line.
```

The Windows wrapper exposes the same flow via `sparrow-engine-install.ps1
-ProbeOnly`. To inspect the probe directly without invoking the wrapper:

```bash
. installer/probe.sh
probe_cuda
echo "$SPARROW_ENGINE_DETECTED_FLAVOR $SPARROW_ENGINE_DETECTED_PROBE_REASON"
```

If you have NVIDIA hardware but the probe returns `cpu`, check:

- `nvidia-smi` is on `PATH` and exits 0
- `ldconfig -p | grep libcuda.so.1` returns a path
- `/dev/nvidia0` exists (or `/dev/dxg` on WSL2)
- For containers without `/dev/nvidia0` (vfio passthrough), pass
  `--flavor gpu` to override

### Permission denied (writing to `~/.sparrow_engine/` or rc-file)

The wrapper writes to `~/.sparrow_engine/` and edits any pre-existing rc files
among `~/.bashrc` (always; created if missing), `~/.zshrc`,
`~/.bash_profile`, and `~/.profile`. All must be writable by the user.
The wrapper does NOT use `sudo`. Fish (`~/.config/fish/config.fish`) is
NOT supported by the wrapper — fish users should manually add `~/.sparrow_engine/bin`
to `$fish_user_paths` (e.g., `fish_add_path ~/.sparrow_engine/bin`).

If your home directory is read-only or quota-locked:

```bash
SPARROW_ENGINE_PREFIX=/tmp/sparrow-engine-install bash installer/sparrow-engine-install.sh
```

### sha256 mismatch (exit 6)

Re-download the tarball (transit corruption) or the sha256 sidecar
(unlikely). If the mismatch persists across multiple downloads, the
release page may have been tampered with — sha256 alone defends against
transit corruption only; without GPG signing on `SHA256SUMS`, a
compromised release page can rewrite both files. Audit the release-page
URL against the official `release_dev_plan.md § R3` reference.

### cuDNN 9.8 detected — STOP (exit 11)

cuDNN 9.8 has a Conv-engine bug on sm_89 (RTX 6000 Ada) that breaks
SpeciesNet inference (asymmetric padding; sources: `docs/lessons.md:29`,
`docs/tech_report/06_gotchas_and_constraints.md:17-25`). The wrapper
refuses to install GPU flavor and prints:

```
WARN: cuDNN 9.8 has a Conv-engine bug on sm_89 that breaks SpeciesNet
inference. Installing CPU flavor instead.

Override (NOT recommended — will fail at first SpeciesNet inference):
  bash installer/sparrow-engine-install.sh --flavor gpu

Fix the cuDNN floor:
  uv pip install --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'
Then re-run.
```

PyTorch and TensorFlow wheels both bundle cuDNN 9.8 — out-of-the-box dev
environments commonly hit this. The fix is to install
`nvidia-cudnn-cu12>=9.10` and source the sparrow-engine env so cuDNN 9.10 is
loaded ahead of the bundled 9.8.

### Driver too old (exit 11 path b)

The wrapper requires NVIDIA driver ≥550.x (CUDA 12.6 runtime). Older
drivers return `pass` from the cuDNN check but fail at first inference
with `CUDA error: no kernel image is available for execution on the
device`. The fix is `apt install nvidia-driver-550` (or your distro's
equivalent) + reboot.

### Probe disagrees with `--flavor` (exit 3 — reserved)

The wrapper does not currently fire exit 3 — passing `--flavor gpu` on
a CPU host will install the GPU artifacts and fail at first inference.
If the user wants to force GPU on a build host targeting a different
machine, pass `--flavor gpu` (or set `SPARROW_ENGINE_INSTALL_FLAVOR=gpu`); be
aware that runtime CUDA / cuDNN diagnostics surface only at first
inference, not at install time.

If the probe is reporting `cpu` on a host you believe has a working
GPU, see "Probe failed" above.

### `SPARROW_ENGINE_RELEASE_BASE` unset (exit 14)

Pre-R3, public hosting has not fired. Set `SPARROW_ENGINE_RELEASE_BASE` to a
local clone or a private mirror:

```bash
SPARROW_ENGINE_RELEASE_BASE=file:///tmp/sparrow-engine-release/v0.1.0/ \
    bash installer/sparrow-engine-install.sh
```

Or run the wrapper from a local clone of the Sparrow Engine repo (the wrapper
falls back to `installer/`-relative paths when invoked locally without
`$SPARROW_ENGINE_RELEASE_BASE`).

### Manual rc-file edit detected (exit 13)

User edited content inside the `# >>> sparrow_engine >>>` ... `# <<< sparrow_engine <<<`
block. The wrapper aborts to avoid clobbering manual changes. Either:

- Move the manual edits OUTSIDE the sentinel block.
- Pass `--force-rc-overwrite` to discard the manual edits and rewrite
  the block.

---

## What the wrapper writes (footprint listing)

| Path | Purpose | Removed by `--uninstall`? |
|---|---|---|
| `~/.sparrow_engine/bin/spe` (or `spe.exe`) | CLI binary (CLI mode) | Yes |
| `~/.sparrow_engine/lib/libonnxruntime.so.1` | Bundled ORT (CLI mode) | Yes |
| `~/.sparrow_engine/lib/libsparrow_engine.so` | sparrow-engine cdylib (CLI mode) | Yes |
| `~/.sparrow_engine/include/sparrow_engine.h` | C header (CLI mode) | Yes |
| `~/.sparrow_engine/share/sparrow-engine/{LICENSE,README.md,manifest-schema.toml,wheels/}` | Tarball share (CLI mode) | Yes |
| `~/.sparrow_engine/installed.json` | State file (flavor + version + mode + install timestamp) | Yes |
| `~/.bashrc` (always) + `~/.zshrc` / `~/.bash_profile` / `~/.profile` (only if pre-existing) | Inserts `# >>> sparrow_engine >>>` ... `# <<< sparrow_engine <<<` block | Yes (block removed; rest of rc-file untouched) |
| Active Python env site-packages: `sparrow-engine` (or `sparrow-engine-gpu`) wheel + transitive deps (pip mode) | Wheel install destination — the env active at `--pip` invocation time | **No** — `--uninstall` does NOT call `pip uninstall`. Run `pip uninstall sparrow-engine` (or `sparrow-engine-gpu`) explicitly to drop the wheel. |

The sentinel block is conda-style — supports multi-line block changes
between releases without breaking idempotency:

```
# >>> sparrow_engine >>>
export PATH="$HOME/.sparrow_engine/bin:$PATH"
# <<< sparrow_engine <<<
```

`SPARROW_ENGINE_NO_MODIFY_PATH=1` skips the rc-file edit entirely.

---

## Per-platform notes

### Linux x86_64

- glibc ≥2.35 required (the spe CLI binary builds against glibc 2.35).
- For systems with glibc <2.35: use the Docker image instead
  (`--docker` mode).
- `nvidia-smi`, `libcuda.so.1`, `/dev/nvidia0` are the layer-1 probe
  signals. WSL2 detection uses `/dev/dxg`.

### macOS arm64 (Apple Silicon)

- CPU flavor only (no NVIDIA GPU support).
- Install path is `~/.sparrow_engine/`; rc-file is `~/.zshrc` (default zsh shell
  on macOS ≥10.15).
- For training workloads on Apple Silicon, use a Linux GPU host; sparrow-engine
  is inference-only.

### macOS x86_64 (Intel)

- CPU flavor only by default. Intel + eGPU is rare and not auto-detected.
- Pass `--flavor gpu` to force (install will fail at first inference if
  no NVIDIA hardware is reachable).

### Windows x86_64

- PowerShell wrapper: `installer\sparrow-engine-install.ps1`.
- Install path is `%USERPROFILE%\.sparrow_engine\`.
- PATH update via `[Environment]::SetEnvironmentVariable("Path", …, "User")`
  (sentinel pattern N/A on Windows; the env-var API is the user-scope
  PATH directly).
- Layer-1 probe: `nvidia-smi.exe` + `nvcuda.dll` via `where` lookup.
- Layer-2 probe: cuDNN ≥9.10 verified via DLL version check.

---

## References

- Design: `docs/design/phase4.1-install-selector/final_design.md` (CONVERGED 2026-05-07)
- DX canonical: `docs/design/phase4.1-install-selector/round_04/dx-architect_proposal.md`
- Packaging canonical: `docs/design/phase4.1-install-selector/round_04/packaging-architect_proposal.md`
- Manual test plan: `docs/review/phase4-manual-test/round_01/manual_test_plan.md` (Phase 4.1 §1.5–§1.11; subsumes design's §1.0 + §1.5–§1.16 coverage outline)
- Release deferral mechanism: `docs/release_dev_plan.md § R1` (PyPI), `§ R3` (GH Releases policy)
- cuDNN floor sources: `docs/lessons.md:29`, `sparrow-engine/scripts/ort-env.sh:167-168`, `docs/tech_report/06_gotchas_and_constraints.md:17-25`
- Strict-flavor invariant: MT-4.1-2 (commit `d2e2202`)
