# Sparrow Engine — User Manual

**For**: anyone using Sparrow Engine (you, teammates, future maintainers).
**Style**: plain words, bite-size sections, one diagram per topic.

**Status**: Sparrow Engine development signed off (Phase 1 → Phase 4.4). Sparrow Studio Web integration shipped 2026-05-14..16. Sparrow Studio Local integration is the active downstream effort on the Windows machine.

---

## Reading guide — words you'll see a lot

Plain definitions before the first technical sentence.

| Term | Plain meaning |
|---|---|
| **sparrow-engine** | The whole system: a Rust library that loads ML models and runs them on images and audio. |
| **ONNX** | A common file format for trained ML models. Sparrow Engine only loads ONNX, nothing else. |
| **ORT** (onnxruntime) | The C++ library that actually executes ONNX models. Sparrow Engine wraps it; ORT does the math. |
| **CUDA EP / CPU EP** | "Execution provider" — the ORT module that runs the model on a GPU (CUDA EP) or CPU (CPU EP). |
| **cuDNN** | NVIDIA's deep-learning library that ORT's CUDA EP needs. Must be 9.10+ on RTX 6000 Ada. |
| **manifest** | A small TOML file (one per model) that tells sparrow-engine how to run the model — input size, normalization, postprocessing, labels. |
| **detector** | A model that finds bounding boxes around things in an image (e.g., MegaDetector v6). |
| **classifier** | A model that takes a single image (or a crop) and returns a label + score. |
| **pipeline** | Detector → crop each box → classifier on each crop. |
| **flavor** | One of two builds of sparrow-engine: `cpu` (no GPU code) or `gpu` (CUDA EP compiled in). |
| **cdylib** | A shared library file: `libsparrow_engine.so` on Linux, `sparrow_engine.dll` on Windows, `libsparrow_engine.dylib` on macOS. Both flavors ship this file under the same name. |
| **C ABI / FFI** | "Foreign Function Interface" — the C-compatible function set that other languages (C#, Python via PyO3) call. Sparrow Engine exports 32 `sparrow_engine_*` functions. |
| **PyO3** | The Rust library that builds Python bindings. The sparrow-engine Python wheel uses it. |
| **csbindgen / cbindgen** | Auto-generators that turn Rust function signatures into a C# `NativeMethods.g.cs` file (csbindgen) and a C `sparrow_engine.h` header (cbindgen). |
| **NCHW** | Tensor layout: Batch × Channels × Height × Width. Sparrow Engine mandates this; NHWC models must be re-exported. |
| **NMS** | Non-Maximum Suppression — the cleanup step that removes duplicate detector boxes. Must live in the ONNX graph, not in sparrow_engine. |
| **letterbox** | Resize-with-padding. Keeps aspect ratio by adding gray bars. MegaDetector's preprocessing. |
| **wheel** | A Python package file (`.whl`). Sparrow Engine ships two: `sparrow-engine` (CPU) and `sparrow-engine-gpu` (GPU). Both import as `sparrow-engine`. |
| **idempotent** | Running it twice has the same effect as running it once. |

---

## Top-level overview

```
                       sparrow-engine (Rust workspace, 7 crates)                                                       
                                   │                                                                                   
       ┌───────────────────────────┼───────────────────────────┐                                                       
       │                           │                           │                                                       
       v                           v                           v                                                       
  sparrow-engine-types               sparrow-engine-core                  sparrow-engine-cpu / sparrow-engine-gpu      
  (shared data              (shared logic              (engine flavors,                                                
   types — no               — no ORT, no               each ships libsparrow_engine.so                                 
   ORT, no CUDA)             CUDA)                      with 32 sparrow_engine_* exports)                              
                                                              │                                                        
                              ┌───────────────────────────────┼───────────────────────┐                                
                              │                               │                       │                                
                              v                               v                       v                                
                        sparrow-engine-server               sparrow-engine-cli                    sparrow-engine-python
                        (HTTP API, 15 routes)      (CLI binary)                 (Python wheel)                         
                              │                               │                       │                                
                              │  ┌────────────────────────────┼───────────────────┐   │                                
                              │  │                                                │   │                                
                              v  v                                                v   v                                
                         Sparrow Studio Web                              Sparrow Studio Local                          
                         (Flask + workers, Docker)                       (Avalonia / .NET desktop;                     
                                                                         loads sparrow_engine.dll via P/Invoke)        

Five ways to call Sparrow Engine:                                                                                      
  CLI binary · Python wheel · HTTP SDK (sparrow-engine-client) · HTTP API (sparrow-engine-server) · Native DLL (C ABI) 

Two device flavors, never co-located in one binary:                                                                    
  cpu  → ORT CPU EP only,  Python wheel "sparrow-engine",     CLI binary "spe"                                         
  gpu  → ORT CUDA EP added, Python wheel "sparrow-engine-gpu", CLI binary "spe-gpu"                                    
```

Both flavors export the same 32 `sparrow_engine_*` symbols and ship as `libsparrow_engine.so` — Sparrow Studio Local's `[DllImport("sparrow_engine")]` resolves either flavor.

---

## 1. What Sparrow Engine is

### Section overview

```
                  ┌──────────────────────────────────────┐                                
                  │              sparrow-engine            │                              
                  │  Loads ONNX models, runs inference.  │                                
                  │  That's it. No annotation, no        │                                
                  │  training, no storage, no registry.  │                                
                  └──────────────────┬───────────────────┘                                
                                     │                                                    
       ┌─────────────────────────────┼─────────────────────────────┐                      
       │                             │                             │                      
       v                             v                             v                      
   Sparrow Studio              sparrow-data sibling        sparrow-ops sibling            
   (consumer; uses             (DEFERRED — data            (DEFERRED — registry,          
    sparrow-engine's HTTP API            substrate, ingestion,        drift Tier-3, CI/CD,
    and native DLL)             logging, snapshots)          monitoring)                  
```

**Why**: Sparrow Engine exists so wildlife-conservation teams can run camera-trap and bioacoustic models fast, in production, without re-implementing inference per consumer.
**What**: a Rust library that loads ONNX models and returns predictions. Engine only.
**How**: TOML manifests describe each model; Sparrow Engine handles preprocessing, inference via ORT, and postprocessing. Consumers (CLI, Python, HTTP, native DLL) wrap the engine.

**Plain words**: "engine only" = Sparrow Engine does not store data, train models, or manage deployments. Sibling repos (deferred) handle those.

---

### 1.1 Origin

```
PytorchWildlife (Python)         sparrow-engine (Rust)
  ┌───────────────┐               ┌───────────────┐   
  │ ~6× slower    │  ── rewrite ─►│ Sub-100 ms    │   
  │ Multi-GB env  │               │ Single binary │   
  │ Python-only   │               │ + 4 consumers │   
  └───────────────┘               └───────────────┘   
```

**Why**: PytorchWildlife (Python + Triton) was slow and hard to ship. Sparrow Studio needed an engine that could be embedded as a DLL.
**What**: a from-scratch Rust rewrite preserving the same models and detection accuracy (`+~4%` more detections than Triton thanks to no redundant NMS).
**How**: ONNX everywhere, NCHW layout, NMS-in-graph, single-binary distribution.

**Cite**: `docs/master_plan.md § Phase 1, Phase 2`; benchmark headline `docs/benchmarks.md § 8.1` (libsparrow_engine 43.86 ms/img median vs PW PyTorch 24.76 ms — note: PW has the cold-start advantage; sparrow-engine is 1.96× faster than PW's FP16 on Phase 3.8 numbers per `docs/master_plan.md`).

---

### 1.2 Locked invariants — never violated by user code

The following constraints are baked into the engine. If you onboard a new model, it must satisfy all of them.

| # | Invariant | What happens if violated |
|---|---|---|
| 1 | ONNX only (vision + audio). | Manifest load fails. |
| 2 | NCHW layout (Batch × Channels × Height × Width). | ORT CUDA EP can crash; sparrow-engine rejects at load. |
| 3 | NMS lives in the ONNX graph, not in sparrow-engine (for detectors). | Sparrow Engine refuses non-conforming detectors. |
| 4 | Normalized bbox `[0,1]` at every public boundary. | Sparrow expects normalized; raw pixels would break consumers. |
| 5 | TOML manifests, not YAML. | `serde_yaml` is deprecated; sparrow-engine only parses TOML. |
| 6 | `Engine` is a singleton (process-global). | Second `Engine::new()` returns error. Python multiprocessing must use `spawn`, not `fork`. |
| 7 | sparrow-engine owns preprocessing + postprocessing. | Eliminates the old Sparrow Local vs Web divergence. |

**Cite**: `README.md:66-75`, `docs/master_plan.md § Locked-in design decisions`.

---

## 2. Installation

### Section overview

```
                       installer/sparrow-engine-install.{sh,ps1}             
                                    │                                        
                  ┌─────────────────┴─────────────────┐                      
                  │           Layer 1 probe           │                      
                  │  nvidia-smi · libcuda.so.1 ·      │                      
                  │  WMI Win32_VideoController        │                      
                  └─────────────────┬─────────────────┘                      
                                    │ found NVIDIA?                          
                           yes ─────┴───── no                                
                            │              │                                 
                            v              v                                 
                      Layer 2 probe     CPU flavor                           
                      (cuDNN ≥9.10)        │                                 
                            │              │                                 
                       pass │ fail         │                                 
                            │   └─exit 11──┘                                 
                            v                                                
                       GPU flavor                                            
                            │                                                
       ┌────────────────────┼─────────────────────────┐                      
       │                    │                         │                      
       v                    v                         v                      
   CLI tarball         pip wheel                Docker image                 
   ~/.sparrow_engine/bin    active Python env       sparrow-engine:cpu / :gpu
```

**Why**: one wrapper hides the GPU-detection complexity. The user runs one command and gets the right build.
**What**: a layered probe + downloader that installs into `~/.sparrow_engine/` (Linux/macOS) or `%USERPROFILE%\.sparrow_engine\` (Windows).
**How**: layer-1 detects NVIDIA hardware; layer-2 verifies cuDNN quality; the wrapper picks `cpu` or `gpu` flavor and fetches the matching artifact.

**Cite**: `docs/install.md`, `installer/sparrow-engine-install.{sh,ps1}`.

---

### 2.1 Quickstart per platform

```
┌────────────────────┬────────────────────────────────────────────┐         
│ Linux x86_64       │ bash installer/sparrow-engine-install.sh            │
│ macOS arm64/x86_64 │ bash installer/sparrow-engine-install.sh (CPU only) │
│ Windows x86_64     │ installer\sparrow-engine-install.ps1                │
└────────────────────┴────────────────────────────────────────────┘         
```

**Why**: same script, same flags, different shell.
**What**: from a clone of the repo, run the wrapper with no flags; it auto-picks flavor.
**How**: the wrapper writes binaries to `~/.sparrow_engine/bin/` and (if Python is active) installs the matching wheel into that environment.

**Footgun**: the `curl … | sh` and `iwr … | iex` one-liners DO NOT WORK today (the wrapper resolves `probe.sh` relative to `$0`, which becomes `bash` under stdin pipes). Use the local-clone form. Tracked for post-public-release.

---

### 2.2 Flags you'll use

| Flag | Purpose |
|------|---------|
| `--flavor cpu` / `--flavor gpu` | Skip the probe and force a flavor. |
| `--docker` | Install the HTTP-server image (`sparrow-engine:cpu` or `:gpu`) instead of CLI + wheel. |
| `--cli` | Install only the CLI binary (skip the Python wheel). |
| `--pip` | Install only the wheel (skip the CLI). |
| `--reprobe` | Uninstall the current flavor and re-detect. Use after a driver upgrade. |
| `--force-rc-overwrite` | Let the wrapper rewrite your shell rc-file block even if you edited it manually. |

---

### 2.3 What lands on disk

```
~/.sparrow_engine/                                            
├── bin/                                                      
│   ├── spe                    (or spe-gpu on the GPU flavor) 
│   └── (symlink helpers, ort-env.sh)                         
├── lib/              (any extra .so files the wrapper places)
└── current_flavor    (one line: "cpu" or "gpu")              

~/.bashrc / ~/.zshrc:                                         
  # >>> sparrow_engine >>>                                    
  export PATH="$HOME/.sparrow_engine/bin:$PATH"               
  # <<< sparrow_engine <<<                                    
```

**Why**: predictable footprint so uninstall is one `rm -rf ~/.sparrow_engine` and one rc-file block delete.
**What**: a single directory + a single rc-file fenced block. No system-wide files.
**How**: tarball extraction + optional pip install + optional Docker image pull.

**Plain words**: rc-file = the shell startup file (`~/.bashrc`, `~/.zshrc`, or PowerShell profile) that runs when you open a new terminal.

> **⚠️ Important — read before invoking the CLI binary directly**
>
> The wrapper sources `ort-env.sh` so the `spe` (or `spe-gpu`) command found on `PATH` has the right `LD_LIBRARY_PATH` baked in. If you call the binary by its **full path** (e.g., `~/.sparrow_engine/bin/spe`, or after copying it to a deploy location), the dynamic linker has no idea where `libonnxruntime.so.1` lives and you get `cannot open shared object file`. Full repro + workaround is at **§13.6 LD_LIBRARY_PATH for CLI binaries**; the short version is: prepend `LD_LIBRARY_PATH=<onnxruntime-libs>` when invoking the binary outside the wrapper-PATH context. This surfaces at install time so you see it before you write the first script that invokes `spe` by full path.

---

### 2.4 The 4 install paths

| Path | Best for | Command |
|------|----------|---------|
| Clean-room from-source build | Developers; reproducibility | `cd sparrow-engine && ./scripts/build_all_flavors.sh` (workspace root) |
| GitHub Releases binary | End users; production | `bash installer/sparrow-engine-install.sh --cli` |
| pip install Python wheel | Notebook + script users | `pip install sparrow-engine` (CPU) or `pip install sparrow-engine-gpu` |
| Docker image pull | Server deployments | `docker pull sparrow-engine:cpu` (or `:gpu`) |

**Cite**: `docs/install.md § Per-consumer install paths` (lines 163-220); `sparrow-engine/scripts/build_all_flavors.sh`.

---

### 2.5 Exit codes you might see

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | User aborted (Ctrl-C) |
| 3 | `--flavor gpu` requested on a host with no NVIDIA hardware |
| 4 | Network failure after retries |
| 5 | Python too old (< 3.11) |
| 6 | sha256 mismatch (transit corruption or tampering) |
| 8 | Required tool missing (curl, tar, docker, or pip) |
| 9 | Platform unsupported (e.g., GPU on macOS) |
| 11 | cuDNN < 9.10 — BLOCKING; see Gotchas §13.1 |
| 12 | Cross-flavor install attempted without `--reprobe` |
| 14 | `SPARROW_ENGINE_RELEASE_BASE` unset (pre-public-release) |

**Cite**: `docs/install.md § Error message catalog`.

---

### 2.6 Air-gapped install

```
ONLINE machine:                 OFFLINE machine:              
┌───────────────────┐           ┌───────────────────┐         
│ Build / download  │   USB     │ Receive tarball   │         
│   sparrow-engine tarball   │  ───────► │ + sha256          │
│ + sparrow_engine.dll       │           │                   │
│ + sha256          │           │ Verify + extract  │         
└───────────────────┘           │ into ~/.sparrow_engine/    │
                                └───────────────────┘         
```

**Why**: many camera-trap deployments are in field sites with no internet.
**What**: a documented manual path that copies a tarball or `docker save` archive onto the offline host.
**How**: download once from a connected machine; transfer via USB; the wrapper accepts a local `file://` URL via `SPARROW_ENGINE_RELEASE_BASE`.

**Cite**: `docs/install.md § Air-gapped / offline install`.

---

## 3. Hardware + system requirements

### Section overview

```
┌─────────────────────────────────────────────────────────────┐
│                       CPU flavor                            │
│  Any x86_64 / arm64 Linux · macOS · Windows                 │
│  No GPU needed; ORT runs on CPU EP                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       GPU flavor                            │
│  NVIDIA driver  ≥ 550.x                                     │
│  CUDA runtime    12.6                                       │
│  cuDNN          ≥ 9.10  ← layer-2 probe blocks 9.8          │
│  ORT version     1.25.x (NOT 1.26+; see Gotchas §13.2)      │
└─────────────────────────────────────────────────────────────┘
```

**Why**: ORT's CUDA EP refuses to load without cuDNN, and cuDNN 9.8 has a bug on sm_89 that crashes SpeciesNet.
**What**: a minimum-supported stack documented per platform.
**How**: the layer-2 probe in `installer/probe_gpu_quality.sh` verifies cuDNN ≥ 9.10 before installing the GPU flavor; failure produces exit 11.

---

### 3.1 GPU prerequisites in detail

| Component | Minimum | Why this version |
|-----------|---------|------------------|
| NVIDIA driver | 550.x | Compatible with CUDA 12.6 runtime per NVIDIA's matrix. |
| CUDA runtime | 12.6 | What `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` ships; matches ORT's build. |
| cuDNN | ≥ 9.10 | 9.8 hits "No valid engine configs for ConvFwd_" on SpeciesNet (sm_89 RTX 6000 Ada). Fixed in 9.10. |
| GPU memory | ~2 GB headroom | MegaDetector v6 fits in ~1.2 GB; pipeline (detect + classify) ~2 GB peak. |

**Cite**: `docs/lessons.md § cuDNN 9.8 has a Conv engine bug`; `docs/install.md § Hardware requirements`.

---

### 3.2 The `spe device` trap

```
spe device  →  prints "cuda:0"   ← compile-time check only
                    │                                     
                    │ does NOT verify:                    
                    │  • GPU hardware present             
                    │  • driver loaded                    
                    │  • cuDNN findable                   
                    │  • CUDA runtime accessible          
                    v                                     
              Real GPU check: run an inference and watch  
              `nvidia-smi --query-compute-apps=...`       
```

**Why**: `ort::ep::CUDA::is_available()` only checks "was ORT built with CUDA EP support" — not that the GPU works.
**What**: a known footgun. `spe device` saying `cuda:0` is necessary, not sufficient.
**How**: to verify GPU is actually being used, run inference and check `nvidia-smi` shows the sparrow-engine process consuming GPU memory.

**Cite**: `docs/lessons.md § "spe device" alone is NOT a GPU check` (MT-14).

---

### 3.3 OS support matrix

| OS | CPU flavor | GPU flavor |
|----|------------|------------|
| Linux x86_64 (Ubuntu 22.04+, 24.04 tested) | YES | YES |
| macOS arm64 (Apple Silicon) | YES | NO (no NVIDIA) |
| macOS x86_64 (Intel) | YES | NO (eGPU not auto-detected) |
| Windows x86_64 | YES | YES |
| Linux arm64 | NOT YET TESTED | NO |

---

## 4. Five ways to call sparrow-engine

### Section overview

```
                       Same engine, same models, five surfaces                                                                     

     sparrow-engine CLI       sparrow-engine-python      sparrow-engine-client      sparrow-engine-server      libsparrow_engine.so
     (terminal)      (Python pkg)      (HTTP SDK)        (HTTP server)     (C ABI DLL)                                             
         │                │                 │                 │                 │                                                  
         │ PyO3 bindings  │  HTTP over      │ axum routes     │ cdylib called   │                                                  
         │ (in-process)   │  localhost      │ + 15 endpoints  │ via P/Invoke    │                                                  
         v                v                 v                 v                 v                                                  
                              ┌─────────────────────────┐                                                                          
                              │   sparrow-engine          │                                                                        
                              │   (sparrow-engine-cpu OR         │                                                                 
                              │    sparrow-engine-gpu)           │                                                                 
                              └─────────────────────────┘                                                                          
```

**Why**: different consumers need different surfaces. A scientist wants the CLI; a notebook wants Python; Sparrow Web wants HTTP; Sparrow Local wants native DLL.
**What**: five surfaces sharing one engine.
**How**: each surface depends on either `sparrow-engine-cpu` or `sparrow-engine-gpu` via a Cargo feature flag (`--features cpu` or `--features gpu`, mutually exclusive, default = cpu).

---

### 4.1 Functionality consistency rule

Both CLI and Python expose the **same** function set with the same conventions. If you can `detect` in Python, you can `detect` in the CLI with the same flags.

| Function | CLI | Python | HTTP | Native DLL |
|----------|-----|--------|------|------------|
| Detect | `spe detect` | `sparrow_engine.detect()` | `POST /v1/detect` | `sparrow_engine_detect()` |
| Classify | `spe classify` | `sparrow_engine.classify()` | `POST /v1/classify` | `sparrow_engine_classify()` |
| Detect audio | `spe detect-audio` | `sparrow_engine.detect_audio()` | `POST /v1/audio/detect` | `sparrow_engine_detect_audio()` |
| Pipeline | `spe pipeline` | `sparrow_engine.pipeline()` | `POST /v1/pipeline` | `sparrow_engine_run_pipeline()` |
| List models | `spe models list` | `sparrow_engine.list_models()` | `GET /v1/models` | `sparrow_engine_list_models()` |
| Model info | `spe models info <id>` | `sparrow_engine.model_info()` | (part of `/v1/models`) | `sparrow_engine_engine_model_info()` |
| Hash file | `spe hash` | `sparrow_engine.hash_file()` | (n/a) | `sparrow_engine_hash_file()` |
| Day/night | `spe day-night` | `sparrow_engine.day_night()` | (n/a) | `sparrow_engine_day_night()` |
| Verify model | `spe models verify` | `sparrow_engine.verify_model()` | (n/a) | `sparrow_engine_verify_model()` |
| Active device | `spe device` | `sparrow_engine.active_device()` | (n/a) | (n/a) |
| Initialize | `spe init` | `sparrow_engine.init()` | (server boots) | `sparrow_engine_engine_new()` |

**Cite**: `docs/master_plan.md § Phase 2.5`; `sparrow-engine/sparrow-engine-cli/src/main.rs:96-119`; `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:127-340`.

---

### 4.2 Device selection

```
              Device::Auto  (default)                                    
                     │                                                   
            ┌────────┴────────┐                                          
            │                 │                                          
       CPU flavor          GPU flavor                                    
            │                 │                                          
            v                 v                                          
              Cpu          ────────►   Cuda(0)                           
        (always)           (always)                                      

              Device::Cpu          →   Cpu (both flavors)                
              Device::Cuda(N)      →   Cuda(N) on GPU flavor;            
                                     warns + coerces to Cpu on CPU flavor
```

**Why**: post-MT-4.1-2, `Device::Auto` is **flavor-strict** — the CPU wheel never opportunistically grabs CUDA via runtime ORT, even if `LD_LIBRARY_PATH` exposes a GPU ORT.
**What**: each flavor has a fixed device set; cross-flavor requests are coerced with a `tracing::warn!` instead of failing.
**How**: sparrow-engine-cpu's `Device::Auto | Cpu` resolves to `Cpu`; `Device::Cuda(_)` gets silently coerced to `Cpu` with a warn-log. sparrow-engine-gpu's `Device::Auto | Cpu` coerces to `Cuda(0)`.

**Cite**: `docs/master_plan.md § Engine + safety invariants`, MT-4.1-2.

---

### 4.3 Future surfaces — R + Julia bindings (planned, not built yet)

```
            Today (5)                             Future (planned)                  
   ─────────────────────────────────         ─────────────────────────────────      
   CLI · Python · HTTP SDK · HTTP API        R bindings  (via `extendr` or          
   · Native DLL (C ABI)                       direct C ABI from R's                 
                                              `.Call` interface)                    
                                             Julia bindings (via `ccall` on         
                                              the existing libsparrow_engine cdylib)
```

**Why**: the camera-trap + ecology research community uses R and Julia heavily; sparrow-engine's C ABI is already the right substrate for both.
**What**: planned R + Julia consumers that wrap the existing 32 `sparrow_engine_*` exports. No new Sparrow Engine-side code expected — the cdylib + C header are the integration point.
**How (planned)**: both languages call the cdylib directly. R via `.Call` + a thin shim package; Julia via `ccall` (zero extra build artifacts beyond `libsparrow_engine.so` itself).

**Status**: NOT YET STARTED. No timeline. Tracked as user-directed future scope (review-round-1 comment 2026-05-19). When kicked off, file under `docs/master_plan.md § Future consumer surfaces`.

**Plain words**: "binding" = a small adapter package in the host language that translates the host's data types (R vectors, Julia arrays) to/from the C ABI sparrow-engine exposes.

---

**Status (revision 2, 2026-05-19)**: ADDRESSED — see new §4.3 "Future surfaces — R + Julia bindings (planned, not built yet)".

---

## 5. CLI — `sparrow-engine`

### Section overview

```
sparrow-engine [GLOBAL FLAGS] <COMMAND> [COMMAND FLAGS] [INPUT...]           

Global flags:                                                                
  --device {auto,cpu,cuda:N}    default: auto                                
  --model-dir <PATH>            default: SPARROW_ENGINE_MODEL_DIR or ./models
  --quiet                       suppress progress bars                       

Commands:                                                                    
  detect          Object detection on images                                 
  classify        Single-label classification on images                      
  detect-audio    Sliding-window audio detection                             
  pipeline        detect → classify on each crop                             
  models list     List loaded models                                         
  models info     Show info for one loaded model                             
  models verify   Verify ONNX SHA-256 against manifest                       
  device          Print the active device label                              
  init            Pre-load the engine (uses global flags)                    
  hash            SHA-256 of one file                                        
  day-night       Day/night classification (BT.709 brightness)               
```

**Why**: a CLI that mirrors the Python API one-for-one so scripts and notebooks stay consistent.
**What**: 9 commands; all batch-capable accept files, directories, or mixed; output goes to stdout (JSON / CSV) or per-file visualization.
**How**: clap-derived parser; engine is initialized lazily on first inference command.

**Plain words**: "lazy" = the engine doesn't actually load ORT or any model until you run a command that needs it. `spe --help` exits without touching the GPU.

---

### 5.1 `spe detect` — bounding-box detection

```
$ spe detect IMG1.jpg IMG2.jpg \                          
    --model megadetector-v6-yolov10e \                    
    --threshold 0.2 \                                     
    --export-format coco --export-output detections.json \
    --visualize --output-dir viz_out/ --show-labels       
```

**Why**: find boxes (animals, vehicles, people) in camera-trap images.
**What**: per-image list of `{bbox: [x,y,w,h] normalized [0,1], class, confidence}`; optional COCO/megadet/CSV export; optional bbox overlay.
**How**: sparrow-engine loads the manifest, runs ORT, parses the in-graph NMS output, returns normalized boxes.

| Flag | What |
|------|------|
| `--model <id>` | Pick a detector (auto-fallback to default if omitted). |
| `--threshold <f>` | Drop detections below this confidence. |
| `--max-detections <n>` | Cap per image. |
| `--print` | Stream JSON/CSV per file to stdout (default off). |
| `--format {json,csv}` | Format for `--print`. |
| `--recursive` | Recurse into input directories. |
| `--summary` | Print batch statistics (counts per class, etc.). |
| `--export-format {megadet,coco,csv}` | Consolidated batch export. |
| `--export-output <path>` | Where to write the export (else stdout). |
| `--visualize` | Render annotated images. |
| `--output-dir <dir>` | Where to write visualization (required with `--visualize`). |
| `--show-labels` | Render `"{label} {conf:.2}"` text above each box (default off). |

**Cite**: `sparrow-engine/sparrow-engine-cli/src/main.rs:121-165`.

---

### 5.2 `spe classify` — single-label classification

```
$ spe classify crops/*.jpg \ 
    --model speciesnet-crop \
    --top-k 3 \              
    --print --format json    
```

**Why**: assign a species label to a pre-cropped image.
**What**: per-image list of top-k `{label, confidence}` pairs.
**How**: ORT softmax → top-k.

| Flag | What |
|------|------|
| `--model <id>` | Required (no default classifier). |
| `--top-k <n>` | How many predictions per image. |
| `--print / --format / --recursive / --visualize / --output-dir / --show-labels` | Same semantics as `detect`. |

---

### 5.3 `spe detect-audio` — sliding-window audio detection

```
$ spe detect-audio recordings/*.wav \
    --model md-audiobirds-v1 \       
    --threshold 0.9 \                
    --raw-segments                   
```

**Why**: detect birds (or other audio classes) in WAV recordings.
**What**: by default, **merged time ranges** `{start_time_s, end_time_s, max_confidence, class}`. With `--raw-segments`, one row per sliding window.
**How**: sparrow-engine decodes WAV (hound), computes mel spectrogram, runs the model with stride, applies sigmoid, merges consecutive above-threshold windows.

| Flag | What |
|------|------|
| `--model <id>` | Audio model (only `md-audiobirds-v1` is in the catalog today). |
| `--threshold <f>` | Per-window sigmoid threshold (manifest default 0.9). |
| `--raw-segments` | Emit pre-merge per-window rows. |
| `--visualize --output-dir <dir>` | Render spectrogram + confidence heatmap PNGs. |
| `--smooth` | Apply Gaussian blur to the heatmap visualization. |
| `--show-windows` | Add the per-window placement diagnostic band. |

**Cite**: `sparrow-engine/sparrow-engine-cli/src/main.rs:206-254`.

---

### 5.4 `spe pipeline` — detector → classifier

```
$ spe pipeline IMG.jpg \                            
    --detector megadetector-v6-yolov10e \           
    --classifier speciesnet-crop \                  
    --threshold 0.2 --top-k 3 \                     
    --export-format megadet --export-output out.json
```

**Why**: most camera-trap workflows are "find the animal, then identify the species".
**What**: per-image list of `{bbox, detection_confidence, top_k: [{label, confidence}]}`.
**How**: sparrow-engine runs the detector, crops each box (normalized → pixel coords using the original image dims), runs the classifier on each crop.

**Adhoc form**: no separate pipeline manifest needed — pass `--detector` and `--classifier` IDs and sparrow-engine wires them at runtime.

---

### 5.5 Model management subcommands

| Subcommand | Behavior |
|------------|----------|
| `spe models list` | List loaded ORT sessions. |
| `spe models info <id>` | Show one model's manifest fields. |
| `spe models verify [--write]` | Re-hash each model's ONNX and compare to manifest. `--write` updates the manifest if hash is stale. |

---

### 5.6 Utility commands

| Command | What it does |
|---------|--------------|
| `spe device` | Print `cpu` / `cuda:N` (compile-time only; not a real GPU check — see §3.2). |
| `spe init` | Initialize the engine without running inference (warms ORT for the next command). |
| `spe hash <file>` | SHA-256 of one file. |
| `spe day-night <image>` | Returns `{is_day: bool, brightness: f32}` using BT.709 weighting. |

---

## 6. Python package — `sparrow-engine`

### Section overview

```
                       pip install sparrow-engine            (CPU; depends on onnxruntime>=1.25.1,<1.26)    
                       pip install sparrow-engine-gpu        (GPU; depends on onnxruntime-gpu>=1.25.1,<1.26)

                       Both wheels:                                                                         
                                  import sparrow_engine                                                     
                                       │                                                                    
                                       v                                                                    
                       _sparrow_engine_core.cpython-3X.so       (PyO3 native module)                        
                                       │                                                                    
                                       v                                                                    
                                sparrow-engine-cpu or sparrow-engine-gpu Rust crate                         
                                       │                                                                    
                                       v                                                                    
                                    ORT (CPU or CUDA EP)                                                    
```

**Why**: scientists work in notebooks. A Python wheel removes the install friction of the CLI.
**What**: a single `import sparrow_engine` exposing 14 functions, plus IDE-ready type stubs (`_core.pyi`).
**How**: PyO3 0.25 builds the Rust → Python bridge; the GIL is released during inference (`py.allow_threads`).

**Plain words**: "GIL" (Global Interpreter Lock) is the Python rule that only one thread can run Python bytecode at a time. Releasing it during inference means other threads keep working.

---

### 6.1 The 14 public functions

| Function | Returns |
|----------|---------|
| `init(device="auto", model_dir=None)` | None; pre-loads the engine. |
| `detect(inputs, model=None, threshold=None, max_detections=None, progress_callback=None)` | `list[DetectResult]` |
| `classify(inputs, model, top_k=None, progress_callback=None)` | `list[ClassifyResult]` |
| `detect_audio(inputs, model=None, threshold=None, raw_segments=False, progress_callback=None)` | `list[AudioResult]` |
| `pipeline(inputs, detector, classifier, threshold=None, top_k=None, progress_callback=None)` | `list[PipelineResult]` |
| `list_models()` | `list[ModelInfo]` |
| `model_info(model_id)` | `ModelInfo` |
| `active_device()` | `str` (`"cpu"`, `"cuda:0"`, …) |
| `hash_file(path)` | `str` (lowercase hex SHA-256) |
| `day_night(path)` | `dict` with `is_day`, `brightness` |
| `verify_model(model_id)` | `dict` with `ok`, `expected`, `actual` |
| `summarize(results)` | `dict` of detection statistics |
| `visualize(results, output_dir, ...)` | None; writes PNGs |
| `export(results, format, output)` | None; writes consolidated batch output |

**Cite**: `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:127-340`.

---

### 6.2 Inputs — flexible at every call

`detect`, `classify`, `detect_audio`, and `pipeline` accept any of:

```
Path-like               → sparrow_engine.detect("img.jpg", ...)           
List of path-likes      → sparrow_engine.detect(["a.jpg", "b.jpg"], ...)  
Directory               → sparrow_engine.detect("/photos/", ...)          
List of directories     → sparrow_engine.detect(["/A/", "/B/"], ...)      
Mixed                   → sparrow_engine.detect(["img.jpg", "/dir/"], ...)
```

**Why**: lets you script the same way you'd hand someone a folder.
**What**: a single input parameter that handles single files, lists, and folders.
**How**: `_resolve_inputs()` (see `__init__.py:92-125`) walks any directory entries; non-image / non-audio files are skipped silently.

---

### 6.3 Progress callback

```python
def on_progress(filename: str, index: int, total: int) -> None: 
    print(f"[{index}/{total}] {filename}")                      

sparrow_engine.detect("/photos/", progress_callback=on_progress)
```

**Why**: notebook UIs need a hook to draw progress bars; the CLI uses indicatif, Python needs its own.
**What**: optional kwarg on the 4 batch methods.
**How**: invoked once per input file (after that file completes).

---

### 6.4 Two wheels, one import

| Wheel | onnxruntime dep | Device::Auto behavior |
|-------|-----------------|------------------------|
| `sparrow-engine` (CPU) | `onnxruntime>=1.25.1,<1.26` | Resolves to `cpu`. `cuda:N` coerces to `cpu` with `tracing::warn!`. |
| `sparrow-engine-gpu` (GPU) | `onnxruntime-gpu>=1.25.1,<1.26` | Resolves to `cuda:0`. `cpu` coerces to `cuda:0` with `tracing::warn!`. |

**Footgun**: never install both. `pip uninstall sparrow-engine` before `pip install sparrow-engine-gpu`.

**Plain words**: "wheel" = a Python package file with a `.whl` extension. The two wheels share an importable module name (`sparrow-engine`); only one can be installed in any environment.

---

### 6.5 Logging bridge

```python
import logging
logging.getLogger("sparrow_engine.python").setLevel(logging.DEBUG)
```

Sparrow Engine's Rust `tracing` events are bridged to the Python `logging` module via `pyo3-log`. The logger name is `sparrow_engine.python`. Set its level to see ORT initialization, model load events, and inference timings.

**Cite**: Phase 3.5 S6 (`docs/master_plan.md § Phase 3.5 Wave 2`); `sparrow-engine/sparrow-engine-python/src/lib.rs`.

---

## 7. HTTP API server — `sparrow-engine-server`

### Section overview

```
                      sparrow-engine-server (axum, 15 routes)                        
                                  │                                                  
       ┌──────────────────────────┼─────────────────────────────┐                    
       │ Inference (5)            │ Management (8)              │ Health (2)         
       │                          │                             │                    
       │ POST /v1/detect          │ GET    /v1/catalog          │ GET /v1/health     
       │ POST /v1/detect/batch    │ GET    /v1/models           │ GET /healthz       
       │ POST /v1/classify        │ POST   /v1/models/load      │                    
       │ POST /v1/audio/detect    │ DELETE /v1/models/{id}      │                    
       │ POST /v1/pipeline        │ GET    /v1/pipelines        │                    
       │                          │ POST   /v1/pipelines        │                    
       │                          │ POST   /v1/pipelines/load   │                    
       │                          │ DELETE /v1/pipelines/{id}   │                    
       └──────────────────────────┴─────────────────────────────┘                    

       Per-request query params (Phase 4):  ?store=true · ?halt_on_store_failure=true
```

**Why**: a network-callable surface for Sparrow Studio Web and any HTTP consumer.
**What**: 15-endpoint REST API, JSON in/out, configurable through `SPARROW_ENGINE_*` env vars.
**How**: axum + tower-http; one ORT engine per process; an inference log sink + drift metrics fold in via Phase 4.

**Cite**: `sparrow-engine/sparrow-engine-server/src/router.rs`.

---

### 7.1 Configuration — `SPARROW_ENGINE_*` env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARROW_ENGINE_BIND_ADDR` | `0.0.0.0:8080` | TCP listen address. |
| `SPARROW_ENGINE_MODEL_DIR` | `/models` | Directory scanned for manifests at boot. |
| `SPARROW_ENGINE_LOG_FORMAT` | `json` | `json` (Docker) or `pretty` (dev). |
| `SPARROW_ENGINE_LOG_LEVEL` | `info` | Standard tracing level. |
| `SPARROW_ENGINE_MAX_BODY_SIZE` | `100mb` | Per-request body limit. |
| `SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE` | `32` | Tokio semaphore around inference. |
| `SPARROW_ENGINE_MAX_BATCH_SIZE` | `64` | `/v1/detect/batch` cap. |
| `SPARROW_ENGINE_REQUEST_TIMEOUT` | `120` (s) | Per-request timeout. |
| `SPARROW_ENGINE_DRAIN_TIMEOUT` | `10` (s) | Graceful-shutdown drain. |
| `SPARROW_ENGINE_DEVICE` | `auto` | `auto`/`cpu`/`cuda:N` (flavor-strict per §4.2). |
| `SPARROW_ENGINE_INTER_THREADS` / `SPARROW_ENGINE_INTRA_THREADS` | unset | ORT thread tuning. |
| `SPARROW_ENGINE_IDLE_UNLOAD_SEC` | `1800` (30 min) | Idle-reaper threshold; set to `0` to disable. |
| `SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N` | `1` | Keep the N most-recently-used even if idle. |
| `SPARROW_ENGINE_PRELOAD` | unset | Comma-separated model IDs to eagerly load at boot. Unknown IDs fail boot. |

**Cite**: `sparrow-engine/sparrow-engine-server/src/config.rs`.

---

### 7.2 Phase 4.4 boot-time CLI

```
$ sparrow-engine-server --help        # exit 0, lists every SPARROW_ENGINE_* env var
$ sparrow-engine-server --version     # exit 0                                      
$ sparrow-engine-server healthcheck   # Docker HEALTHCHECK probe                    
$ sparrow-engine-server               # serve (no positional args)                  
$ sparrow-engine-server --unknown     # exit 2, clap error message                  
```

**Why**: `--help` and `--version` used to be silently ignored (the server booted ORT before checking argv). Phase 4.4 fixes that.
**What**: clap parses argv BEFORE the tokio runtime, ORT engine, or TCP bind exist. `EADDRINUSE` now logs cleanly and exits 1 instead of panicking with a stack trace.
**How**: `sparrow-engine/sparrow-engine-server/src/cli.rs` (new module); `main.rs` switched from `#[tokio::main]` to sync `fn main()`.

**Cite**: `docs/master_plan.md § Phase 4.4`.

---

### 7.3 Inference endpoints (5)

| Endpoint | Body | Response |
|----------|------|----------|
| `POST /v1/detect?model=<id>` | multipart image OR JSON `{image_b64}` | `{detections: [{bbox:[x,y,w,h], class, confidence}, ...], inference_ms}` |
| `POST /v1/detect/batch?model=<id>` | JSON `{images: [b64, ...]}` | `{results: [...same-as-detect...], inference_ms}` |
| `POST /v1/classify?model=<id>&top_k=N` | multipart image | `{predictions: [{label, confidence}, ...], inference_ms}` |
| `POST /v1/audio/detect?model=<id>&threshold=<f>` | multipart WAV | `{ranges: [...], inference_ms}` (or per-window with `raw_segments=true`) |
| `POST /v1/pipeline?detector=<id>&classifier=<id>` (Shape X) OR `?pipeline=<id>` (Shape Y) | multipart image | `{results: [{bbox, detection_confidence, top_k: [...]}], inference_ms}` |

**Plain words**: "Shape X" = adhoc detector+classifier passed as query params; "Shape Y" = a named pipeline alias previously registered via `POST /v1/pipelines`.

**Cite**: `sparrow-engine/sparrow-engine-server/src/router.rs:24-32`; `sparrow-engine/sparrow-engine-server/src/handlers/*.rs`.

---

### 7.4 Per-request log + drift query params (Phase 4)

```
POST /v1/detect?model=<id>&store=true&halt_on_store_failure=false                   
                                  │            │                                    
                                  │            └── if sink errs:                    
                                  │                  false → 200 + warn-log         
                                  │                  true  → 500 INTERNAL_ERROR     
                                  └── emit InferenceLogRecord (schema_version="1.0")
                                       AFTER successful inference                   
```

**Why**: sparrow-data (deferred sibling) will ingest these records; this is the pre-positioning hook.
**What**: two optional query params on every inference endpoint, both default false.
**How**: emit happens AFTER inference returns successfully; never on the error path. The default sink writes one JSON line per record to stderr; future sinks plug in via the `InferenceLogSink` trait.

**Cite**: `sparrow-engine/sparrow-engine-server/src/sink.rs`, `sparrow-engine/sparrow-engine-types/src/inference_log.rs`, `docs/design/phase4/schema.md`.

---

### 7.5 Phase 4.2 catalog + lazy-load endpoints

```
GET /v1/catalog            → all discovered models + pipelines + loaded state        
GET /v1/models             → only currently-loaded ORT sessions                      
POST /v1/models/load       → load by id (idempotent: get_or_load)                    
DELETE /v1/models/{id}     → unload one model                                        

GET /v1/pipelines          → list registered alias pipelines                         
POST /v1/pipelines         → create named alias (idempotent; ?replace=true overrides)
POST /v1/pipelines/load    → load all component models of an existing alias          
DELETE /v1/pipelines/{id}  → remove an alias                                         
```

**Why**: Phase 4.2 made the server boot fast (no eager model load) and added alias management so consumers can register pipelines at runtime.
**What**: catalog discovery + lazy load + runtime alias CRUD.
**How**: discovery scans `SPARROW_ENGINE_MODEL_DIR/<id>/manifest.toml` at boot; models load on first inference request (or on explicit `POST /v1/models/load`); `SPARROW_ENGINE_PRELOAD=id1,id2` restores explicit eager load.

**Cite**: `docs/master_plan.md § Phase 4.2`; `sparrow-engine/sparrow-engine-server/src/handlers/{catalog,models,pipelines_mgmt}.rs`.

---

### 7.6 Idle-reaper

```
loop every clamp(SPARROW_ENGINE_IDLE_UNLOAD_SEC, 1, 60) seconds:
  for each loaded model:                                        
    if last_used_age > SPARROW_ENGINE_IDLE_UNLOAD_SEC           
       and model not in MRU keep_last_N:                        
       Engine::unload_model_by_id(model)                        
       tracing::info!("unloaded: {model_id}")                   
```

**Why**: long-running servers with many models leak GPU memory if nothing unloads sessions; eager unload would thrash the cache; idle-with-keep-N is the middle ground.
**What**: a background tokio task that drops idle models, except the N most-recently-used.
**How**: `LoadedModel.last_used` (Arc<AtomicU64>) updates on every `get_model_handle`; the reaper compares against `SPARROW_ENGINE_IDLE_UNLOAD_SEC`.

**Default**: 30 min idle, keep 1 most-recently-used. Set `SPARROW_ENGINE_IDLE_UNLOAD_SEC=0` to disable.

**Cite**: `docs/changelog.md § 2026-05-15` (commit `1a92413`).

---

### 7.7 Health endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /v1/health` | `{status: "ready" \| "no_models"}` — `ready` requires at least one loaded model. |
| `GET /healthz` | `{status: "alive"}` — process liveness; passes even with zero loaded models. |

**Cite**: `sparrow-engine/sparrow-engine-server/src/handlers/health.rs`.

---

## 8. HTTP SDK — `sparrow-engine-client`

### Section overview

```
sparrow-engine-client (Python package — separate from sparrow-engine)     
  ┌─────────────────────────────────────────────────────┐                 
  │ from sparrow_engine_client import SparrowEngineClient                │
  │                                                     │                 
  │ c = SparrowEngineClient("http://server:8080")               │         
  │ result = c.detect(open("img.jpg","rb"),             │                 
  │                   model="megadetector-v6-yolov10e") │                 
  └─────────────────────────────────────────────────────┘                 
                            │                                             
                            v                                             
                  HTTP POST /v1/detect                                    
                            │                                             
                            v                                             
                       sparrow-engine-server                              
```

**Why**: Sparrow Studio Web and other operators run sparrow-engine as a remote service. A Python SDK saves them writing HTTP clients.
**What**: a thin Python wrapper that mirrors sparrow-engine-server's endpoints.
**How**: ~385 LOC, 20 tests; uses `requests` under the hood.

**Plain words**: `sparrow-engine-client` ≠ `sparrow-engine` (PyO3). The two coexist by design — `sparrow-engine` is for local inference; `sparrow-engine-client` is for talking to a remote server.

**Cite**: `docs/master_plan.md § Phase 2`.

---

### 8.1 When to use which Python package

| You want to... | Package | Why |
|----------------|---------|-----|
| Run inference locally in a notebook | `sparrow-engine` (PyO3) | No server needed; lower latency. |
| Run against a remote sparrow-engine-server | `sparrow-engine-client` (HTTP) | No GPU needed locally; lets ops centralize inference. |
| Run inside Sparrow Studio Web | `sparrow-engine-client` | Sparrow's stack is HTTP-first. |
| Build a desktop GUI with embedded inference | `sparrow-engine` (PyO3) OR the C ABI cdylib | Pick based on Python vs .NET host. |

---

## 9. Native DLL — C ABI cdylib

### Section overview

```
                          libsparrow_engine.so / sparrow_engine.dll / libsparrow_engine.dylib          
                                          │                                                            
                                  32 exported symbols                                                  
                                  (all begin with `sparrow_engine_`)                                   
                                          │                                                            
        ┌─────────────────────────────────┼─────────────────────────────────┐                          
        v                                 v                                 v                          
   sparrow_engine.h (auto-generated by      NativeMethods.g.cs (auto-       Avalonia / .NET desktop app
   cbindgen, in repo)              generated by csbindgen)        uses `[DllImport("sparrow_engine")]` 
                                                                  to call the 32 exports               
```

**Why**: Sparrow Studio Local is a cross-platform desktop app written in C# (Avalonia). It can't link Rust directly; it needs a stable C ABI.
**What**: a single shared library with a fixed 32-symbol surface, validated byte-identical across both flavors (G5 acceptance gate).
**How**: `sparrow-engine-cpu/Cargo.toml` and `sparrow-engine-gpu/Cargo.toml` both set `[lib] name = "sparrow_engine"`, producing `libsparrow_engine.so` (or `.dll`/`.dylib`). `cbindgen` emits the C header; `csbindgen` emits the C# P/Invoke file.

**Plain words**: "P/Invoke" = .NET's mechanism for calling native shared libraries. `[DllImport("sparrow_engine")]` tells .NET to load `sparrow_engine.dll` (Windows) or `libsparrow_engine.so` (Linux).

**Cite**: `docs/master_plan.md § Sparrow Studio Local Integration`; `sparrow-engine/sparrow-engine-cpu/src/ffi.rs`.

---

### 9.1 The 32 exported functions

| Category | Functions |
|----------|-----------|
| Engine lifecycle (2) | `sparrow_engine_engine_new`, `sparrow_engine_engine_free` |
| Model lifecycle (4) | `sparrow_engine_load_model`, `sparrow_engine_load_model_by_id`, `sparrow_engine_unload_model`, `sparrow_engine_list_models` |
| Pipeline lifecycle (3) | `sparrow_engine_load_pipeline`, `sparrow_engine_load_pipeline_by_id`, `sparrow_engine_unload_pipeline` |
| Inference (6) | `sparrow_engine_detect`, `sparrow_engine_detect_raw`, `sparrow_engine_detect_batch`, `sparrow_engine_classify`, `sparrow_engine_run_pipeline`, `sparrow_engine_detect_audio`, `sparrow_engine_detect_audio_streaming` |
| Result lifecycle (5) | `sparrow_engine_audio_result_free`, `sparrow_engine_detections_free`, `sparrow_engine_classify_result_free`, `sparrow_engine_pipeline_result_free`, `sparrow_engine_free_string` |
| Errors + health (2) | `sparrow_engine_health`, `sparrow_engine_last_error` |
| Standalone utilities (5) | `sparrow_engine_hash_file`, `sparrow_engine_hash_result_free`, `sparrow_engine_day_night`, `sparrow_engine_image_brightness`, `sparrow_engine_verify_model`, `sparrow_engine_verify_result_free` |
| Engine-bound utilities (3) | `sparrow_engine_engine_verify_model`, `sparrow_engine_engine_model_info`, `sparrow_engine_engine_list_models_extended` |

Total: 32. Both `libsparrow_engine.so` flavors must export this exact set (G5 acceptance gate enforces this byte-identical).

**Cite**: `sparrow-engine/sparrow-engine-cpu/src/ffi.rs`; G5 gate at `docs/review/phase3.8-phase-c/round_01/acceptance_gates.md`.

---

### 9.2 Memory ownership rules

```
Returns from sparrow-engine:                          You must call:                                    
  *mut SparrowEngine   (from engine_new)          ──► sparrow_engine_engine_free                        
  *mut SparrowEngineDetections                    ──► sparrow_engine_detections_free                    
  *mut SparrowEngineClassifyResult                ──► sparrow_engine_classify_result_free               
  *mut SparrowEnginePipelineResult                ──► sparrow_engine_pipeline_result_free               
  *mut SparrowEngineAudioResult                   ──► sparrow_engine_audio_result_free                  
  *mut c_char (strings)                           ──► sparrow_engine_free_string                        
  Verify result                                   ──► sparrow_engine_verify_result_free                 

Errors:                                                                                                 
  After ANY *_new / *_load / *_detect that returns NULL or non-zero error,                              
  call `sparrow_engine_last_error()`                → returns *const c_char (read-only, no free needed).
```

**Why**: Rust frees what it allocates; C# / C / C++ must hand pointers back to sparrow-engine to free.
**What**: a strict ownership model — every `*mut T` returned has a matching `*_free` function.
**How**: `Drop` is not exposed across FFI. Calling the wrong free function (or none) leaks or crashes.

**Plain words**: "ownership" = who is responsible for freeing memory. Across an FFI boundary, you must hand pointers back to the side that allocated them.

---

### 9.3 Opaque handle safety

```
SparrowEngine    = c_void   (opaque)      
SparrowEngineModel     = c_void   (opaque)
SparrowEnginePipeline  = c_void   (opaque)
```

**Why**: the C side never sees Rust's struct layout, so structs can evolve without an ABI break.
**What**: opaque types — you only ever hold `*mut SparrowEngine`, never an instance.
**How**: every function that touches an engine/model/pipeline takes the opaque pointer and Rust looks up the real object.

**ABI evolution**: when the surface changes, new functions get `_v2` suffix; old functions stay frozen. There are no "reserved" fields in any FFI struct.

**Cite**: `docs/master_plan.md § Engine + safety invariants`.

---

### 9.4 Both flavors export the same symbols

```
libsparrow_engine.so (CPU flavor):                 libsparrow_engine.so (GPU flavor):                                   
  32 sparrow_engine_* symbols                        32 sparrow_engine_* symbols  ── byte-identical                     
  sparrow-engine-cpu/Cargo.toml: [lib] name="sparrow-engine"  sparrow-engine-gpu/Cargo.toml: [lib] name="sparrow-engine"
  cdylib filename: libsparrow_engine.so              cdylib filename: libsparrow_engine.so                              
```

**Why**: Sparrow Studio Local's `[DllImport("sparrow_engine")]` must resolve regardless of which flavor is installed.
**What**: the cdylib filename invariant + the 32-symbol invariant + the byte-identical-signature invariant (the implementation differs, but the symbol table is the same).
**How**: G5 acceptance gate diffs `nm -D` output between the two `libsparrow_engine.so` files; a mismatch fails the gate.

**Practical constraint**: never co-locate both flavors in the same `target/release/`. Per-flavor target dirs (`target-cpu/`, `target-gpu/`) are mandatory. Phase 3.8 Phase C's `scripts/build_all_flavors.sh` enforces this.

**Cite**: `docs/master_plan.md § Phase 3.8 Phase A C8 + Phase C Wave 4b`; G5 gate.

---

## 10. Models, catalogs, and TOML manifests

### Section overview

```
$MODEL_DIR/                       (SPARROW_ENGINE_MODEL_DIR)                           
├── megadetector-v6-yolov10e/                                                          
│   ├── manifest.toml             (canonical schema)                                   
│   ├── *.onnx                    (model file)                                         
│   ├── *_fp16.onnx               (optional FP16 file)                                 
│   └── *_labels.txt              (class names)                                        
├── speciesnet-crop/                                                                   
│   ├── manifest.toml                                                                  
│   ├── ...                                                                            
├── megadet-speciesnet/                                                                
│   └── pipeline.toml             (named pipeline alias)                               
└── ...                                                                                

Sparrow Engine discovers each directory whose name matches the manifest's `[model] id`.
Mismatch → directory skipped + tracing::warn.                                          
```

**Why**: Sparrow Engine is model-agnostic. Onboarding a new model = writing a manifest, not patching the engine.
**What**: a directory-per-model convention; one TOML file per model; optional pipeline TOMLs at the same level.
**How**: at boot, sparrow-engine-server scans `SPARROW_ENGINE_MODEL_DIR` and parses every `manifest.toml` / `pipeline.toml`; CLI/Python honor `--model-dir` / `init(model_dir=)`.

**Cite**: `sparrow-engine/sparrow-engine-server/src/discover.rs`; `sparrow-engine/models/*.toml`.

---

### 10.1 Manifest schema (TOML)

```toml
[model]                                                                                                  
id              = "megadetector-v6-yolov10e"     # MUST match the parent directory name                  
format          = "onnx"                                                                                 
file            = "model.onnx"                                                                           
file_fp16       = "model_fp16.onnx"               # optional                                             
onnx_sha256     = "9a9f22b8..."                   # used by `spe models verify`                          
onnx_size_bytes = 118484529                       # additional integrity field                           

[inference]                                                                                              
precision = "fp16"                                 # "fp16" or "fp32"; default fp32                      
strategy  = "single"                               # "single" or "tiled" (HerdNet, OWL-T)                
# cudnn_search_mode = "exhaustive"                 # (planned, P3.8-8) — not in current schema           

[preprocessing]                                                                                          
input_size      = [1280, 1280]                                                                           
layout          = "nchw"                           # mandatory; NHWC rejected                            
channel_order   = "bgr"                            # "rgb" (default) or "bgr"                            
method          = "letterbox"                      # "letterbox" or "resize"                             
normalization   = "unit"                           # "unit" (/255) or "imagenet"                         
pad_value       = 0.447                                                                                  

[postprocessing]                                                                                         
method                = "yolo_e2e"                  # see methods table below                            
confidence_threshold  = 0.2                                                                              
iou_threshold         = 0.45                        # for megadet_v5a (class-aware NMS in sparrow-engine)

[labels]                                                                                                 
file   = "labels.txt"                                                                                    
format = "name_index_csv"                                                                                

[provenance]   # OPTIONAL; round-tripped on output; sparrow-engine never interprets                      
training_dataset_id    = "..."                                                                           
training_experiment_id = "..."                                                                           
training_repo_commit   = "..."                                                                           

[drift_reference]   # OPTIONAL; per-class frequency for PSI computation                                  
buffalo  = 0.32                                                                                          
elephant = 0.20                                                                                          
empty    = 0.48                                                                                          
```

**Cite**: `sparrow-engine/sparrow-engine-types/src/manifest.rs`; canonical examples at `sparrow-engine/models/audiobirds.toml`, `sparrow-engine/models/herdnet.toml`, `sparrow-engine/models/owlt.toml`.

---

### 10.2 Postprocessing methods

| Method | What | Used by |
|--------|------|---------|
| `yolo_e2e` | YOLO with NMS-in-graph; output is `[N, 6]` boxes directly. | MDv6, DeepFaune, Amazon CT v2 |
| `megadet_v5a` | YOLOv5 raw rows `[N, 5+C]`; sparrow-engine does cxcywh→xyxy + class-aware NMS in Rust. | `Species_Net_MDV5a` (legacy) |
| `softmax` | Single-image softmax → top-k. | SpeciesNet-Crop |
| `sigmoid_window` | Per-window sigmoid (audio sliding-window). | MD_AudioBirds_V1 |
| `tiled_dual_heatmap` | Two-output heatmap (animal mask + class). HerdNet. | HerdNet |
| `tiled_single_heatmap` | One-output heatmap (binary). OWL-T. | OWL-T |

**Cite**: `sparrow-engine/sparrow-engine-types/src/manifest.rs::PostprocessMethod`.

---

### 10.3 Catalog at a glance (production set)

| Model ID | Type | Resolution | Default precision |
|----------|------|------------|-------------------|
| `megadetector-v6-yolov10e` | Detector (animals/vehicles/people) | 1280×1280 | FP16 |
| `megadetector-v6-yolov10e-prov` | MDv6 + provenance fields | 1280×1280 | FP16 |
| `deepfaune-yolo8s` | Detector | 1280×1280 | FP32 (HELD; FP16 audit fails on a borderline image — see Gotchas §13.5) |
| `herdnet-general-2022` | Overhead-detector (dual heatmap) | tiled 512×512 | FP16 |
| `owl-t` | Overhead-detector (single heatmap) | tiled 512×512 | FP16 |
| `speciesnet-crop` | Classifier | 480×480 | FP32 |
| `amazon-cameratrap-v2` | Detector | 1280×1280 | FP16 |
| `md-audiobirds-v1` | Audio binary detector | 1.0s window, 0.3s stride | FP16 |
| `megadet-speciesnet` | **Pipeline alias** (MDv6 → SpeciesNet) | n/a | n/a |
| `Species_Net_MDV5a` | Legacy YOLOv5 detector | 1280×1280 | FP32 |
| `SpeciesNet-Crop` | (re-export pending: NHWC → NCHW) | 480×480 | FP32 |

**Why these defaults**: each is the result of the Phase 3.8 per-model FP16 audit (`docs/research/phase3.8/step1/fp16_audit.md`, `step2/fp16_audit.md`). DeepFaune stays on FP32 pending P3.8-7 closure.

**Cite**: `test_files/sparrow_engine_models/*/manifest.toml`.

---

### 10.4 Pipeline manifests

```toml
[pipeline]                        
id = "megadet-speciesnet"         

[[pipeline.steps]]                
role  = "detector"                
model = "megadetector-v6-yolov10e"

[[pipeline.steps]]                
role  = "classifier"              
model = "speciesnet-crop"         
```

**Why**: named aliases let consumers say "run pipeline X" instead of specifying both models per request.
**What**: a tiny TOML mapping an alias ID to a sequence of `(role, model_id)` steps.
**How**: discovered at boot like model manifests; sparrow-engine validates compatibility (detector before classifier, no audio-after-image, etc.) via `sparrow-engine-core/src/pipeline_compat.rs`. Phase 4.2 also lets you `POST /v1/pipelines` to register an alias at runtime without writing a file.

**Cite**: `test_files/sparrow_engine_models/megadet-speciesnet/pipeline.toml`; `docs/design/phase4.2-cold-start/pipeline_compatibility.md`.

---

### 10.5 Onboarding a new model — the checklist

1. Export to ONNX with **NCHW** layout (use `tf2onnx --inputs-as-nchw` if upstream is NHWC).
2. Confirm NMS is **inside** the graph (detectors). Run `onnxsim` to make sure.
3. Place under `$MODEL_DIR/<your-id>/<files>`.
4. Write `manifest.toml` with the schema above; the directory name MUST equal `[model] id`.
5. Compute SHA-256 of the ONNX file; put it in `manifest.toml`.
6. Validate: `spe models verify` (CLI) or `sparrow_engine.verify_model("<id>")` (Python). If the hash drifts, you have a transit corruption.
7. Smoke test: `spe detect <one-image>.jpg --model <your-id>`.
8. If audio, smoke `spe detect-audio <one-clip>.wav --model <your-id>`.

---

## 11. Phase 4 surface — inference log, drift, provenance

### Section overview

```
                    Per-request opt-in:  ?store=true         
                                  │                          
                                  v                          
                  ┌────────────────────────────────┐         
                  │      InferenceLogRecord        │         
                  │      schema_version = "1.0"    │         
                  │                                │         
                  │  request_id   (UUID v4)        │         
                  │  timestamp_utc (RFC3339 ms)    │         
                  │  media_hash   (sha256-hex)     │         
                  │  model_id                      │         
                  │  device       ("cuda:0", ...)  │         
                  │  inference_ms (f64)            │         
                  │  result       (full payload)   │         
                  │  provenance   (optional)       │         
                  │  drift_metrics (optional)      │         
                  └────────────────┬───────────────┘         
                                   │                         
                                   v                         
                          InferenceLogSink trait             
                                   │                         
                          ┌────────┴────────┐                
                          │                 │                
                          v                 v                
                  StderrJsonLines     (future: sparrow-data  
                  (default sink)       HTTP sink, filesystem,
                   JSON line per       etc.)                 
                   record, stderr-                           
                   locked emit                               
```

**Why**: sparrow-data (deferred sibling) needs a wire-format to ingest from. Phase 4 freezes that format so sparrow-data work doesn't churn sparrow_engine.
**What**: a per-request, opt-in log emission + per-request drift metrics, both shaped by the canonical `InferenceLogRecord`.
**How**: the default sink writes JSON lines to stderr; alternative sinks plug in via the `InferenceLogSink` trait.

**Cite**: `sparrow-engine/sparrow-engine-types/src/inference_log.rs`; `sparrow-engine/sparrow-engine-server/src/sink.rs`; `docs/design/phase4/schema.md`.

---

### 11.1 `?store=true` — the opt-in

```
POST /v1/detect?model=<id>                                 # no log                
POST /v1/detect?model=<id>&store=true                      # emit; warn on sink err
POST /v1/detect?model=<id>&store=true&halt_on_store_failure=true                   
                                                            # emit; 500 on sink err
```

**Why**: most requests don't need a log. Storing every request is expensive and noisy.
**What**: two query params, both default false. `store=true` triggers emit; `halt_on_store_failure=true` makes sink failure surface as HTTP 500 instead of a silent warn.
**How**: emit happens AFTER inference returns successfully. On the error path (model load failure, etc.) no record is emitted.

**Plain words**: "sink" = the destination that receives each log record. The default is "print to stderr"; a future sparrow-data sink might POST to an HTTP endpoint.

---

### 11.2 InferenceLogRecord fields

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | string | Always `"1.0"`. Additive changes keep "1.0"; renames/type-changes bump to "2.0". |
| `request_id` | string | UUID v4, lowercase hex. |
| `timestamp_utc` | string | RFC3339 millis UTC, e.g. `"2026-05-07T12:34:56.789Z"`. |
| `media_hash` | string | Lowercase hex SHA-256 of the request media bytes. Batch detect uses the first image. |
| `model_id` | string | For `/v1/pipeline`, this is the **pipeline_id** — the dedup key, not the detector or classifier. |
| `model_version` | optional string | Always `None` today; reserved for sparrow-data to populate. |
| `device` | string | `"cpu"` or `"cuda:N"`. Never `"auto"` — `Engine::active_device` resolves Auto first. |
| `inference_ms` | f64 | Engine processing time on single-image endpoints; wall-clock on `/v1/detect/batch`. |
| `result` | JSON | The full HTTP response payload, round-tripped unchanged. |
| `provenance` | optional `ProvenanceRecord` | From the manifest `[provenance]` block; omitted if manifest has none. |
| `drift_metrics` | optional `DriftMetrics` | Per-request Tier-1/2 metrics; only populated under `?store=true`. |

**Cite**: `sparrow-engine/sparrow-engine-types/src/inference_log.rs`.

---

### 11.3 DriftMetrics — Tier-1/2 per-request

| Field | What |
|-------|------|
| `confidence_p50` | Median confidence across this request's predictions (nearest-rank percentile, NaN-filtered). |
| `confidence_p95` | 95th percentile confidence. |
| `detections_per_image` | `total_predictions / image_count.max(1)`. |
| `class_distribution_psi` | Optional. PSI vs the manifest's `[drift_reference]` distribution; eps=1e-4 smoothing on both sides; summed `Σ (p_i - q_i) * ln(p_i / q_i)` over the union of class buckets. |

**Why**: Sparrow Web operators want a per-request drift signal that doesn't require talking to a remote service.
**What**: 3 always-on metrics + 1 optional (PSI requires the manifest to declare a reference distribution).
**How**: computed inside the handler after inference; embedded in the InferenceLogRecord.

**Plain words**: "PSI" (Population Stability Index) = a number that says "how different is today's class mix from the training-time class mix". Higher = more drift.

**Tier-3 drift** (reference distributions managed centrally, CUSUM alarms, etc.) is intentionally OUT of sparrow-engine — it lives in the deferred `sparrow-ops` sibling.

**Cite**: `sparrow-engine/sparrow-engine-types/src/drift_metrics.rs`; `sparrow-engine/sparrow-engine-server/src/drift.rs`.

---

### 11.4 Provenance round-trip

```
Manifest:                                         InferenceLogRecord:                     
  [provenance]                                      "provenance": {                       
  training_dataset_id    = "ct-2024"      ───►        "training_dataset_id":    "ct-2024",
  training_experiment_id = "exp-7"                    "training_experiment_id": "exp-7",  
  training_repo_commit   = "abc123"                   "training_repo_commit":   "abc123"  
                                                    }                                     
```

**Why**: when sparrow-data eventually exists, it needs to map each prediction back to the exact model build that produced it. Provenance fields are the join key.
**What**: 3 optional `Option<String>` fields on the manifest, never interpreted by sparrow-engine, faithfully round-tripped.
**How**: `#[serde(default)]` for backward compat; `#[serde(skip_serializing_if = "Option::is_none")]` so absent fields don't bloat the wire.

---

### 11.5 Idempotency at the storage layer (not in sparrow-engine)

```
sparrow-engine (engine):                sparrow-data (deferred sibling):   
  emit one JSON line             treats (media_hash, model_id) as UNIQUE   
  every request                  silently drops duplicates on second insert
        │                              ▲                                   
        └────── HTTP / FS sink ────────┘                                   
```

**Why**: sparrow-engine should not maintain a "did I already see this" cache — that's storage's job.
**What**: sparrow-engine emits a duplicate line on retry; the storage layer (sparrow-data) enforces the UNIQUE constraint.
**How**: the default `StderrJsonLinesSink` does NOT enforce uniqueness; future HTTP sinks will rely on the upstream DB's constraint.

**Cite**: `docs/design/phase4/schema.md § Idempotency`.

---

## 12. Cold-start + lazy load (Phase 4.2)

### Section overview

```
                Server boot (cold)                                                                 
                       │                                                                           
                       │  Phase 4.2 contract:                                                      
                       │   1. Scan SPARROW_ENGINE_MODEL_DIR/<id>/manifest.toml → Catalog           
                       │   2. Bind TCP listener                                                    
                       │   3. Become ready (GET /v1/health                     → "no_models")      
                       │   4. DO NOT load any ORT sessions yet                                     
                       │                                                                           
                       v                                                                           
            ┌──────────────────────┐                 ┌────────────────────┐                        
            │  GET /v1/catalog     │                 │  SPARROW_ENGINE_PRELOAD set │               
                       │  returns all models  │                 │              → eager-load those │
            │  + `loaded: false`   │                 │  IDs at boot       │                        
            └──────────┬───────────┘                 └─────────┬──────────┘                        
                       │                                       │                                   
            ┌──────────┴──────────┬──────────────────┐         │                                   
            v                     v                  v         v                                   
      POST /v1/models/load   POST /v1/detect    POST /v1/pipeline                                  
      (explicit eager)        (lazy: load on    (lazy: load all                                    
                              first request)     step models)                                      
```

**Why**: Phase 4.1 §8.13 surfaced that boot used to eager-load every model, which made server startup take minutes when the catalog had 14 models.
**What**: server now boots in seconds; ORT sessions load on demand (or when explicitly listed in `SPARROW_ENGINE_PRELOAD`).
**How**: `Engine::get_or_load_model` is idempotent and is the single load path for the catalog, `/v1/models/load`, FFI `sparrow_engine_load_model_by_id`, and all 5 CLI inference sites.

**Cite**: `docs/master_plan.md § Phase 4.2`; `docs/design/phase4.2-cold-start/`.

---

### 12.1 What lazy-loads vs. what doesn't

| Action | Triggers load? |
|--------|----------------|
| Server boot | NO (catalog only) |
| `SPARROW_ENGINE_PRELOAD=id1,id2` | YES, those IDs only, at boot |
| `GET /v1/catalog` | NO |
| `GET /v1/models` | NO (lists already-loaded only) |
| `POST /v1/models/load` | YES (explicit, idempotent) |
| `POST /v1/pipeline?detector=&classifier=` | YES (both step models) |
| `POST /v1/pipeline?pipeline=<id>` | YES (all step models of the alias) |
| `POST /v1/detect` etc. (non-pipeline) | **(Phase 4.2)** NO — requires explicit `POST /v1/models/load` first |
| `POST /v1/detect` etc. (2026-05-14 follow-up `d31f33b`) | YES — lazy `get_or_load_model` inside `run_blocking` |

**Phase 4.2 nuance**: original Phase 4.2 kept non-pipeline endpoints requiring explicit load. The 2026-05-14 follow-up extended lazy-load through `/v1/detect`, `/v1/classify`, `/v1/audio/detect` for Sparrow Web worker compatibility.

**Cite**: `docs/changelog.md § 2026-05-14` (commit `d31f33b`).

---

### 12.2 `SPARROW_ENGINE_PRELOAD` semantics

```
SPARROW_ENGINE_PRELOAD=megadetector-v6-yolov10e,md-audiobirds-v1 sparrow-engine-server
                        │                                                             
                        v                                                             
   Boot: load both models eagerly (parallel, blocking until both succeed).            
   Unknown ID → fail boot with a clear log line.                                      
   Empty      → no preload (default).                                                 
```

**Why**: production deployments often have a known "hot" subset that should always be ready.
**What**: a comma-separated list of model IDs to load at boot.
**How**: validated against the catalog; any unknown ID aborts boot.

---

### 12.3 Runtime pipeline aliases

```
POST /v1/pipelines                                            
{                                                             
  "id": "my-alias",                                           
  "steps": [                                                  
    {"role": "detector", "model": "megadetector-v6-yolov10e"},
    {"role": "classifier", "model": "speciesnet-crop"}        
  ]                                                           
}                                                             
```

| Behavior | Detail |
|----------|--------|
| Slug validation | ID must be `[a-z0-9-]+`. |
| Idempotent same-def | Posting the same definition again → 200. |
| Conflict on different-def | Posting a different definition under the same ID → 409. |
| `?replace=true` | Override an existing alias with a different definition. |
| Per-alias write lock | Concurrent creates against the same ID are serialized. |
| Optional persistence | If a `pipeline.toml` is written under `SPARROW_ENGINE_MODEL_DIR/<id>/`, the alias survives restart. |

**Cite**: `sparrow-engine/sparrow-engine-server/src/handlers/pipelines_mgmt.rs`; manual test rows in `docs/review/phase4.2-manual-test/round_01/manual_test_plan.md § §4`.

---

### 12.4 `GET /v1/catalog` shape

```json
[                                                                                                            
  {"model_id": "megadetector-v6-yolov10e", "model_type": "detector",   "framework": "onnx", "loaded": true}, 
  {"model_id": "speciesnet-crop",          "model_type": "classifier", "framework": "onnx", "loaded": false},
  {"model_id": "md-audiobirds-v1",         "model_type": "audio",      "framework": "onnx", "loaded": false},
  {"model_id": "megadet-speciesnet",       "model_type": "pipeline",   "framework": "alias", "loaded": false}
]                                                                                                            
```

**Note**: response is a **flat JSON array**, not an envelope. (Phase 4.2 manual test MT-4.2-7 corrected docs that had assumed an envelope shape.)

**Cite**: `sparrow-engine/sparrow-engine-server/src/handlers/catalog.rs`.

---

## 13. Gotchas + edge cases (the real-world traps)

### Section overview

```
        ┌───────────────────────────────────────────────────────┐
        │  Things that look fine but break in production:       │
        │                                                       │
        │  1. cuDNN 9.8 Conv bug (must be ≥ 9.10)               │
        │  2. ORT ABI version pin (<1.26)                       │
        │  3. GPU teardown heap corruption (MT-17)              │
        │  4. fork() vs the Engine singleton                    │
        │  5. DeepFaune FP16 borderline detection loss          │
        │  6. LD_LIBRARY_PATH for CLI binaries                  │
        │  7. NHWC ONNX models                                  │
        │  8. eza/exa aliases capturing icons in path vars      │
        │  9. `--flavor gpu` install on no-GPU host             │
        │  10. ORT 1.26.x dropping symbol version VERS_1.25.1   │
        │  11. Dependency version pins — today vs. planned      │
        └───────────────────────────────────────────────────────┘
```

**Why this section exists**: each item below cost a real session to track down. Reading them saves you the next session.

---

### 13.1 cuDNN 9.8 vs ≥ 9.10

```
Session creation crashes:                                       
  "[E:onnxruntime:default] No valid engine configs for ConvFwd_"
  + dozens of engine-config dumps                               
  → SpeciesNet fails to load on RTX 6000 Ada (sm_89)            
```

**Why**: cuDNN 9.8 has a Conv-engine bug with asymmetric padding; PyTorch and TensorFlow wheels both bundle cuDNN 9.8 by default, so the dev box picks one and silently breaks.
**What**: sparrow-engine requires cuDNN ≥ 9.10. The installer's layer-2 probe enforces this (exit 11).
**Fix**: `uv pip install --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'`. `scripts/ort-env.sh` searches `~/.local/cudnn/nvidia/cudnn/lib` before falling back to PyTorch's bundle.

**Cite**: `docs/lessons.md § cuDNN 9.8 has a Conv engine bug` (MT-15).

---

### 13.2 ORT ABI version pin (`<1.26`)

```
  nm -D --with-symbol-versions libonnxruntime.so.1.25.1   → VERS_1.25.1   present                   
  nm -D --with-symbol-versions libonnxruntime.so.1.26.0   → VERS_1.25.1   DROPPED                   
                                                          (new VERS_1.26.0 namespace)               

If your wheel was compiled against the `ort` Rust crate's 1.25.x ABI:                               
  installing onnxruntime 1.26+ at runtime                 → ImportError on `_sparrow_engine_core.so`
```

**Why**: ORT bumps its symbol-version namespace between minor releases. The `ort` Rust crate compiles against a specific ORT minor; running against a different one breaks linking.
**What**: the sparrow-engine Python wheels pin `onnxruntime>=1.25.1,<1.26` (CPU) and `onnxruntime-gpu>=1.25.1,<1.26` (GPU).
**Fix**: do NOT bump onnxruntime past 1.25.x until the `ort` Rust crate ships an rc.13+ targeting 1.26+ (tracked at `docs/master_plan.md § P4D-RT-1`).

**Cite**: MT-4.1-14; memory note `ORT ABI versioning`.

---

### 13.3 GPU teardown heap corruption (MT-17)

```
$ spe pipeline IMG.jpg --device cuda:0                                 
... inference completes correctly ...                                  
*** corrupted double-linked list (SIGABRT, exit 134)  ← at PROCESS EXIT
```

**Why**: ORT's CUDA EP retains hooks into `libonnxruntime_providers_cuda.so`. glibc's `_dl_fini` can finalize the shared object BEFORE Rust's field-drops fire, so the session-drop path reads freed memory.
**What**: sparrow-engine applies a Drop-order mitigation (explicit `HashMap::clear()` inside `Drop::drop` + leak the SessionBuilder Arc). Pre-mitigation: 10–33% per process; post: ~5% residual.
**Fix**: nothing you can do at the call site; the residual is upstream (pykeio/ort #564, closed `not_planned`). Reliability-sensitive paths should prefer CPU or subprocess-per-call.

**Cite**: `docs/lessons.md § MT-17`; `docs/bugs.md MT-17`.

---

### 13.4 fork() + Engine singleton

```
import multiprocessing as mp                                                 
import sparrow_engine                                                        
sparrow_engine.init(device="cuda:0")                                         

# Spawning a child process:                                                  
ctx = mp.get_context("fork")   ← BREAKS: ENGINE_EXISTS bool leaks into child;
                                  child's Engine::new() returns error.       

ctx = mp.get_context("spawn")  ← OK: child has a fresh process state.        
```

**Why**: `ENGINE_EXISTS` is an `AtomicBool` that prevents two engines in one process. `fork()` duplicates it into the child, which then thinks an engine already exists.
**What**: a hard constraint: Python multiprocessing MUST use `spawn`, not `fork`, when Sparrow Engine is loaded in the parent.
**Fix**: `mp.set_start_method("spawn", force=True)` early in your script.

**Cite**: `docs/master_plan.md § Engine + safety invariants`.

---

### 13.5 DeepFaune FP16 borderline detection loss

```
DeepFaune FP32:                    DeepFaune FP16:
  244 detections on test set         243 detections (loses one borderline)
```

**Why**: a single image's lowest-confidence detection drops below threshold under FP16 quantization. The change-decoder experiment (P3.8-7) proved it's NOT a JPEG-decoder issue.
**What**: DeepFaune stays HELD on FP32 by default in the manifest. All other vision + audio models defaulted to FP16 in Phase 3.8.
**Fix**: don't flip DeepFaune to FP16 without re-running the per-model audit.

**Cite**: `docs/ideas.md § P3.8-1, P3.8-7`.

---

### 13.6 LD_LIBRARY_PATH for CLI binaries

```
$ /home/me/.sparrow_engine/bin/spe detect IMG.jpg
error while loading shared libraries: libonnxruntime.so.1: cannot open shared object file
```

**Why**: the CLI binary doesn't bake an RPATH/RUNPATH that points at the wheel-installed ORT. It uses the dynamic linker, which needs `LD_LIBRARY_PATH`.
**What**: the sparrow-engine install wrapper sets up `scripts/ort-env.sh` to point at the right place; the rc-file block sources it via `PATH` + an alias.
**Fix**: when invoking the CLI directly (not via the `spe` alias), prepend `LD_LIBRARY_PATH=<onnxruntime-libs>`. Phase 4.2 manual test MT-4.2-12 surfaced this.

---

### 13.7 NHWC ONNX models

```
ORT CUDA EP + NHWC + dynamic shapes  →  SafeInt overflow in Conv
                                         (ORT issues #27912, #12288)
```

**Why**: ORT's CUDA EP has known bugs with NHWC + dynamic shapes.
**What**: sparrow-engine rejects non-NCHW models at manifest load.
**Fix**: re-export with `tf2onnx --inputs-as-nchw` and run through `onnx-simplifier`. Two of the catalog models (`SpeciesNet-Crop`, `Species_Net_MDV5a`) need this re-emission step; sparrow-engine refuses them otherwise.

**Cite**: `docs/master_plan.md § Format + interface invariants`; sparrow integration's `manifest_reemit_*.sh` scripts.

---

### 13.8 eza/exa aliases corrupting paths

```
$ GPU_WHL=$(ls -t target/wheels/*.whl | head -1)         
$ pip install "$GPU_WHL"                                 
Error: Expected package name, found '-I'                 

# Reason: zsh aliased `ls` to `eza --git --icons=always`.
# `$GPU_WHL` captured "-I <icon> target/wheels/...whl"   
```

**Why**: zsh/bash users alias `ls` to `eza` or `exa`, which inject git-status flags + nerd-font icons before file names. When you capture `$(ls ...)` in a variable, you capture the noise too.
**What**: a paste-block trap surfaced in MT-4.2-18.
**Fix**: use `\ls` (backslash-ls) to bypass aliases. Same applies to `\grep`, `\cat`, etc. inside test paste-blocks.

**Cite**: memory note `manual test paste-blocks`.

---

### 13.9 SIGPIPE

```
$ spe detect /photos/ --print --format json | head -1
# spe crashes with SIGPIPE  ← old behavior
```

**Why**: when the downstream pipe closes early (e.g., `| head`), the writer side of the pipe gets SIGPIPE; Rust's default handler kills the process.
**What**: sparrow-engine installs a SIGPIPE handler that translates the signal into a graceful exit.
**Fix**: nothing you need to do at the call site; this is just informational.

**Cite**: `docs/tech_report/06_gotchas_and_constraints.md`.

---

### 13.10 Cross-flavor refusal

```
$ bash installer/sparrow-engine-install.sh --flavor gpu          
# Current install is CPU.                                        
Error: cross-flavor install attempted without --reprobe (exit 12)
```

**Why**: switching between CPU and GPU flavors requires uninstalling the current one first (different ORT runtime deps; cross-contamination crashes).
**What**: the wrapper refuses cross-flavor installs unless `--reprobe` is passed.
**Fix**: `bash installer/sparrow-engine-install.sh --reprobe --flavor gpu` (or `--flavor cpu`). The reprobe runs the GPU quality check BEFORE the destructive uninstall, per `F-R3-1`.

---

### 13.11 Dependency version pins — what's enforced today vs. what's planned

```
ENFORCEMENT TIERS (today)                                                              
   1. Wheel METADATA Requires-Dist:                  ENFORCED at pip install           
        sparrow-engine:     onnxruntime>=1.25.1,<1.26         (pip refuses out-of-range
        sparrow-engine-gpu: onnxruntime-gpu>=1.25.1,<1.26      installs)               
   2. installer/probe_gpu_quality.{sh,ps1}:          ENFORCED at install               
        cuDNN >= 9.10                                 (exit 11 if violated)            
   3. NVIDIA driver >= 550.x                         ENFORCED via cuDNN probe          
                                                      side-effect; weak                
   4. Provides-Dist / Conflicts-Dist                 ADVISORY ONLY in pip ≥22          
                                                      (does not block install)         
```

**Why**: people installing sparrow-engine into mismatched environments produce the worst-class of "it loads but crashes" bugs (MT-4.1-14 ORT 1.26 symbol-namespace drop; cuDNN 9.8 Conv crash; CPU/GPU wheel coexistence).
**What**: today's enforcement leans on pip's hard `Requires-Dist` ceiling for ORT + the installer's cuDNN probe for the GPU stack. It does NOT yet enforce: matched sparrow-engine CPU/GPU vs. installed ORT EP, runtime detection of an ORT version drift after install, or refusal-to-import when both `sparrow-engine` and `sparrow-engine-gpu` are present.
**How (today)**: trust pip's resolver + the installer probe. The user manual (this doc) is the secondary safety net — readers see §3.1, §6.4, §13.2 before they hit production.

**Open work — STRICTER enforcement (planned, not built; review-round-1 comment 2026-05-19)**:

| ID | Planned guard | Where it would live | Today's gap it closes |
|----|---------------|----------------------|-----------------------|
| STR-1 | Runtime ORT-version check inside `Engine::new`; refuse to initialize if `OrtGetVersionString()` is outside the pinned range. | `sparrow-engine-core/src/engine.rs` + flavor crates | Catches the case where a user `pip install onnxruntime==1.26.0` AFTER installing sparrow-engine (pip's pin protects the install moment, not subsequent upgrades). |
| STR-2 | Refuse-to-import guard inside `sparrow_engine/__init__.py` if `sparrow_engine_gpu` package is also resolvable in `sys.modules` / `sys.path`. | `sparrow-engine-python/python/sparrow_engine/__init__.py` | Catches the two-wheels-in-one-env footgun. |
| STR-3 | Driver-version probe at install (not just cuDNN). | `installer/probe_gpu_quality.{sh,ps1}` | Drivers older than 550.x can load CUDA EP but crash inside cuDNN; today this only surfaces as a runtime "no valid engine configs" message. |
| STR-4 | Manifest schema pin in catalog (manifest declares `min_sparrow_engine_version`); engine refuses to load if the running sparrow-engine is older. | `sparrow-engine-types/src/manifest.rs` | Catches the case where a manifest is updated for a new sparrow-engine feature but the deploying env is on an older sparrow_engine. |
| STR-5 | `spe doctor` CLI subcommand that prints + checks every enforced pin in one place. | `sparrow-engine-cli/src/main.rs` | Diagnostic; not a guard. Surfaces drift before it bites. |

**Disposition**: STR-1..STR-5 captured here for the release-prep cycle (`docs/release_dev_plan.md`); none are blockers for today's dev-cycle handoff. The user requested they land before public release so the wrong-version footgun never reaches production users.

**Cite**: user directive (review-round-1, 2026-05-19); existing pin sites — `sparrow-engine/sparrow-engine-python/pyproject.toml:55`, `sparrow-engine/sparrow-engine-python/build.sh` GPU sed pattern, `installer/probe_gpu_quality.{sh,ps1}`.

---

> the LD_LIBRARY_PATH issue is also worth mentioning upfront when release

**Status (revision 2, 2026-05-19)**: BOTH ADDRESSED.
> (a) Strict version enforcement → new §13.11 "Dependency version pins — what's enforced today vs. what's planned" with the planned guards (STR-1..STR-5) docketed for the release-prep cycle.
> (b) LD_LIBRARY_PATH upfront → callout added to §2.3 (right under the install footprint diagram), cross-linking §13.6.

---

## 14. Performance characteristics

### Section overview

```
                    Workload                       Latency (RTX 6000 Ada)              
   ─────────────────────────────────────────────  ───────────────────────────          
   sparrow-engine-gpu  MDv6 (yolov10-e) FP16               13.46 ms / image (median)   
   sparrow-engine-gpu  MD_AudioBirds_V1 FP16                8.52 ms / 1.0s window (p50)
   sparrow-engine-gpu  Pipeline (MDv6 → SpeciesNet)        ~60 ms / image              
   sparrow-engine-cpu  MDv6 FP32 (debian:bookworm-slim)    ~1.9 s  / image             
   sparrow-engine-gpu  Cold start (CUDA EP init + load)    ~2.8 s  (first request only)
   sparrow-engine-server  HTTP cold boot (no preload)      Sub-second to ready         
```

**Why**: production deployments care more about p95 latency than peak GPU FLOPs.
**What**: representative numbers from the Phase 3.8 benchmarks; not contractual SLOs.
**How**: full methodology in `docs/benchmarks.md`; each row reproducible via `scripts/bench_*.py`.

**Comparative**: 1.96× faster than PytorchWildlife FP16 on MDv6 (Phase 3.8 numbers); 2.10× faster than PW on audio (post-FP16).

**Cite**: `docs/benchmarks.md § 8, § 11, § 12`; `docs/master_plan.md § Benchmarks` paragraph.

---

### 14.1 Where Rust wins

| Axis | Rust win | Why |
|------|----------|-----|
| Cold start (cdylib load) | ~2.8s vs ~10s Python | No interpreter startup, no `import torch` overhead. |
| Per-image steady-state | 1.5×–2.1× vs PW PyTorch | No Python GIL contention; ORT FP16 + IoBinding (GPU). |
| Docker image size | 167 MB CPU vs ~2 GB PyTorch | No Python runtime, no PyTorch wheel; just the cdylib + minimal deps. |
| Memory footprint | ~1.2 GB peak RSS vs ~2.7 GB PW | No Python heap, no PyTorch tensor cache. |

---

### 14.2 Where Rust does NOT win

| Axis | Notes |
|------|-------|
| Numerical accuracy | Exact parity with PytorchWildlife on detection counts within FP16/FP32 noise. |
| Algorithmic optimization | Same ONNX → ORT path; ORT does the heavy lifting either way. |
| Multi-batch GPU throughput | sparrow-engine runs one image at a time (single-image latency-optimized). Batch is FYI per-image with sequential dispatch. |

---

### 14.3 Idle reaper memory behavior

```
Long-running sparrow-engine-server with 14 models:                           
  Without reaper:  all 14 sessions resident → ~10 GB GPU memory (and growing)
  With reaper (default 30 min, keep 1):                                      
     14 sessions briefly, then drops to 1 + whatever's in active use         
                                            → ~1.5–2 GB steady-state         
```

**Why**: ORT sessions hold GPU memory until dropped. A server that's been up for hours can pin out the GPU.
**What**: the reaper drops idle sessions, keeping the keep-last-N most-recently-used. Default 30 min / keep 1.
**How**: `Engine::reap_idle_models` runs in a background tokio task; tunable via `SPARROW_ENGINE_IDLE_UNLOAD_SEC` and `SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N`.

---

## 15. Sparrow Studio integration (Web + Local)

### Section overview

```
                    Sparrow Studio                                                           
        ┌────────────────────┴────────────────────┐                                          
        │                                         │                                          
   Sparrow Web                              Sparrow Local                                    
   (server, multi-user)                     (desktop, single-user)                           
        │                                         │                                          
   HTTP API                                  Native DLL                                      
        │                                         │                                          
        v                                         v                                          
   sparrow-engine-server                              libsparrow_engine.{so,dll,dylib}       
   (Docker, CPU + GPU images)                (cdylib + sparrow_engine.h + NativeMethods.g.cs)
```

**Why**: sparrow-engine has two primary consumers with very different deployment models. Each has its own integration contract.
**What**: Web ships as a Docker image consumed via HTTP; Local ships as a native DLL consumed via `[DllImport("sparrow_engine")]`.
**How**: Web went live 2026-05-14..16; Local is the current downstream effort on a Windows machine.

---

### 15.1 Sparrow Studio Web — what's locked in

```
sparrow/sparrow-engine/                            (image-pin contract)          
├── sparrow_engine.version                         tag + digest pin + ORT version
├── sparrow-engine-source.toml                     human-readable provenance     
├── sync.lock                             machine-checkable hashes               
└── docker-compose.override.yaml.example  dev bind-mount template                
```

| Item | Status |
|------|--------|
| Image: `sparrow-engine-server:sparrow-combined` (CPU 168 MB) | LIVE 2026-05-14 |
| Image: `sparrow-engine-server-gpu:sparrow-combined` (GPU 3.67 GB) | LIVE 2026-05-14 |
| ORT runtime | 1.25.1 |
| `ort` Rust crate | `2.0.0-rc.12` |
| Cutover mode | **Combined cutover** (NOT two-cutover — see Phase 4.1's design rev) |
| Pin-sync workflow | Manual `sparrow/scripts/sync_bongo.sh` (Option A) — Option C CI auto-PR DEFERRED as SW-1 |
| 14/14 catalog models | PASS verified against real cameratrap image 2026-05-15 |

**Cite**: `docs/master_plan.md § Sparrow Studio Web Integration`; `docs/design/sparrow-web-integration/final.md`.

---

### 15.2 Sparrow Studio Local — what's in flight

```
Linux side (bongo_dev):                    Windows side (Sparrow Local):             
  ┌──────────────────────┐                 ┌──────────────────────┐                  
  │ libsparrow_engine.{so}        │                 │ sparrow_engine.dll (build)    │
  │ sparrow_engine.h (cbindgen)   │   port  ──►     │ NativeMethods.g.cs   │         
  │ NativeMethods.g.cs   │                 │ Avalonia desktop app │                  
  │ 32 sparrow_engine_* exports   │                 │ [DllImport("sparrow_engine")] │
  │ G5 invariant         │                 │ G5-equivalent check  │                  
  └──────────────────────┘                 └──────────────────────┘                  
```

| Step | Status |
|------|--------|
| Sparrow Engine-side primitives | READY 2026-05-06 (Phase 3.8 Phase C Wave 5 + Phase 4.1 §1.11 §9.1) |
| C# `NativeMethods.g.cs` | Auto-generated from csbindgen; checked in |
| Windows `sparrow_engine.dll` build | IN PROGRESS on Windows machine |
| Linux-mocked C ABI smoke | SIGNED OFF in Phase 4.1 §1.11 + §9.1 |
| End-to-end desktop-app smoke | NOT YET (the Windows hands-on goal) |
| MT-SL ticket prefix | Reserved for any Sparrow Engine-side defects found on Windows |

**Why this is separate from Web**: Web consumes sparrow-engine via HTTP (long-running multi-tenant server). Local consumes sparrow-engine as a native DLL (per-process single-user desktop). Different latency profile, different distribution, different platform priority.

**Cite**: `docs/master_plan.md § Sparrow Studio Local Integration`; `docs/ideas.md § Sparrow Studio Local Integration follow-ups`.

---

### 15.3 What happens if Windows surfaces a sparrow-engine defect

```
Windows hands-on finds defect                                 
            │                                                 
            v                                                 
File MT-SL-N ticket in:                                       
   docs/ideas.md § Sparrow Studio Local Integration follow-ups
            │                                                 
            v                                                 
If defect is in sparrow-engine (not Sparrow app):             
   → branch off main                                          
   → fix                                                      
   → re-run Linux G5 + Phase 4.1 §1.11 + §9.1                 
   → cross-build on Windows                                   
   → close ticket                                             
```

**Why**: every Windows-surfaced sparrow-engine defect should be reproducible on Linux first, fixed on Linux, then re-tested on Windows.
**What**: a ticket lifecycle that uses the existing MT-N convention from Phase 4.1.
**How**: ticket IDs are `MT-SL-1`, `MT-SL-2`, … parallel to `MT-4.1-N`. Sparrow Local desktop app issues stay in the `sparrow_studio_local` repo.

---

<!-- SUMMARY-MARKER -->

## Summary

```
                              The sparrow-engine surface at a glance                          

      What it is                  How you call it                What it produces             
   ───────────────────       ───────────────────────────       ────────────────────           
   Rust ML engine for        1. sparrow-engine CLI                       Normalized bbox [0,1]
   ONNX vision + audio       2. sparrow-engine Python wheel              + class + confidence 
   (camera-trap species,     3. sparrow-engine-client (HTTP SDK)           or                 
    bioacoustics)            4. sparrow-engine-server (HTTP API)         Audio time-ranges    
                             5. libsparrow_engine cdylib (C ABI)         + max_confidence     
                                                                 or                           
                                                                Pipeline crops                
                                                                + top_k labels                
```

| Metric | Value |
|--------|-------|
| Workspace crates | 7 (types, core, cpu, gpu, server, cli, python) |
| Device flavors | 2 (cpu, gpu — mutually exclusive at build time) |
| C ABI exports | 32 `sparrow_engine_*` symbols (byte-identical across flavors) |
| HTTP endpoints | 15 (5 inference + 8 management + 2 health) |
| CLI commands | 9 (detect, classify, detect-audio, pipeline, models, device, init, hash, day-night) |
| Python public functions | 14 |
| Catalog models (production) | 9 image/audio + 1 pipeline alias + 2 legacy + 2 quarantined (NHWC, yolo_v5) |
| `SPARROW_ENGINE_*` env vars (server) | 14 |
| Install paths | 4 (from-source, GH Releases, pip, Docker) |
| OS support | Linux x86_64, macOS arm64/x86_64, Windows x86_64 |
| Default GPU | NVIDIA RTX 6000 Ada (sm_89); driver ≥550, CUDA 12.6, cuDNN ≥9.10 |
| Headline latency (MDv6 FP16, RTX 6000 Ada) | 13.46 ms / image median |
| Headline audio latency (MD_AudioBirds_V1 FP16) | 8.52 ms / 1.0s window p50 |
| Wheel ORT pin | `>=1.25.1, <1.26` |

**Next steps** (none of these block normal use of sparrow-engine):

- Sparrow Studio Local Windows hands-on (in progress)
- File `MT-SL-N` tickets for any Sparrow Engine-side defects surfaced on Windows
- Sibling repos (`sparrow-data`, `sparrow-ops`) — DEFERRED until forcing-function triggers per `docs/master_plan.md § Sibling Projects`
- Public release prep (`docs/release_dev_plan.md`) — held until all dev + sibling work converges

---

