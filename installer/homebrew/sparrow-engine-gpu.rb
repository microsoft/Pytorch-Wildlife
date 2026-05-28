class SparrowEngineGpu < Formula
  desc "Camera-trap ML inference engine — GPU (NVIDIA CUDA) CLI binary"
  homepage "https://github.com/microsoft/Pytorch-Wildlife"
  license "MIT"
  version "0.1.10"

  # RP-4 + RP-17 (2026-05-27): canonical GPU formula template. Substitution
  # workflow mirrors the CPU formula (`sparrow-engine.rb`):
  #
  #   1. CI cuts vX.Y.Z, publish-cli-release-assets attaches the GPU tarball
  #      sparrow-engine-gpu-X.Y.Z-linux-x86_64.tar.gz + .sha256 to the GH
  #      Release.
  #   2. Operator fetches the .sha256 sidecar, replaces the placeholder
  #      below, copies this file to microsoft/homebrew-sparrow-engine
  #      tap repo at Formula/sparrow-engine-gpu.rb, commits + pushes.
  #
  # GPU is Linux x86_64 only — NVIDIA CUDA does not exist on macOS, and
  # Linux aarch64 has no matching tarball in the RP-4 release matrix.

  on_linux do
    on_intel do
      url "https://github.com/microsoft/Pytorch-Wildlife/releases/download/v#{version}/sparrow-engine-gpu-#{version}-linux-x86_64.tar.gz"
      sha256 "REPLACE_WITH_gpu_linux-x86_64_sha256"
    end
  end

  def caveats
    <<~EOS
      sparrow-engine-gpu auto-discovers ALL required CUDA sidecar libraries
      from common host locations at startup via a small wrapper script that
      brew installs alongside the binary. The wrapper covers the 6 hard-
      required libs from probe_gpu_quality.sh (libcudnn.so.9, libnvjpeg.so.12,
      libnvrtc.so.12, libcudart.so.12, libcublas.so.12 + libcublasLt.so.12,
      libcurand.so.10, libcufft.so.11) and checks (in order, per lib):

        1. SPARROW_ENGINE_CUDA_LIB_DIR (user override; honoured if set,
           wrapper skips all auto-discovery for that flavor)
        2. ~/.sparrow-engine/cuda-sidecars/lib/python*/site-packages/nvidia/<pkg>/lib
        3. /usr/lib/python3/dist-packages/torch/lib   (Lambda Stack / PyTorch)
        4. /usr/lib/python3/dist-packages/tensorflow  (Lambda Stack / TF)
        5. /usr/lib/python3/dist-packages/jax_cuda12_plugin  (JAX)
        6. /usr/local/cuda/lib64                      (NVIDIA CUDA toolkit)
        7. /usr/lib/x86_64-linux-gnu                  (Ubuntu apt nvidia-*)
        8. /opt/nvidia/cuda/lib64                     (HPC-style CUDA installs)
        9. /usr/lib64                                 (RHEL / Fedora)

      If a required lib is not found in any of these, install ONE of:

      Option A — system CUDA (Ubuntu / Debian, recommended for servers):
        sudo apt install nvidia-cuda-toolkit nvidia-cudnn

      Option B — Python sidecar wheels (no root, no system CUDA):
        uv venv ~/.sparrow-engine/cuda-sidecars --python 3.11
        ~/.sparrow-engine/cuda-sidecars/bin/pip install nvidia-cudnn-cu12 \\
            nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 \\
            nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

      Verify the host is ready (NO env-var tweaks required):
        spe-gpu device       # expected: {"device":"cuda:0"}

      Full GPU install path:
        https://github.com/microsoft/Pytorch-Wildlife/blob/sparrow-engine-dev/docs/user-manual.md

      The tarball is ~256 MB — bundles libonnxruntime + ORT CUDA provider
      sidecars. NVIDIA-managed shared libraries (cuDNN / cuBLAS / cuRAND /
      cuFFT / nvJPEG / nvRTC / CUDA runtime) are NOT bundled (NVIDIA license
      forbids redistribution).
    EOS
  end

  def install
    # 1. Lay down the bundled tarball under libexec/{bin,lib,...}. The
    # in-binary ort_resolver canonicalises current_exe() and walks one dir
    # up from bin/ to find lib/, so libexec/lib/libonnxruntime.so.X.Y.Z is
    # auto-discovered. GPU additionally prepends libexec/lib to
    # LD_LIBRARY_PATH at startup so the CUDA provider sidecars next to
    # libonnxruntime get picked up too.
    libexec.install Dir["*"]

    # 2. Write a small POSIX shell wrapper that auto-discovers cuDNN +
    # nvJPEG from common host locations BEFORE exec'ing the real binary.
    # This eliminates the manual LD_LIBRARY_PATH= dance that production
    # users would otherwise need. Search order matches the caveats block.
    (bin/"spe-gpu").write <<~WRAPPER
      #!/bin/sh
      # sparrow-engine-gpu wrapper — auto-discovers cuDNN / nvJPEG / CUDA libs
      # from common host locations. Edit this file at your own risk; brew
      # rewrites it on every `brew (re)install sparrow-engine-gpu`.
      #
      # User override: set SPARROW_ENGINE_CUDA_LIB_DIR to a colon-separated
      # list of directories to prepend to LD_LIBRARY_PATH. The wrapper then
      # skips its own auto-discovery for that flavor.

      SE_LIBEXEC="#{libexec}"

      add_lib_dir() {
        # $1 must contain the required .so to be useful; caller already
        # checked existence.
        case ":$LD_LIBRARY_PATH:" in
          *":$1:"*) ;;
          *) LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
      }

      # discover_sidecar <pip-package-subpath> <library-filename>
      #
      # Walks the standard CUDA sidecar search path (pip wheel under
      # ~/.sparrow-engine/cuda-sidecars, then PyTorch/TF bundles, then
      # system CUDA, apt, HPC, RHEL) and prepends the first dir containing
      # the named library to LD_LIBRARY_PATH. Returns silently if no dir
      # matches — the runtime check in probe_gpu_quality.sh / spe-gpu
      # itself produces the operator-facing error.
      discover_sidecar() {
        for dir in \\
          "$HOME/.sparrow-engine/cuda-sidecars"/lib/python*/site-packages/nvidia/$1/lib \\
          /usr/lib/python3/dist-packages/torch/lib \\
          /usr/lib/python3/dist-packages/tensorflow \\
          /usr/lib/python3/dist-packages/jax_cuda12_plugin \\
          /usr/local/cuda/lib64 \\
          /usr/lib/x86_64-linux-gnu \\
          /opt/nvidia/cuda/lib64 \\
          /usr/lib64; do
          if [ -e "$dir/$2" ]; then
            add_lib_dir "$dir"
            return 0
          fi
        done
        return 1
      }

      if [ -n "$SPARROW_ENGINE_CUDA_LIB_DIR" ]; then
        # Honour user override verbatim — no auto-discovery.
        LD_LIBRARY_PATH="$SPARROW_ENGINE_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      else
        # E-R2-3: cover ALL 6 hard-required libs from probe_gpu_quality.sh
        # _required_libs (lines 144-150). Each entry pairs the pip-wheel
        # subpath under nvidia/<pkg>/lib with the canonical .so filename.
        # libcublasLt.so.12 ships in the same dir as libcublas.so.12 so the
        # cublas entry covers both.
        discover_sidecar cudnn         libcudnn.so.9
        discover_sidecar nvjpeg        libnvjpeg.so.12
        discover_sidecar cuda_nvrtc    libnvrtc.so.12
        discover_sidecar cuda_runtime  libcudart.so.12
        discover_sidecar cublas        libcublas.so.12
        discover_sidecar curand        libcurand.so.10
        discover_sidecar cufft         libcufft.so.11
      fi

      export LD_LIBRARY_PATH
      exec "$SE_LIBEXEC/bin/spe-gpu" "$@"
    WRAPPER
    (bin/"spe-gpu").chmod 0755
  end

  test do
    # `--version` exercises clap + the wrapper script + the in-binary
    # resolver, but not ORT init. ORT init requires an NVIDIA GPU which
    # the brew test sandbox cannot guarantee. Operators verify device
    # with `spe-gpu device` post-install.
    #
    # Version-agnostic shape check (B-03 fix): clap emits the binary name
    # followed by CARGO_PKG_VERSION + the " (GPU flavor)" suffix from
    # `sparrow-engine-cli/src/main.rs:45`. Pinning to the formula's
    # `version` string drifts whenever CARGO_PKG_VERSION bumps ahead of
    # the brew tap.
    assert_match(/\Aspe-gpu \d+\.\d+\.\d+ \(GPU flavor\)\Z/,
                 shell_output("#{bin}/spe-gpu --version").strip)
  end
end
