class SparrowEngine < Formula
  desc "Camera-trap ML inference engine (sparrow-engine CLI binary)"
  homepage "https://github.com/microsoft/Pytorch-Wildlife"
  license "MIT"
  version "0.1.10"

  # RP-4 (2026-05-26): the formula points at the GH Release tarballs produced
  # by .github/workflows/release.yml § build-cli-* and attached by
  # publish-cli-release-assets. Layout is:
  #   sparrow-engine-cpu-<ver>-<platform>/
  #   ├── bin/spe(.exe)
  #   ├── lib/libonnxruntime.{so.X.Y.Z,dylib}
  #   ├── README.md
  #   └── VERSION
  #
  # SHA256 placeholders MUST be replaced before tap publish — the cut-release
  # script fetches the .sha256 sidecars from the GH Release and substitutes
  # them in. Until then, this formula will not install (brew validates the
  # checksum before unpacking).

  on_macos do
    on_arm do
      url "https://github.com/microsoft/Pytorch-Wildlife/releases/download/v#{version}/sparrow-engine-cpu-#{version}-macos-aarch64.tar.gz"
      sha256 "REPLACE_WITH_macos-aarch64_sha256"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/microsoft/Pytorch-Wildlife/releases/download/v#{version}/sparrow-engine-cpu-#{version}-linux-x86_64.tar.gz"
      sha256 "REPLACE_WITH_linux-x86_64_sha256"
    end
  end

  def install
    # The tarball roots at sparrow-engine-cpu-<ver>-<platform>/ — brew strips
    # one level of tarball root automatically, so Dir["*"] sees bin/, lib/,
    # README.md, VERSION directly.
    #
    # We install into libexec/ (not bin/ + lib/ directly under prefix) and
    # symlink bin/spe to libexec/bin/spe. Rationale: the in-binary
    # ort_resolver::init_ort_env() canonicalises current_exe() and walks one
    # dir up from bin/ to find lib/. With libexec/{bin,lib} the resolver
    # sees <libexec_dir>/lib/libonnxruntime.<ver> and dlopens it correctly.
    libexec.install Dir["*"]
    bin.install_symlink libexec/"bin/spe"
  end

  test do
    # Validates the resolver: `spe --version` exercises clap parsing AND the
    # in-binary ort_resolver path. If the resolver didn't find lib/, the
    # subsequent device subcommand would fail at ORT init — keeping the test
    # tight to --version so it's fast and dep-free.
    #
    # Shape-check (paired with sparrow-engine-gpu.rb for symmetry):
    # `spe --version` is emitted by clap from the const
    #   sparrow-engine-cli/src/main.rs:43
    #   `const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (CPU flavor)");`
    # so the output shape is `spe X.Y.Z (CPU flavor)`. The regex catches both
    # (a) Phase E B-03-style drift where the binary stops reporting cargo-pkg-
    # version dynamically and (b) any rename/reshape of the version string.
    # We deliberately do NOT pin to `version.to_s` because the formula header
    # `version` is bumped only on release-cut and lags the cargo crate version
    # between bumps; pinning would break brew CI on every cargo-side bump.
    assert_match(/\Aspe \d+\.\d+\.\d+ \(CPU flavor\)\Z/, shell_output("#{bin}/spe --version").strip)
  end
end
