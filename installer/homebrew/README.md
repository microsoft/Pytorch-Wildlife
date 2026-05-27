# Homebrew tap bootstrap — sparrow-engine

This directory holds the source-of-truth Homebrew formula for the
`sparrow-engine` CLI (RP-17). The formula lives here, not in a separate
tap repo, so it versions with the rest of the codebase.

## What ships

- `sparrow-engine.rb` — formula pointing at the GH Release tarballs produced
  by RP-4 (`.github/workflows/release.yml § build-cli-*`).

## End-user UX (post-tap-publish)

```bash
brew tap microsoft/sparrow-engine
brew install sparrow-engine
spe --version
spe detect --image /path/to/photo.jpg
```

`spe` is a symlink under brew's `bin/` pointing at
`<Cellar>/sparrow-engine/<ver>/libexec/bin/spe`. The in-binary
`ort_resolver::init_ort_env()` (RP-4 step 1) canonicalises
`current_exe()`, walks one directory up, and dlopens
`<Cellar>/.../libexec/lib/libonnxruntime.<dylib|so.X.Y.Z>` — no
`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` setup required.

## Bootstrapping the tap repo (one-time, operator action)

The formula in this directory is the source of truth; the public tap
repo at `microsoft/homebrew-sparrow-engine` is a thin distribution
surface. Procedure:

1. Wait for the first RP-4 release to fire — `git tag v0.1.6` + `git push
   origin v0.1.6` cuts the GH Release and attaches the tarballs.
2. Fetch the `.sha256` files from the GH Release:
   ```bash
   gh release download v0.1.6 --pattern '*.sha256' --dir /tmp/sha
   ```
3. Replace the `REPLACE_WITH_*_sha256` placeholders in
   `sparrow-engine.rb` with the matching checksums.
4. Create the public tap repo `microsoft/homebrew-sparrow-engine` (one-
   time GitHub UI / `gh repo create`). Repos named `homebrew-*` are
   recognised as brew taps automatically.
5. Copy `sparrow-engine.rb` into `Formula/sparrow-engine.rb` in the new
   tap repo. Commit + push.
6. Smoke test on a macOS arm64 + a brew-Linux x86_64 host:
   ```bash
   brew tap microsoft/sparrow-engine
   brew install sparrow-engine
   spe device   # expected: {"device":"cpu"}
   ```

## Per-release bump (after bootstrap)

Each subsequent release follows the same shape: fetch new `.sha256`
files, substitute, commit to the tap repo, push. Automatable via a
small helper script (deferred — see `docs/ideas.md § RP-17`).

## Why not just submit to homebrew-core?

`homebrew-core` has acceptance criteria (notable, maintained, widely
used). Pre-public-release, sparrow-engine doesn't meet them yet. The
custom tap is the bridge until the project is established enough to
warrant a core submission. Migration from custom-tap → core later is
straightforward (formula source code is the same).
