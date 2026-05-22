# pytorchwildlife (compatibility shim)

This is a compatibility shim for users migrating from
[PytorchWildlife](https://github.com/microsoft/CameraTraps) to
[sparrow-engine](https://github.com/microsoft/CameraTraps) (the Rust-core
replacement).

Installing this wheel gives you a `pytorchwildlife` module that re-exports
sparrow-engine's public API and emits a `DeprecationWarning` on import. The
`pytorchwildlife` alias is scheduled for removal in **0.2.0**.

## Install

The wheel is distributed as a GitHub Releases artifact. There is no public
PyPI package for `pytorchwildlife` (see "PyPI availability" below).

```bash
pip install https://github.com/microsoft/CameraTraps/releases/download/<tag>/pytorchwildlife-0.1.0-py3-none-any.whl
```

`sparrow-engine` (the dependency) must be installable in the same environment. It
is distributed via the same GitHub Releases channel for the 3.x + 4.x
cycle.

## Usage

```python
import pytorchwildlife  # DeprecationWarning emitted
from pytorchwildlife import detect, classify, pipeline  # routed to sparrow_engine.*
```

Because the module is a wildcard re-export of `sparrow_engine`, everything listed
in `sparrow_engine.__all__` is available under `pytorchwildlife.` — inference
functions (`detect`, `classify`, `detect_audio`, `pipeline`), standalone
helpers (`hash_file`, `day_night`, `verify_model`, `summarize`,
`visualize`, `export`), and result types (`DetectResult`,
`ClassifyResult`, `PipelineResult`, `AudioResult`, `BBox`, `Detection`,
`ModelInfo`, `SparrowEngineError`, etc.).

For the authoritative API surface and usage examples, see the sparrow-engine
documentation.

## Migration

1. Install `sparrow-engine` directly and update imports:
   ```python
   # Before
   import pytorchwildlife
   from pytorchwildlife import detect

   # After
   import sparrow_engine
   from sparrow_engine import detect
   ```
2. Silence the warning during transition if needed:
   ```python
   import warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning, module="pytorchwildlife")
   ```
3. The `pytorchwildlife` package will be removed in `0.2.0`. Migrate
   before upgrading to that release.

## Version co-bump rule (maintainers)

The shim's runtime dependency on `sparrow-engine==0.1.0` (declared in
`pytorchwildlife-compat/pyproject.toml`) MUST be bumped in lockstep when
sparrow-engine's version moves. The clean-room smoke test in
`sparrow-engine/scripts/clean_room_test.sh` installs the shim with `pip install
--no-deps`, which papers over the version pin to keep the test green
during dev. Real installs (without `--no-deps`) will refuse to install if
sparrow-engine's actual version doesn't match the pin, and `import pytorchwildlife`
may fail at any time the shim's wildcard re-export touches a sparrow-engine public
API that drifted past `0.1.0`. Treat this as part of every sparrow-engine
version-bump checklist: bump sparrow-engine → bump the shim's `sparrow-engine==X.Y.Z` pin in
the same commit / PR.

## PyPI availability

This package is not on public PyPI. Public PyPI release pipeline (for
both `sparrow-engine` and `pytorchwildlife`) is planned for Phase 4.5 or later,
after Phase 4 (Docker data management) ships and the platform stabilizes.
See `docs/design/phase3.5/final_design.md` §7.1 for the rationale and the
three tracked sub-items (name availability for `sparrow-engine`, name availability
and upstream-maintainer coordination for `pytorchwildlife`, GPU-wheel
packaging story).
