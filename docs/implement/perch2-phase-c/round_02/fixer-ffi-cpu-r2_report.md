<a id="ffi-v2-audio-cpu"></a>
# ffi-v2-audio-cpu — fixer-ffi-cpu-r2 report

- Fix applied: changed .args(&["-D", "--defined-only"]) to .args(["-D", "--defined-only"]) in sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs.
- Commit: ce8edf1a52c9b2b168daffa1fbf58f0c9eb42641
- SIGNATURE: cdylib_exports_match_exports_def
- Clippy: CLEAN (cargo clippy -q -p sparrow-engine-cpu --features ffi --all-targets -- -D warnings)
- Test: 2/2 PASS (cargo test -q -p sparrow-engine-cpu --features ffi --test integration_ffi_symbols)
- Key design decisions: none (narrow fix).
- Cross-boundary issues: none.

Before:
~~~rust
.args(&["-D", "--defined-only"])
~~~

After:
~~~rust
.args(["-D", "--defined-only"])
~~~

STATUS: DONE COMMIT=ce8edf1a52c9b2b168daffa1fbf58f0c9eb42641 SIGNATURE="cdylib_exports_match_exports_def" ITEM=ffi-v2-audio-cpu
