//! Phase 3.8 Phase C Wave 4a — per-flavor engine dispatch shim.
//!
//! `sparrow-engine-python` builds either a CPU wheel (`sparrow-engine`) or a GPU
//! wheel (`sparrow-engine-gpu`) via mutually-exclusive Cargo features. Both
//! `sparrow-engine-cpu` and `sparrow-engine-gpu` set
//! `[lib] name = "sparrow_engine"` so the cdylib filename matches
//! `libsparrow_engine.so` on either flavor and the Python
//! `_sparrow_engine_core.cpython-*.so` extension links against whichever one is
//! feature-active.
//!
//! Cargo's feature mutex (enforced by `compile_error!` in `lib.rs`)
//! picks the dependency at build time — only one `sparrow_engine` lib is
//! in the dep graph per invocation, so the
//! `pub use ::sparrow_engine::*;` glob below resolves unambiguously to
//! the active engine crate's full public surface. `lib.rs` aliases this
//! module as `sparrow_engine` so consumer code (`sparrow_engine::Engine`,
//! `sparrow_engine::detect::detect`, `sparrow_engine::viz::*`,
//! `sparrow_engine::SparrowEngineError`, etc.) still routes through this single
//! dispatch point rather than through the cargo extern-crate prelude.
//! This mirrors the `sparrow-engine-server` Wave 2 and
//! `sparrow-engine-cli` Wave 3 shims (post-c6b0e86 cfg-free pattern) —
//! see `docs/design/phase3.8/phase_c/implementation_plan.md` §2.1.
//!
//! Phase 3.8 Phase C audit-fix R1 (I-4 / R-S1 / A1, 2026-05-06): widened
//! from a 9-name enumeration to the full glob, and the alias in `lib.rs`
//! makes the shim load-bearing rather than decorative — single-point
//! dispatch boundary per implementation_plan.md §2.1.

pub use sparrow_engine::*;
