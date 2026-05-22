//! Compile-time engine dispatch shim for `sparrow-engine-server` (Phase 3.8 Phase C Wave 2).
//!
//! `sparrow-engine-server` consumes exactly one engine crate, selected at
//! compile time via the mutually exclusive `cpu` / `gpu` Cargo features.
//! This module re-exports the chosen engine crate's public surface
//! (`Engine`, `ModelHandle`, top-level inference modules, error type,
//! manifest types, result/option types) so the rest of
//! `sparrow-engine-server` is engine-agnostic and writes
//! `use crate::engine_dispatch::*` instead of depending on the
//! `sparrow_engine` crate link directly.
//!
//! Both `sparrow-engine-cpu` and `sparrow-engine-gpu` glob-re-export
//! `sparrow_engine_core::*` and `sparrow_engine_types::*` from their
//! respective `lib.rs` (per Phase A S2 closure and Phase C Wave 1
//! audit-fix R2 I-S1), so the surfaces match item-for-item at the
//! cargo-build dependency boundary.
//!
//! See `docs/design/phase3.8/phase_c/implementation_plan.md` §2.1.

// Mutual-exclusivity enforcement. Exactly one of `cpu` / `gpu` MUST be set.
#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!(
    "sparrow-engine-server: features `cpu` and `gpu` are mutually exclusive; \
     pick one (default = cpu)"
);
#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!(
    "sparrow-engine-server: one of `cpu` or `gpu` must be enabled (default = cpu); \
     do not pass --no-default-features without explicitly enabling a flavor"
);

// Both `sparrow-engine-cpu` and `sparrow-engine-gpu` now set
// `[lib] name = "sparrow_engine"` (cdylib `libsparrow_engine.so`
// invariant for both flavors). The mutex above ensures only one is in
// scope at a time, so `pub use sparrow_engine::*` resolves to the active
// engine crate.
pub use sparrow_engine::*;
