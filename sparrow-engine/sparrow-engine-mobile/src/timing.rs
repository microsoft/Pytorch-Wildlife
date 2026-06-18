//! Env-gated per-stage timing for the E2 batching micro-benchmark (Pi-field
//! two-stage cascade optimization, `docs/design/pi-field-twostage-optimization`).
//!
//! Activated by the `SPE_TIMING` environment variable (`1`/`true`/`yes`, etc.).
//! Off by default and effectively free when off (one cached bool check per span).
//!
//! # Why a thread-local accumulator
//! The decision-grade E2 question is "what fraction of each model invoke is
//! *fixed* per-invoke setup (buffer alloc / lock / copy) versus arithmetic the
//! batch lever cannot remove?". That split lives deep inside
//! [`crate::tflite::LiteRtBackend::invoke`]. Rather than thread timing values
//! back up through every signature (which would perturb the 18-symbol FFI ABI
//! and the `invoke_single` contract), the backend records its setup/run/readout
//! spans into a thread-local, and [`crate::pipeline::run_pipeline`] reads them
//! per window. The mobile engine is thread-affine (LiteRT is `&mut`/`Rc`), so
//! every invoke runs on the owner thread and the thread-local is always correct.
//!
//! Records go to **stderr** (`SPE_TIMING ...` per window + one `SPE_TIMING_AGG`
//! line) so they never contaminate the CLI's stdout result table.

use std::cell::Cell;
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Parse the `SPE_TIMING` value into an on/off flag. Empty, `0`, `false`, `no`,
/// `off` (case-insensitive) are off; any other non-empty value is on.
fn truthy(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(v) => {
            let v = v.trim();
            !(v.is_empty()
                || v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off"))
        }
    }
}

/// Whether per-stage timing is enabled (cached from `SPE_TIMING` on first call).
pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| truthy(std::env::var("SPE_TIMING").ok().as_deref()))
}

thread_local! {
    /// (setup_ns, run_ns, readout_ns) accumulated for the current invoke.
    static SPANS: Cell<(u128, u128, u128)> = const { Cell::new((0, 0, 0)) };
}

/// Clear the accumulator before an invoke that should be measured in isolation.
pub(crate) fn reset_invoke() {
    SPANS.with(|c| c.set((0, 0, 0)));
}

/// Add a measured input/output buffer-setup span (nanoseconds).
pub(crate) fn add_setup(ns: u128) {
    SPANS.with(|c| {
        let (s, r, rd) = c.get();
        c.set((s + ns, r, rd));
    });
}

/// Add a measured `LiteRtRunCompiledModel` span (nanoseconds).
pub(crate) fn add_run(ns: u128) {
    SPANS.with(|c| {
        let (s, r, rd) = c.get();
        c.set((s, r + ns, rd));
    });
}

/// Add a measured output-readout span (nanoseconds).
pub(crate) fn add_read(ns: u128) {
    SPANS.with(|c| {
        let (s, r, rd) = c.get();
        c.set((s, r, rd + ns));
    });
}

/// Take the (setup_ns, run_ns, readout_ns) spans accumulated since the last
/// [`reset_invoke`], and clear the accumulator.
pub(crate) fn take_invoke() -> (u128, u128, u128) {
    SPANS.with(|c| c.replace((0, 0, 0)))
}

/// Nanoseconds → milliseconds as f64 (for the stderr records).
pub(crate) fn ns_ms(ns: u128) -> f64 {
    ns as f64 / 1.0e6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_parsing() {
        for on in ["1", "true", "TRUE", "yes", "Y", "on", "2", "anything"] {
            assert!(truthy(Some(on)), "{on:?} should be ON");
        }
        for off in [None, Some(""), Some("  "), Some("0"), Some("false"), Some("No"), Some("OFF")] {
            assert!(!truthy(off), "{off:?} should be OFF");
        }
    }

    #[test]
    fn spans_accumulate_and_take() {
        reset_invoke();
        add_setup(100);
        add_setup(50);
        add_run(400);
        add_read(20);
        assert_eq!(take_invoke(), (150, 400, 20));
        // take clears
        assert_eq!(take_invoke(), (0, 0, 0));
    }

    #[test]
    fn ns_ms_converts() {
        assert!((ns_ms(1_000_000) - 1.0).abs() < 1e-9);
    }
}
