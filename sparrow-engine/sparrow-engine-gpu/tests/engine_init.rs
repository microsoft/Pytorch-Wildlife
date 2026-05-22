//! Engine smoke test: construct + drop without panic, verify singleton
//! discipline, verify resolved device.
//!
//! No model is loaded — Wave 1 stubs all model-loading paths. Wave 2
//! adds end-to-end MDv6 + DeepFaune tests in the same file.

use sparrow_engine::Engine;
use sparrow_engine_types::{SparrowEngineError, Device, EngineConfig};

fn gpu_tests_enabled() -> bool {
    !matches!(std::env::var("SPARROW_ENGINE_GPU_TESTS").as_deref(), Ok("0"))
}

fn cfg() -> EngineConfig {
    EngineConfig::new(Device::Auto, "/tmp")
}

// All three tests share the same singleton slot, so they must NOT run
// in parallel (Cargo's default). serial_test isn't a workspace dep here
// (we don't pull it from sparrow-engine-cpu's dev-deps), so we serialize manually
// via a single test body that exercises the lifecycle steps in order.
//
// This keeps the public assertions visible at the test-name level
// (engine_init::all_lifecycle_steps) while avoiding parallel-execution
// false-fails on EngineAlreadyExists. If serial_test gets pulled into
// sparrow-engine-gpu's dev-deps in Wave 2, this can split back into 3 #[test]
// fns with #[serial] decorators.
#[test]
fn all_lifecycle_steps() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping engine lifecycle test");
        return;
    }

    // Step 1: construct + resolve device.
    let e = Engine::new(cfg()).expect("Engine::new (1st)");
    match e.active_device() {
        Device::Cuda(_) => {}
        other => panic!("expected Cuda device, got {other:?}"),
    }

    // Step 2: loaded_models is empty in Wave 1.
    assert!(e.loaded_models().is_empty(), "Wave 1 loaded_models must be empty");

    // Step 3: singleton — second Engine::new fails while `e` is alive.
    // Engine deliberately does NOT derive Debug (holds CUDA context handles
    // whose Debug impls are flaky across cudarc versions), so we can't use
    // `expect_err`; match the Result directly instead.
    match Engine::new(cfg()) {
        Ok(_) => panic!("second Engine::new must fail"),
        Err(SparrowEngineError::EngineAlreadyExists) => {}
        Err(other) => panic!("expected EngineAlreadyExists, got {other:?}"),
    }

    // Step 4: Drop releases the slot — a fresh construct succeeds.
    drop(e);
    let _e2 = Engine::new(cfg()).expect("post-drop Engine::new");
}
