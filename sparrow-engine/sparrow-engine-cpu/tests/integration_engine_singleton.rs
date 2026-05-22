//
// The unit test `engine::tests::singleton_enforcement` already covers this in
// `src/engine.rs`. This file adds an INTEGRATION-LEVEL singleton test through
// the public `sparrow_engine::Engine` re-export, so the assertion fires against the
// crate's external surface (the path consumers actually use).
//
// Run with:
//   ORT_LIB_LOCATION=target/ort-lib ORT_PREFER_DYNAMIC_LINK=1 \
//     LD_LIBRARY_PATH=target/ort-lib \
//     cargo test -p sparrow-engine-cpu --test integration_engine_singleton -- --ignored --test-threads=1
//
// We need `--test-threads=1` because Engine is a process-global singleton and
// other ORT-using integration tests would race.

use std::path::PathBuf;

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{SparrowEngineError, Engine};

fn nonexistent_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("__bongo_nonexistent_model_dir__")
}

#[test]
#[ignore] // Requires ORT runtime
fn engine_singleton_at_integration_scope() {
    // Construct two configs against a bogus model_dir — Engine::new only
    // initializes the ORT environment + thread pool; it does NOT touch the
    // model_dir until load_model is called. So this test is ORT-only, no
    // model files needed.
    let cfg1 = EngineConfig::new(Device::Cpu, nonexistent_model_dir());
    let engine1 = Engine::new(cfg1).expect("first Engine::new should succeed");

    let cfg2 = EngineConfig::new(Device::Cpu, nonexistent_model_dir());
    let result2 = Engine::new(cfg2);
    // `Engine` does not derive Debug (the inner ORT session is not Debug), so
    // we destructure manually for the assertion message.
    match &result2 {
        Err(SparrowEngineError::EngineAlreadyExists) => {}
        Err(e) => panic!(
            "Second Engine::new must return EngineAlreadyExists at integration \
             scope, got Err({})",
            e
        ),
        Ok(_) => panic!(
            "Second Engine::new must return EngineAlreadyExists at integration \
             scope, got Ok(_)"
        ),
    }
    drop(result2);

    // Drop and re-create — singleton flag clears on Drop.
    drop(engine1);
    let cfg3 = EngineConfig::new(Device::Cpu, nonexistent_model_dir());
    let engine3 = Engine::new(cfg3)
        .expect("third Engine::new after drop should succeed (Drop clears ENGINE_EXISTS)");
    drop(engine3);
}
