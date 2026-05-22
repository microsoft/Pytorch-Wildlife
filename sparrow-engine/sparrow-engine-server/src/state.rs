use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Semaphore;

use crate::config::Config;
use crate::discover::{discover_catalog, Catalog};
use crate::engine_dispatch::Engine;
use crate::sink::{InferenceLogSink, StderrJsonLinesSink};

/// Shared application state, cheaply cloneable via inner `Arc`s.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub config: Arc<Config>,
    pub inference_semaphore: Arc<Semaphore>,
    pub catalog: Arc<Catalog>,
    pub pipeline_write_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    /// Phase 4 W3 — pluggable inference-log sink. Default is the
    /// stderr JSON-lines sink. Tests / future sparrow-data integration
    /// substitute via [`AppState::with_catalog_and_sink`].
    pub log_sink: Arc<dyn InferenceLogSink>,
}

impl AppState {
    /// Default constructor — wires the stderr JSON-lines sink and triggers
    /// a one-shot `discover_catalog(&config.model_dir)` scan to populate the
    /// `Catalog`. Use [`AppState::with_catalog`] /
    /// [`AppState::with_catalog_and_sink`] when the caller already has a
    /// `Catalog` (e.g. `main.rs` reuses the discovered catalog to also
    /// register pipeline aliases on the engine) — those constructors are
    /// pure and incur no filesystem I/O.
    pub fn new(engine: Engine, config: Config) -> Self {
        let catalog = discover_catalog(&config.model_dir);
        Self::with_catalog_and_sink(engine, config, catalog, Arc::new(StderrJsonLinesSink))
    }

    pub fn with_catalog(engine: Engine, config: Config, catalog: Catalog) -> Self {
        Self::with_catalog_and_sink(engine, config, catalog, Arc::new(StderrJsonLinesSink))
    }

    pub fn with_catalog_and_sink(
        engine: Engine,
        config: Config,
        catalog: Catalog,
        log_sink: Arc<dyn InferenceLogSink>,
    ) -> Self {
        let max_concurrent = config.max_concurrent_inference;
        Self {
            engine: Arc::new(engine),
            config: Arc::new(config),
            inference_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            catalog: Arc::new(catalog),
            pipeline_write_locks: Arc::new(Mutex::new(HashMap::new())),
            log_sink,
        }
    }
}
