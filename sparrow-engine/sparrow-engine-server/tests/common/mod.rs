#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sparrow_engine_server::config::{Config, LogFormat};
use sparrow_engine_server::engine_dispatch::{Device, Engine, EngineConfig};
use sparrow_engine_server::router;
use sparrow_engine_server::state::AppState;
use reqwest::Client;
use serde::Deserialize;
use tokio::net::TcpListener;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

pub fn test_files_dir() -> PathBuf {
    PathBuf::from("/home/miao/repos/PW_refactor/test_files")
}

pub fn onnx_dir() -> PathBuf {
    test_files_dir().join("onnx")
}

pub fn test_cameratrap_dir() -> PathBuf {
    test_files_dir().join("test_cameratrap")
}

pub fn test_overhead_dir() -> PathBuf {
    test_files_dir().join("test_overhead")
}

pub fn test_audio_dir() -> PathBuf {
    test_files_dir().join("test_audio")
}

/// A known-good camera trap image for smoke tests.
pub fn test_image_path() -> PathBuf {
    let dir = test_cameratrap_dir();
    first_file_with_ext(&dir, &["jpg", "jpeg", "png"])
        .unwrap_or_else(|| panic!("No test image found in {}", dir.display()))
}

/// A known-good WAV file for audio smoke tests.
pub fn test_audio_path() -> PathBuf {
    let dir = test_audio_dir();
    first_file_with_ext(&dir, &["wav"])
        .unwrap_or_else(|| panic!("No WAV file found in {}", dir.display()))
}

fn first_file_with_ext(dir: &Path, exts: &[&str]) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| {
                let e = ext.to_ascii_lowercase();
                exts.iter().any(|&x| e == x)
            })
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

// ---------------------------------------------------------------------------
// TestServer
// ---------------------------------------------------------------------------

/// In-process sparrow-engine-server for integration tests.
///
/// Spawns the real axum router on a random port. Shared across tests in a
/// single test binary via `server()`.
pub struct TestServer {
    pub base_url: String,
    pub client: Client,
    _handle: tokio::task::JoinHandle<()>,
}

static TEST_SERVER: tokio::sync::OnceCell<TestServer> = tokio::sync::OnceCell::const_new();

/// Get (or lazily start) the shared test server.
pub async fn server() -> &'static TestServer {
    TEST_SERVER.get_or_init(TestServer::start).await
}

impl TestServer {
    pub async fn start() -> Self {
        let engine_config = EngineConfig::new(Device::Cpu, onnx_dir());
        let engine = Engine::new(engine_config).expect("failed to create Engine");

        // Auto-load all manifests from test onnx dir.
        // Test files are flat (e.g., mdv6_manifest.toml), not in subdirectories.
        for entry in std::fs::read_dir(onnx_dir()).unwrap().flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap().to_str().unwrap_or("");
            if name.ends_with("_manifest.toml") {
                if let Err(e) = engine.load_model(&path) {
                    eprintln!("WARN: skip model {}: {e}", path.display());
                }
            } else if name.ends_with("_pipeline.toml") || name == "pipeline.toml" {
                if let Err(e) = engine.load_pipeline(&path) {
                    eprintln!("WARN: skip pipeline {}: {e}", path.display());
                }
            }
        }

        let config = Config {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            model_dir: onnx_dir(),
            log_format: LogFormat::Pretty,
            log_level: "warn".to_string(),
            max_body_size: 250 * 1024 * 1024,
            max_concurrent_inference: 4,
            max_batch_size: 16,
            request_timeout_secs: 600,
            drain_timeout_secs: 5,
            device: "cpu".to_string(),
            inter_threads: None,
            intra_threads: None,
            idle_unload_seconds: 0,
            idle_unload_keep_last_n: 1,
        };

        let state = AppState::new(engine, config);
        let app = router::build_router(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            client: Client::new(),
            _handle: handle,
        }
    }

    // -----------------------------------------------------------------------
    // Request helpers
    // -----------------------------------------------------------------------

    pub async fn health(&self) -> reqwest::Response {
        self.client
            .get(format!("{}/v1/health", self.base_url))
            .send()
            .await
            .unwrap()
    }

    pub async fn detect(&self, model: &str, image_path: &Path) -> reqwest::Response {
        let form = image_form("image", image_path);
        self.client
            .post(format!("{}/v1/detect?model={model}", self.base_url))
            .multipart(form)
            .send()
            .await
            .unwrap()
    }

    pub async fn classify(&self, model: &str, image_path: &Path) -> reqwest::Response {
        let form = image_form("image", image_path);
        self.client
            .post(format!("{}/v1/classify?model={model}", self.base_url))
            .multipart(form)
            .send()
            .await
            .unwrap()
    }

    pub async fn audio_detect(&self, model: &str, audio_path: &Path) -> reqwest::Response {
        let form = image_form("audio", audio_path);
        self.client
            .post(format!("{}/v1/audio/detect?model={model}", self.base_url))
            .multipart(form)
            .send()
            .await
            .unwrap()
    }

    pub async fn detect_batch(
        &self,
        model: &str,
        image_paths: &[PathBuf],
    ) -> reqwest::Response {
        let mut form = reqwest::multipart::Form::new();
        for path in image_paths {
            let bytes = std::fs::read(path).unwrap();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            let part = reqwest::multipart::Part::bytes(bytes).file_name(name);
            form = form.part("images", part);
        }
        self.client
            .post(format!("{}/v1/detect/batch?model={model}", self.base_url))
            .multipart(form)
            .send()
            .await
            .unwrap()
    }

    pub async fn list_models(&self) -> reqwest::Response {
        self.client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await
            .unwrap()
    }

    pub async fn list_pipelines(&self) -> reqwest::Response {
        self.client
            .get(format!("{}/v1/pipelines", self.base_url))
            .send()
            .await
            .unwrap()
    }
}

/// Build a multipart form with a single file field.
fn image_form(field_name: &str, path: &Path) -> reqwest::multipart::Form {
    let bytes = std::fs::read(path).unwrap();
    let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
    let part = reqwest::multipart::Part::bytes(bytes).file_name(file_name);
    reqwest::multipart::Form::new().part(field_name.to_string(), part)
}

// ---------------------------------------------------------------------------
// Response deserialization types (mirror server response.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BBoxResponse {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

#[derive(Debug, Deserialize)]
pub struct DetectionResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    pub bbox: BBoxResponse,
}

#[derive(Debug, Deserialize)]
pub struct DetectResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub detections: Vec<DetectionResponse>,
}

#[derive(Debug, Deserialize)]
pub struct ClassificationResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

#[derive(Debug, Deserialize)]
pub struct ClassifyResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub classifications: Vec<ClassificationResponse>,
}

#[derive(Debug, Deserialize)]
pub struct AudioSegmentResponse {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
}

#[derive(Debug, Deserialize)]
pub struct AudioDetectResponse {
    pub model_id: String,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
    pub segments: Vec<AudioSegmentResponse>,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub models_loaded: usize,
    pub pipelines_loaded: usize,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub status: u16,
}

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}
