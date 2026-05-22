//! Integration tests for management endpoints: health, models, pipelines.
//!
//! All tests are `#[ignore]` — they require a running sparrow-engine-server with model files + ORT.
//! Run with: `cargo test --test test_management -- --ignored`

mod common;

use reqwest::StatusCode;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_health() {
    let server = common::TestServer::start().await;

    let resp = server
        .client
        .get(format!("{}/v1/health", server.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ready");
    assert!(body["models_loaded"].as_u64().unwrap() > 0);
    assert!(body["version"].as_str().is_some());
}

#[tokio::test]
#[ignore]
async fn test_liveness() {
    let server = common::TestServer::start().await;

    let resp = server
        .client
        .get(format!("{}/healthz", server.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["alive"], true);
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/// Get the first loaded model ID from the server.
async fn first_model_id(server: &common::TestServer) -> String {
    let resp = server
        .client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    body["models"][0]["id"]
        .as_str()
        .expect("no models loaded — cannot run test")
        .to_string()
}

#[tokio::test]
#[ignore]
async fn test_list_models() {
    let server = common::TestServer::start().await;

    let resp = server
        .client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    let models = body["models"].as_array().unwrap();
    assert!(!models.is_empty(), "expected at least one loaded model");

    for m in models {
        assert!(m["id"].as_str().is_some());
        assert!(m["model_type"].as_str().is_some());
        // SRV1: `default` is a required field surfaced from ModelInfo.
        assert!(m["default"].as_bool().is_some());
    }

    // SRV1: exactly one model must be flagged default:true per type family
    // (enforced upstream by manifest validation). Check at least one default
    // exists if any detector/classifier is loaded.
    let any_default = models.iter().any(|m| m["default"].as_bool() == Some(true));
    assert!(
        any_default,
        "expected at least one default model to be flagged default=true"
    );
}

#[tokio::test]
#[ignore]
async fn test_load_model_idempotent() {
    let server = common::TestServer::start().await;
    let model_id = first_model_id(&server).await;

    // load_model_by_id resolves {model_dir}/{id}/manifest.toml.
    // Test model dir uses flat files (*_manifest.toml), so reload will 404.
    // Skip if the expected directory doesn't exist.
    let model_subdir = common::onnx_dir().join(&model_id);
    if !model_subdir.join("manifest.toml").exists() {
        eprintln!(
            "SKIP: test_load_model_idempotent — flat test dir, no {}/manifest.toml",
            model_id
        );
        return;
    }

    // Load an already-loaded model — should return 200, not 409.
    let resp = server
        .client
        .post(format!("{}/v1/models/load", server.base_url))
        .json(&serde_json::json!({ "model_id": model_id }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"].as_str().unwrap(), model_id);
}

#[tokio::test]
#[ignore]
async fn test_unload_model() {
    let server = common::TestServer::start().await;
    let model_id = first_model_id(&server).await;

    // load_model_by_id resolves {model_dir}/{id}/manifest.toml for the reload
    // cleanup. Skip flat fixture layouts before unloading so this test cannot
    // leave the shared server missing its default model.
    let model_subdir = common::onnx_dir().join(&model_id);
    if !model_subdir.join("manifest.toml").exists() {
        eprintln!(
            "SKIP: test_unload_model — flat test dir, no {}/manifest.toml",
            model_id
        );
        return;
    }

    // Unload
    let resp = server
        .client
        .delete(format!("{}/v1/models/{}", server.base_url, model_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Verify it's gone
    let resp = server
        .client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        !ids.contains(&model_id.as_str()),
        "model should no longer appear after unload"
    );

    // Reload so we don't break other tests sharing this server.
    let reload_resp = server
        .client
        .post(format!("{}/v1/models/load", server.base_url))
        .json(&serde_json::json!({ "model_id": model_id }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        reload_resp.status(),
        StatusCode::OK,
        "model reload after unload must succeed before test exits"
    );

    let resp = server
        .client
        .get(format!("{}/v1/models", server.base_url))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&model_id.as_str()),
        "model should reappear after reload"
    );
}

#[tokio::test]
#[ignore]
async fn test_unload_nonexistent() {
    let server = common::TestServer::start().await;

    let resp = server
        .client
        .delete(format!("{}/v1/models/nosuchmodel", server.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
#[ignore]
async fn test_load_reload_detect() {
    let server = common::TestServer::start().await;
    let model_id = first_model_id(&server).await;

    // load_model_by_id resolves {model_dir}/{id}/manifest.toml.
    // Skip if test dir uses flat layout.
    let model_subdir = common::onnx_dir().join(&model_id);
    if !model_subdir.join("manifest.toml").exists() {
        eprintln!(
            "SKIP: test_load_reload_detect — flat test dir, no {}/manifest.toml",
            model_id
        );
        return;
    }

    // Step 1: Unload
    let resp = server
        .client
        .delete(format!("{}/v1/models/{}", server.base_url, model_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Step 2: Reload
    let resp = server
        .client
        .post(format!("{}/v1/models/load", server.base_url))
        .json(&serde_json::json!({ "model_id": model_id }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Step 3: Detect with the reloaded model (minimal 1x1 RGB PNG).
    #[rustfmt::skip]
    let png_1x1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(png_1x1.to_vec())
            .file_name("test.png")
            .mime_str("image/png")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!("{}/v1/detect?model={}", server.base_url, model_id))
        .multipart(form)
        .send()
        .await
        .unwrap();

    // Accept 200 (inference ok) or 422 (image too small for model).
    // The point: model is functional after reload — not 404/500.
    assert!(
        resp.status() == StatusCode::OK || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "unexpected status {} after reload + detect",
        resp.status()
    );
}

// ---------------------------------------------------------------------------
// Pipeline management
// ---------------------------------------------------------------------------

/// Get the first loaded pipeline ID, or None if none exist.
async fn first_pipeline_id(server: &common::TestServer) -> Option<String> {
    let resp = server
        .client
        .get(format!("{}/v1/pipelines", server.base_url))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    body["pipelines"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|p| p["id"].as_str())
        .map(|s| s.to_string())
}

#[tokio::test]
#[ignore]
async fn test_list_pipelines() {
    let server = common::TestServer::start().await;

    let resp = server
        .client
        .get(format!("{}/v1/pipelines", server.base_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert!(body["pipelines"].as_array().is_some());
}

#[tokio::test]
#[ignore]
async fn test_load_unload_pipeline() {
    let server = common::TestServer::start().await;

    let pipeline_id = match first_pipeline_id(&server).await {
        Some(id) => id,
        None => {
            eprintln!("no pipelines loaded — skipping test_load_unload_pipeline");
            return;
        }
    };

    // Unload
    let resp = server
        .client
        .delete(format!("{}/v1/pipelines/{}", server.base_url, pipeline_id))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // Verify gone
    let resp = server
        .client
        .get(format!("{}/v1/pipelines", server.base_url))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["pipelines"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|p| p["id"].as_str())
        .collect();
    assert!(!ids.contains(&pipeline_id.as_str()));

    // Reload
    let resp = server
        .client
        .post(format!("{}/v1/pipelines/load", server.base_url))
        .json(&serde_json::json!({ "pipeline_id": pipeline_id }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"].as_str().unwrap(), pipeline_id);
}
