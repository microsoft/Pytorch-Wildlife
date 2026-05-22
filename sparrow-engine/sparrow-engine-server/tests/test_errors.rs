//! Integration tests for error handling and edge cases.
//!
//! All tests are `#[ignore]` — they require ORT runtime and model files.
//!
//! Run with:
//! ```sh
//! ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 LD_LIBRARY_PATH=/tmp/ort-lib \
//!   cargo test -p sparrow-engine-server --test test_errors -- --ignored --test-threads=1
//! ```

mod common;

use reqwest::multipart;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the first loaded detector model ID from the server.
async fn first_detector_model_id(server: &common::TestServer) -> String {
    let resp = server.list_models().await;
    let body: Value = resp.json().await.unwrap();
    body["models"]
        .as_array()
        .and_then(|models| {
            models.iter().find_map(|model| {
                (model["model_type"].as_str() == Some("detector"))
                    .then(|| model["id"].as_str())
                    .flatten()
            })
        })
        .expect("no detector models loaded — cannot run detect-route test")
        .to_string()
}

/// Assert the canonical error response shape:
/// `{ "error": { "code": "<string>", "message": "<string>", "status": <u16> } }`
fn assert_error_shape(body: &Value, expected_status: u16) {
    let error = body.get("error").expect("response missing 'error' key");
    let err_obj = error.as_object().expect("'error' must be a JSON object");
    assert!(err_obj.contains_key("code"), "error must have 'code'");
    assert!(err_obj.contains_key("message"), "error must have 'message'");
    assert!(err_obj.contains_key("status"), "error must have 'status'");
    assert!(err_obj["code"].is_string(), "error.code must be a string");
    assert!(
        err_obj["message"].is_string(),
        "error.message must be a string"
    );
    assert!(
        err_obj["status"].is_number(),
        "error.status must be a number"
    );
    assert_eq!(
        err_obj["status"].as_u64().unwrap(),
        expected_status as u64,
        "error.status mismatch"
    );
}

// ---------------------------------------------------------------------------
// Bad input
// ---------------------------------------------------------------------------

/// POST /v1/detect with garbage bytes as "image" — the server should return
/// 422 with IMAGE_DECODE_ERROR (image decode fails after model lookup succeeds).
#[tokio::test]
#[ignore]
async fn test_bad_image_data() {
    let server = common::TestServer::start().await;
    let model_id = first_detector_model_id(&server).await;

    let form = multipart::Form::new().part(
        "image",
        multipart::Part::bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]).file_name("garbage.bin"),
    );

    let resp = server
        .client
        .post(format!("{}/v1/detect?model={model_id}", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 422, "garbage image should produce 422");
    let body: Value = resp.json().await.unwrap();
    assert_error_shape(&body, 422);
    assert_eq!(body["error"]["code"], "IMAGE_DECODE_ERROR");
}

/// POST /v1/detect with no "image" field in the multipart — should return 400.
/// The handler calls `extract_field("image")` before model lookup, so the model
/// name is irrelevant.
#[tokio::test]
#[ignore]
async fn test_missing_image_field() {
    let server = common::TestServer::start().await;

    // Send a multipart form with a field that is NOT "image".
    let form = multipart::Form::new().text("not_image", "irrelevant");

    let resp = server
        .client
        .post(format!("{}/v1/detect?model=anything", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        400,
        "missing 'image' field should produce 400"
    );
    let body: Value = resp.json().await.unwrap();
    assert_error_shape(&body, 400);
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("image"),
        "error message should mention the missing 'image' field, got: {msg}"
    );
}

/// POST /v1/detect?model=nonexistent — should return 404.
/// The handler extracts the image field first (succeeds), then fails at model lookup.
#[tokio::test]
#[ignore]
async fn test_model_not_found() {
    let server = common::TestServer::start().await;

    let form = multipart::Form::new().part(
        "image",
        multipart::Part::bytes(b"payload".to_vec()).file_name("test.bin"),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect?model=nonexistent_model_xyz",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 404, "unknown model should produce 404");
    let body: Value = resp.json().await.unwrap();
    assert_error_shape(&body, 404);
    assert_eq!(body["error"]["code"], "MANIFEST_NOT_FOUND");
}

// ---------------------------------------------------------------------------
// Error response format
// ---------------------------------------------------------------------------

/// Verify the full error response shape: `{ "error": { "code", "message", "status" } }`.
/// Uses a model-not-found error as the trigger.
#[tokio::test]
#[ignore]
async fn test_error_response_shape() {
    let server = common::TestServer::start().await;

    let form = multipart::Form::new().part(
        "image",
        multipart::Part::bytes(b"data".to_vec()).file_name("test.bin"),
    );

    let resp = server
        .client
        .post(format!("{}/v1/detect?model=nonexistent", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();

    let status = resp.status().as_u16();
    assert!(status >= 400, "expected an error status code, got {status}");

    let body: Value = resp.json().await.unwrap();

    // Top-level must be an object containing "error".
    assert!(body.is_object(), "response must be a JSON object");
    let obj = body.as_object().unwrap();
    assert!(
        obj.contains_key("error"),
        "response must contain 'error' key"
    );

    // "error" must be an object with exactly code, message, status.
    let error = body["error"]
        .as_object()
        .expect("'error' must be an object");
    assert!(error.contains_key("code"), "error must contain 'code'");
    assert!(
        error.contains_key("message"),
        "error must contain 'message'"
    );
    assert!(error.contains_key("status"), "error must contain 'status'");

    // Type checks.
    assert!(error["code"].is_string(), "code must be a string");
    assert!(error["message"].is_string(), "message must be a string");
    assert!(error["status"].is_number(), "status must be a number");

    // status field must match HTTP status code.
    assert_eq!(
        error["status"].as_u64().unwrap(),
        status as u64,
        "error.status must match HTTP status code"
    );
}

// ---------------------------------------------------------------------------
// Semaphore (concurrency limit)
// ---------------------------------------------------------------------------

/// Fire many concurrent requests to trigger semaphore exhaustion (503).
///
/// KNOWN FLAKY: concurrent multipart requests from the same tokio runtime
/// sometimes fail at the HTTP layer (400) before reaching the semaphore.
/// The semaphore implementation was verified correct in code review (R1 B01 fix).
/// This test validates the behavior when it works; failures indicate a test
/// infrastructure issue, not a server bug.
#[tokio::test]
#[ignore]
async fn test_semaphore_503() {
    let server = common::TestServer::start().await;
    let model_id = first_detector_model_id(&server).await;

    // Read a real test image so inference takes measurable time.
    let image_bytes = std::fs::read(common::test_image_path()).unwrap();

    // Fire many concurrent requests. Shared server has max_concurrent=4,
    // so with 20 simultaneous requests most should get 503.
    let n = 20;
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let client = server.client.clone();
        let url = format!("{}/v1/detect?model={}", server.base_url, model_id);
        let bytes = image_bytes.clone();
        handles.push(tokio::spawn(async move {
            let form = multipart::Form::new()
                .part("image", multipart::Part::bytes(bytes).file_name("test.jpg"));
            let resp = client.post(&url).multipart(form).send().await.unwrap();
            let status = resp.status();
            let body: Value = resp.json().await.unwrap_or(Value::Null);
            (status, body)
        }));
    }

    let mut got_503 = false;
    let mut statuses = Vec::with_capacity(n);
    for h in handles {
        let (status, body) = h.await.unwrap();
        statuses.push(status.as_u16());
        if status == 503 {
            got_503 = true;
            assert_error_shape(&body, 503);
            assert_eq!(body["error"]["code"], "SERVICE_OVERLOADED");
        }
    }

    assert!(
        got_503,
        "expected at least one 503 with max_concurrent_inference=4 and {n} concurrent requests, got: {statuses:?}"
    );
}

// ---------------------------------------------------------------------------
// Batch limits
// ---------------------------------------------------------------------------

/// POST /v1/detect/batch with more images than max_batch_size — should return 413.
/// Shared test server has max_batch_size=16, so we send 17 images.
#[tokio::test]
#[ignore]
async fn test_batch_too_large() {
    let server = common::TestServer::start().await;

    let mut form = multipart::Form::new();
    for i in 0..17 {
        form = form.part(
            "images",
            multipart::Part::bytes(vec![0u8; 4]).file_name(format!("{i}.bin")),
        );
    }

    let resp = server
        .client
        .post(format!("{}/v1/detect/batch?model=any", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        413,
        "batch exceeding limit should produce 413"
    );
    let body: Value = resp.json().await.unwrap();
    assert_error_shape(&body, 413);
    assert_eq!(body["error"]["code"], "PAYLOAD_TOO_LARGE");
}
