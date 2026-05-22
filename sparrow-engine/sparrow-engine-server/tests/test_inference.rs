//! Integration tests for sparrow-engine-server inference endpoints.
//!
//! All tests are `#[ignore]` — they require ORT runtime and model files.
//!
//! Run with:
//! ```sh
//! ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 LD_LIBRARY_PATH=/tmp/ort-lib \
//!   cargo test -p sparrow-engine-server --test test_inference -- --ignored --test-threads=1
//! ```

mod common;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Response types for deserialization
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DetectResponse {
    model_id: String,
    image_size: [u32; 2],
    #[allow(dead_code)]
    processing_time_ms: f32,
    detections: Vec<Detection>,
}

#[derive(Deserialize)]
struct Detection {
    label: String,
    #[allow(dead_code)]
    label_id: u32,
    confidence: f32,
    bbox: BBox,
}

#[derive(Deserialize)]
struct BBox {
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
}

#[derive(Deserialize)]
struct BatchDetectResponse {
    model_id: String,
    count: usize,
    #[allow(dead_code)]
    processing_time_ms: f32,
    results: Vec<BatchDetectResultItem>,
}

#[derive(Deserialize)]
struct BatchDetectResultItem {
    #[allow(dead_code)]
    index: usize,
    image_size: [u32; 2],
    detections: Vec<Detection>,
}

#[derive(Deserialize)]
struct ClassifyResponse {
    model_id: String,
    image_size: [u32; 2],
    #[allow(dead_code)]
    processing_time_ms: f32,
    classifications: Vec<Classification>,
}

#[derive(Deserialize)]
struct Classification {
    #[allow(dead_code)]
    label: String,
    #[allow(dead_code)]
    label_id: u32,
    confidence: f32,
}

#[derive(Deserialize)]
struct AudioDetectResponse {
    model_id: String,
    duration_s: f32,
    sample_rate: u32,
    #[allow(dead_code)]
    processing_time_ms: f32,
    segments: Vec<AudioSegment>,
}

#[derive(Deserialize)]
struct AudioSegment {
    start_time_s: f32,
    end_time_s: f32,
    confidence: f32,
}

#[derive(Deserialize)]
struct PipelineResponse {
    pipeline_id: String,
    image_size: [u32; 2],
    #[allow(dead_code)]
    processing_time_ms: f32,
    detections: Vec<PipelineDetection>,
}

#[derive(Deserialize)]
struct PipelineDetection {
    #[allow(dead_code)]
    label: String,
    #[allow(dead_code)]
    label_id: u32,
    confidence: f32,
    bbox: BBox,
    #[allow(dead_code)]
    classification: Option<Classification>,
}

// ---------------------------------------------------------------------------
// Test file paths
// ---------------------------------------------------------------------------

fn test_files_dir() -> std::path::PathBuf {
    std::path::PathBuf::from("/home/miao/repos/PW_refactor/test_files")
}

fn cameratrap_image() -> std::path::PathBuf {
    test_files_dir()
        .join("test_cameratrap")
        .join("0a64f82b-8dc4-47b8-b408-1c865805edd1.jpg")
}

fn overhead_image() -> std::path::PathBuf {
    test_files_dir()
        .join("test_overhead")
        .join("S_11_05_16_DSC01556.JPG")
}

fn test_audio_wav() -> std::path::PathBuf {
    test_files_dir()
        .join("test_audio")
        .join("G010_timelapse_20250629.wav")
}

/// Get N camera trap image paths for batch tests.
fn cameratrap_images(n: usize) -> Vec<std::path::PathBuf> {
    let dir = test_files_dir().join("test_cameratrap");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read test_cameratrap dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "jpg" || ext == "JPG" || ext == "jpeg")
        })
        .collect();
    entries.sort();
    entries.truncate(n);
    assert!(
        entries.len() == n,
        "Need {} camera trap images, found {}",
        n,
        entries.len()
    );
    entries
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

fn assert_bbox_normalized(bbox: &BBox) {
    assert!(
        bbox.x_min >= 0.0 && bbox.x_min <= 1.0,
        "x_min out of [0,1]: {}",
        bbox.x_min
    );
    assert!(
        bbox.y_min >= 0.0 && bbox.y_min <= 1.0,
        "y_min out of [0,1]: {}",
        bbox.y_min
    );
    assert!(
        bbox.x_max >= 0.0 && bbox.x_max <= 1.0,
        "x_max out of [0,1]: {}",
        bbox.x_max
    );
    assert!(
        bbox.y_max >= 0.0 && bbox.y_max <= 1.0,
        "y_max out of [0,1]: {}",
        bbox.y_max
    );
    assert!(bbox.x_max >= bbox.x_min, "x_max < x_min");
    assert!(bbox.y_max >= bbox.y_min, "y_max < y_min");
}

fn assert_confidence_valid(confidence: f32) {
    assert!(
        confidence.is_finite() && confidence >= 0.0,
        "confidence must be finite and non-negative: {}",
        confidence
    );
    // Note: heatmap models (HerdNet, OWL-T) can produce peak values > 1.0
    // (raw heatmap intensity, not probability). Only check finite/non-negative.
}

fn assert_probability_valid(confidence: f32) {
    assert!(
        confidence.is_finite() && (0.0..=1.0).contains(&confidence),
        "probability confidence must be finite and in [0,1]: {}",
        confidence
    );
}

// ---------------------------------------------------------------------------
// Single detection tests
// ---------------------------------------------------------------------------

/// POST /v1/detect?model=megadetector-v6-yolov10e — camera trap image.
/// Expect non-empty detections with normalized bboxes and valid confidences.
#[tokio::test]
#[ignore]
async fn test_detect_mdv6() {
    let server = common::TestServer::start().await;
    let image_path = cameratrap_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect?model=megadetector-v6-yolov10e",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send detect request");

    assert_eq!(resp.status(), 200, "detect endpoint returned non-200");

    let body: DetectResponse = resp.json().await.expect("parse detect response");

    assert_eq!(body.model_id, "megadetector-v6-yolov10e");
    assert!(body.image_size[0] > 0 && body.image_size[1] > 0);
    assert!(
        !body.detections.is_empty(),
        "MDV6 should detect animals in camera trap image"
    );

    for det in &body.detections {
        assert_confidence_valid(det.confidence);
        assert_bbox_normalized(&det.bbox);
        assert!(!det.label.is_empty(), "detection label should not be empty");
    }
}

/// POST /v1/detect?model=deepfaune-yolo8s — camera trap image.
#[tokio::test]
#[ignore]
async fn test_detect_deepfaune() {
    let server = common::TestServer::start().await;
    let image_path = cameratrap_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect?model=deepfaune-yolo8s",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send detect request");

    assert_eq!(resp.status(), 200, "detect endpoint returned non-200");

    let body: DetectResponse = resp.json().await.expect("parse detect response");

    assert_eq!(body.model_id, "deepfaune-yolo8s");
    assert!(body.image_size[0] > 0 && body.image_size[1] > 0);
    assert!(
        !body.detections.is_empty(),
        "DeepFaune should detect animals in camera trap image"
    );

    for det in &body.detections {
        assert_confidence_valid(det.confidence);
        assert_bbox_normalized(&det.bbox);
    }
}

/// POST /v1/detect?model=herdnet-general-2022 — overhead image (tiled detection).
#[tokio::test]
#[ignore]
async fn test_detect_tiled_herdnet() {
    let server = common::TestServer::start().await;
    let image_path = overhead_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect?model=herdnet-general-2022",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send detect request");

    assert_eq!(resp.status(), 200, "detect endpoint returned non-200");

    let body: DetectResponse = resp.json().await.expect("parse detect response");

    assert_eq!(body.model_id, "herdnet-general-2022");
    // Overhead image is 6000x4000
    assert_eq!(body.image_size, [6000, 4000]);
    assert!(
        !body.detections.is_empty(),
        "HerdNet should detect animals in overhead image"
    );

    for det in &body.detections {
        assert_confidence_valid(det.confidence);
        assert_bbox_normalized(&det.bbox);
    }
}

// ---------------------------------------------------------------------------
// Batch detection
// ---------------------------------------------------------------------------

/// POST /v1/detect/batch?model=megadetector-v6-yolov10e — 3 images.
/// Assert results.len() == 3 and each result has valid structure.
#[tokio::test]
#[ignore]
async fn test_detect_batch() {
    let server = common::TestServer::start().await;
    let images = cameratrap_images(3);

    let mut form = reqwest::multipart::Form::new();
    for img_path in &images {
        let part = reqwest::multipart::Part::bytes(std::fs::read(img_path).expect("read image"))
            .file_name(
                img_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
            )
            .mime_str("image/jpeg")
            .unwrap();
        form = form.part("images", part);
    }

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect/batch?model=megadetector-v6-yolov10e",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send batch detect request");

    assert_eq!(resp.status(), 200, "batch detect endpoint returned non-200");

    let body: BatchDetectResponse = resp.json().await.expect("parse batch detect response");

    assert_eq!(body.model_id, "megadetector-v6-yolov10e");
    assert_eq!(body.count, 3);
    assert_eq!(body.results.len(), 3, "should have one result per image");

    for item in &body.results {
        assert!(item.image_size[0] > 0 && item.image_size[1] > 0);
        for det in &item.detections {
            assert_confidence_valid(det.confidence);
            assert_bbox_normalized(&det.bbox);
        }
    }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// POST /v1/classify?model=speciesnet-crop — camera trap image.
/// Assert classifications non-empty, sorted by confidence descending.
#[tokio::test]
#[ignore]
async fn test_classify_speciesnet() {
    let server = common::TestServer::start().await;
    let image_path = cameratrap_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/classify?model=speciesnet-crop",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send classify request");

    assert_eq!(resp.status(), 200, "classify endpoint returned non-200");

    let body: ClassifyResponse = resp.json().await.expect("parse classify response");

    assert_eq!(body.model_id, "speciesnet-crop");
    assert!(body.image_size[0] > 0 && body.image_size[1] > 0);
    assert!(
        !body.classifications.is_empty(),
        "SpeciesNet should return classifications"
    );

    for cls in &body.classifications {
        assert_probability_valid(cls.confidence);
    }

    // Top classification should have the highest confidence.
    if body.classifications.len() > 1 {
        for window in body.classifications.windows(2) {
            assert!(
                window[0].confidence >= window[1].confidence,
                "Classifications not sorted by confidence: {} < {}",
                window[0].confidence,
                window[1].confidence,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Audio detection
// ---------------------------------------------------------------------------

/// POST /v1/audio/detect?model=md-audiobirds-v1 — WAV file.
/// Assert segments are returned with valid time ranges and confidences.
#[tokio::test]
#[ignore]
async fn test_audio_detect() {
    let server = common::TestServer::start().await;
    let audio_path = test_audio_wav();

    let form = reqwest::multipart::Form::new().part(
        "audio",
        reqwest::multipart::Part::bytes(std::fs::read(&audio_path).expect("read audio"))
            .file_name(audio_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("audio/wav")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/audio/detect?model=md-audiobirds-v1",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send audio detect request");

    assert_eq!(resp.status(), 200, "audio detect endpoint returned non-200");

    let body: AudioDetectResponse = resp.json().await.expect("parse audio detect response");

    assert_eq!(body.model_id, "md-audiobirds-v1");
    assert!(body.duration_s > 0.0, "audio duration should be positive");
    assert!(body.sample_rate > 0, "sample rate should be positive");
    assert!(
        !body.segments.is_empty(),
        "AudioBirds should detect bird segments in wildlife audio"
    );

    for seg in &body.segments {
        assert_probability_valid(seg.confidence);
        assert!(
            seg.start_time_s >= 0.0,
            "segment start_time_s should be >= 0"
        );
        assert!(
            seg.end_time_s > seg.start_time_s,
            "segment end_time_s ({}) should be > start_time_s ({})",
            seg.end_time_s,
            seg.start_time_s
        );
        assert!(
            seg.end_time_s <= body.duration_s + 0.1, // small tolerance for rounding
            "segment end_time_s ({}) should not exceed audio duration ({})",
            seg.end_time_s,
            body.duration_s
        );
    }
}

// ---------------------------------------------------------------------------
// Pipeline (detect + classify)
// ---------------------------------------------------------------------------

/// POST /v1/pipeline?pipeline=... — camera trap image.
/// Only runs if a pipeline manifest exists in the test model directory.
/// This test is conditional: skip if no pipeline is configured.
#[tokio::test]
#[ignore]
async fn test_pipeline() {
    let server = common::TestServer::start().await;

    // Check if any pipeline is loaded by querying the list endpoint.
    let resp = server
        .client
        .get(format!("{}/v1/pipelines", server.base_url))
        .send()
        .await
        .expect("query pipelines list");

    assert_eq!(resp.status(), 200);

    #[derive(Deserialize)]
    struct PipelinesListResponse {
        pipelines: Vec<PipelineEntry>,
    }
    #[derive(Deserialize)]
    struct PipelineEntry {
        id: String,
    }

    let list: PipelinesListResponse = resp.json().await.expect("parse pipelines list");
    if list.pipelines.is_empty() {
        eprintln!("SKIP: no pipelines loaded — pipeline test requires a pipeline manifest");
        return;
    }

    let pipeline_id = &list.pipelines[0].id;
    let image_path = cameratrap_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/pipeline?pipeline={}",
            server.base_url, pipeline_id
        ))
        .multipart(form)
        .send()
        .await
        .expect("send pipeline request");

    assert_eq!(resp.status(), 200, "pipeline endpoint returned non-200");

    let body: PipelineResponse = resp.json().await.expect("parse pipeline response");

    assert_eq!(body.pipeline_id, *pipeline_id);
    assert!(body.image_size[0] > 0 && body.image_size[1] > 0);

    for det in &body.detections {
        assert_confidence_valid(det.confidence);
        assert_bbox_normalized(&det.bbox);
    }
}

// ---------------------------------------------------------------------------
// Error case: model not loaded
// ---------------------------------------------------------------------------

/// POST /v1/detect with a non-existent model should return 404.
#[tokio::test]
#[ignore]
async fn test_detect_model_not_found() {
    let server = common::TestServer::start().await;
    let image_path = cameratrap_image();

    let form = reqwest::multipart::Form::new().part(
        "image",
        reqwest::multipart::Part::bytes(std::fs::read(&image_path).expect("read image"))
            .file_name(image_path.file_name().unwrap().to_str().unwrap().to_string())
            .mime_str("image/jpeg")
            .unwrap(),
    );

    let resp = server
        .client
        .post(format!(
            "{}/v1/detect?model=nonexistent-model",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .expect("send detect request");

    assert_eq!(
        resp.status(),
        404,
        "requesting non-existent model should return 404"
    );
}
