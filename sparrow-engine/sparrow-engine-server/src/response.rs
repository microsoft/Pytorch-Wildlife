use serde::Serialize;

use crate::engine_dispatch::{AudioSegment, BBox, Classification, Detection, PipelineDetection};

// ---------------------------------------------------------------------------
// Bbox (object format with named fields)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct BBoxResponse {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl From<BBox> for BBoxResponse {
    fn from(b: BBox) -> Self {
        Self {
            x_min: b.x_min,
            y_min: b.y_min,
            x_max: b.x_max,
            y_max: b.y_max,
        }
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct DetectionResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    pub bbox: BBoxResponse,
}

impl From<Detection> for DetectionResponse {
    fn from(d: Detection) -> Self {
        Self {
            label: d.label,
            label_id: d.label_id,
            confidence: d.confidence,
            bbox: d.bbox.into(),
        }
    }
}

#[derive(Serialize)]
pub struct DetectResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub detections: Vec<DetectionResponse>,
}

// ---------------------------------------------------------------------------
// Batch detection
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct BatchDetectResultItem {
    pub index: usize,
    pub image_size: [u32; 2],
    pub detections: Vec<DetectionResponse>,
}

#[derive(Serialize)]
pub struct BatchDetectResponse {
    pub model_id: String,
    pub count: usize,
    pub processing_time_ms: f32,
    pub results: Vec<BatchDetectResultItem>,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ClassificationResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

impl From<Classification> for ClassificationResponse {
    fn from(c: Classification) -> Self {
        Self {
            label: c.label,
            label_id: c.label_id,
            confidence: c.confidence,
        }
    }
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub classifications: Vec<ClassificationResponse>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct PipelineDetectionResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    pub bbox: BBoxResponse,
    pub classification: Option<ClassificationResponse>,
}

impl From<PipelineDetection> for PipelineDetectionResponse {
    fn from(pd: PipelineDetection) -> Self {
        Self {
            label: pd.detection.label,
            label_id: pd.detection.label_id,
            confidence: pd.detection.confidence,
            bbox: pd.detection.bbox.into(),
            classification: pd.classification.map(Into::into),
        }
    }
}

#[derive(Serialize)]
pub struct PipelineResponse {
    pub pipeline_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub detections: Vec<PipelineDetectionResponse>,
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct AudioSegmentResponse {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
}

impl From<AudioSegment> for AudioSegmentResponse {
    fn from(s: AudioSegment) -> Self {
        Self {
            start_time_s: s.start_time_s,
            end_time_s: s.end_time_s,
            confidence: s.confidence,
        }
    }
}

#[derive(Serialize)]
pub struct AudioDetectResponse {
    pub model_id: String,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
    pub segments: Vec<AudioSegmentResponse>,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub models_loaded: usize,
    pub pipelines_loaded: usize,
    /// Phase 4.2: total parseable manifests discovered at boot. Lets operators
    /// distinguish "lazy-empty but ready" (catalog_size > 0, models_loaded = 0)
    /// from "discovery failed" (catalog_size = 0).
    pub catalog_size: usize,
    pub version: String,
}
