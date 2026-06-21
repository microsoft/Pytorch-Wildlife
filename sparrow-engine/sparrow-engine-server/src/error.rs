use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::engine_dispatch::SparrowEngineError;

/// Unified error type for handlers.
pub enum AppError {
    Bongo(SparrowEngineError),
    Http {
        status: StatusCode,
        code: String,
        message: String,
    },
}

impl AppError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        AppError::Http {
            status: StatusCode::BAD_REQUEST,
            code: "BAD_REQUEST".to_string(),
            message: message.into(),
        }
    }

    pub fn from_json_rejection(rejection: JsonRejection) -> Self {
        let status = rejection.status();
        let code = match status {
            StatusCode::BAD_REQUEST => "BAD_REQUEST",
            StatusCode::UNSUPPORTED_MEDIA_TYPE => "UNSUPPORTED_MEDIA_TYPE",
            StatusCode::UNPROCESSABLE_ENTITY => "UNPROCESSABLE_ENTITY",
            StatusCode::PAYLOAD_TOO_LARGE => "PAYLOAD_TOO_LARGE",
            _ => "BAD_REQUEST",
        };
        AppError::Http {
            status,
            code: code.to_string(),
            message: rejection.body_text(),
        }
    }

    pub fn model_not_loaded(model_id: &str) -> Self {
        AppError::Http {
            status: StatusCode::NOT_FOUND,
            code: "MODEL_NOT_LOADED".to_string(),
            message: format!("Model '{model_id}' is not loaded. Load it via POST /v1/models/load."),
        }
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        AppError::Http {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "PAYLOAD_TOO_LARGE".to_string(),
            message: message.into(),
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        AppError::Http {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "SERVICE_OVERLOADED".to_string(),
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        AppError::Http {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "INTERNAL_ERROR".to_string(),
            message: message.into(),
        }
    }
}

impl From<SparrowEngineError> for AppError {
    fn from(e: SparrowEngineError) -> Self {
        AppError::Bongo(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::Http {
                status,
                code,
                message,
            } => error_json(status, &code, &message),
            AppError::Bongo(e) => bongo_into_response(e),
        }
    }
}

fn bongo_into_response(e: SparrowEngineError) -> Response {
    use SparrowEngineError::*;
    let (status, code) = match &e {
        // Engine lifecycle
        EngineAlreadyExists => (StatusCode::INTERNAL_SERVER_ERROR, "ENGINE_ALREADY_EXISTS"),
        EngineFreed => (StatusCode::SERVICE_UNAVAILABLE, "ENGINE_UNAVAILABLE"),
        // Model loading
        ManifestNotFound(_) => (StatusCode::NOT_FOUND, "MANIFEST_NOT_FOUND"),
        InvalidManifest(_) => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_MANIFEST"),
        UnsupportedFormat { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "UNSUPPORTED_FORMAT"),
        OutputShapeMismatch { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "OUTPUT_SHAPE_MISMATCH"),
        PathTraversal(_) => (StatusCode::BAD_REQUEST, "PATH_TRAVERSAL"),
        LabelFileNotFound(_) => (StatusCode::UNPROCESSABLE_ENTITY, "LABEL_FILE_NOT_FOUND"),
        InvalidLabelFormat(_) => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_LABEL_FORMAT"),
        // Model usage
        ModelUnloaded => (StatusCode::GONE, "MODEL_UNLOADED"),
        NotADetector { .. } => (StatusCode::BAD_REQUEST, "WRONG_MODEL_TYPE"),
        NotAClassifier { .. } => (StatusCode::BAD_REQUEST, "WRONG_MODEL_TYPE"),
        // Pipeline
        PipelineNotFound { .. } => (StatusCode::NOT_FOUND, "PIPELINE_NOT_FOUND"),
        PipelineMissingModels { .. } => (
            StatusCode::from_u16(424).unwrap(),
            "PIPELINE_MISSING_MODELS",
        ),
        InvalidPipeline(_) => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_PIPELINE"),
        IncompatiblePipeline { .. } => (StatusCode::BAD_REQUEST, "INCOMPATIBLE_PIPELINE"),
        EmptyPipeline => (StatusCode::BAD_REQUEST, "EMPTY_PIPELINE"),
        // Manifest validation
        MissingTiledFields => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_MANIFEST"),
        WrongManifestType => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_MANIFEST"),
        WrongPipelineType => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_MANIFEST"),
        // Audio
        AudioDecode(_) => (StatusCode::UNPROCESSABLE_ENTITY, "AUDIO_DECODE_ERROR"),
        AudioPreprocess(_) => (StatusCode::UNPROCESSABLE_ENTITY, "AUDIO_PREPROCESS_ERROR"),
        Resample(_) => (StatusCode::UNPROCESSABLE_ENTITY, "RESAMPLE_ERROR"),
        NotAnAudioModel { .. } => (StatusCode::BAD_REQUEST, "WRONG_MODEL_TYPE"),
        IsAudioModel { .. } => (StatusCode::BAD_REQUEST, "WRONG_MODEL_TYPE"),
        // Image
        ImageDecode(_) => (StatusCode::UNPROCESSABLE_ENTITY, "IMAGE_DECODE_ERROR"),
        InvalidStride { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_IMAGE_INPUT"),
        ImageFileNotFound(_) => (StatusCode::NOT_FOUND, "IMAGE_NOT_FOUND"),
        // GPU resources
        NvjpegUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "NVJPEG_UNAVAILABLE"),
        // Required runtime is missing, so the service cannot serve this model yet.
        TrtRuntimeMissing(_) => (StatusCode::SERVICE_UNAVAILABLE, "TRT_RUNTIME_UNAVAILABLE"),
        // ORT / IO
        Ort(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_ERROR"),
        Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "IO_ERROR"),
        TomlParse(_) => (StatusCode::UNPROCESSABLE_ENTITY, "MANIFEST_PARSE_ERROR"),
        Json(_) => (StatusCode::INTERNAL_SERVER_ERROR, "JSON_ERROR"),
    };
    error_json(status, code, &e.to_string())
}

fn error_json(status: StatusCode, code: &str, message: &str) -> Response {
    let body = json!({
        "error": {
            "code": code,
            "message": message,
            "status": status.as_u16(),
        }
    });
    (status, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::ModelType;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn pipeline_compat_errors_map_to_bad_request() {
        let incompatible = AppError::from(SparrowEngineError::IncompatiblePipeline {
            detector: Some(ModelType::AudioDetector),
            classifier: Some(ModelType::Classifier),
            reason: "modality mismatch",
        })
        .into_response();
        assert_eq!(incompatible.status(), StatusCode::BAD_REQUEST);
        let body = error_body(incompatible).await;
        assert_eq!(body["error"]["status"], 400);
        assert_eq!(body["error"]["code"], "INCOMPATIBLE_PIPELINE");

        let empty = AppError::from(SparrowEngineError::EmptyPipeline).into_response();
        assert_eq!(empty.status(), StatusCode::BAD_REQUEST);
        let body = error_body(empty).await;
        assert_eq!(body["error"]["status"], 400);
        assert_eq!(body["error"]["code"], "EMPTY_PIPELINE");
    }

    #[tokio::test]
    async fn trt_runtime_missing_maps_to_service_unavailable() {
        let response = AppError::from(SparrowEngineError::TrtRuntimeMissing(
            "Model detector requires TensorRT but libnvinfer was not found.".to_string(),
        ))
        .into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = error_body(response).await;
        assert_eq!(body["error"]["status"], 503);
        assert_eq!(body["error"]["code"], "TRT_RUNTIME_UNAVAILABLE");
    }

    async fn error_body(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
}
