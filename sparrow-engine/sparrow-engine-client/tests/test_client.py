"""Tests for sparrow_engine_client using pytest-httpserver to mock sparrow-engine-server."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx
import pytest

# Add parent dir so sparrow_engine_client is importable without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparrow_engine_client import (
    AudioResult,
    BBox,
    SparrowEngineClient,
    SparrowEngineClientError,
    ClassifyResult,
    DetectResult,
    ModelInfo,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(httpserver):
    """SparrowEngineClient pointed at the pytest-httpserver."""
    with SparrowEngineClient(base_url=httpserver.url_for(""), timeout=5.0) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample response payloads (match sparrow-engine-server JSON exactly)
# ---------------------------------------------------------------------------

DETECT_RESPONSE = {
    "model_id": "megadetector_v6",
    "image_size": [1920, 1080],
    "processing_time_ms": 42.5,
    "detections": [
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.95,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
        }
    ],
}

CLASSIFY_RESPONSE = {
    "model_id": "speciesnet",
    "image_size": [224, 224],
    "processing_time_ms": 15.3,
    "classifications": [
        {"label": "deer", "label_id": 3, "confidence": 0.88},
        {"label": "elk", "label_id": 7, "confidence": 0.05},
    ],
}

AUDIO_RESPONSE = {
    "model_id": "md_audiobirds_v1",
    "duration_s": 10.0,
    "sample_rate": 48000,
    "processing_time_ms": 120.0,
    "segments": [
        {"start_time_s": 1.5, "end_time_s": 3.2, "confidence": 0.92},
    ],
}

BATCH_DETECT_RESPONSE = {
    "model_id": "megadetector_v6",
    "count": 2,
    "processing_time_ms": 85.0,
    "results": [
        {
            "index": 0,
            "image_size": [1920, 1080],
            "detections": [
                {
                    "label": "animal",
                    "label_id": 1,
                    "confidence": 0.93,
                    "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.7},
                }
            ],
        },
        {
            "index": 1,
            "image_size": [640, 480],
            "detections": [],
        },
    ],
}

PIPELINE_RESPONSE = {
    "pipeline_id": "md_speciesnet",
    "model_id": None,
    "image_size": [1920, 1080],
    "processing_time_ms": 65.0,
    "detections": [
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.95,
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.6},
            "classification": {"label": "deer", "label_id": 3, "confidence": 0.88},
        },
        {
            "label": "animal",
            "label_id": 1,
            "confidence": 0.70,
            "bbox": {"x_min": 0.6, "y_min": 0.1, "x_max": 0.9, "y_max": 0.4},
            "classification": None,
        },
    ],
}

MODELS_RESPONSE = {
    "models": [
        {
            "id": "megadetector_v6",
            "model_type": "detector",
            "default": True,
            "version": "1.0.0",
            "description": "MegaDetector v6 general wildlife detector",
            "onnx_sha256": "abcdef0123456789",
            "onnx_size_bytes": 123456789,
        },
        {"id": "speciesnet", "model_type": "classifier", "default": False},
    ]
}

LOAD_MODEL_RESPONSE = {
    "id": "megadetector_v6",
    "model_type": "detector",
    "default": True,
    "version": "1.0.0",
    "description": "MegaDetector v6 general wildlife detector",
    "onnx_sha256": "abcdef0123456789",
    "onnx_size_bytes": 123456789,
}

HEALTH_RESPONSE = {
    "status": "ready",
    "models_loaded": 2,
    "pipelines_loaded": 1,
    "version": "0.1.0",
}

ERROR_RESPONSE = {
    "error": {
        "code": "MODEL_NOT_LOADED",
        "message": "Model 'foo' is not loaded.",
        "status": 499,
    }
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect(httpserver, client):
    httpserver.expect_request(
        "/v1/detect",
        method="POST",
        query_string="model=megadetector_v6&threshold=0.4&max_detections=10&store=true&halt_on_store_failure=true",
    ).respond_with_json(DETECT_RESPONSE)
    result = client.detect(
        image=b"\xff\xd8fake-jpeg",
        model="megadetector_v6",
        threshold=0.4,
        max_detections=10,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, DetectResult)
    assert result.model_id == "megadetector_v6"
    assert result.image_size == (1920, 1080)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.label == "animal"
    assert det.label_id == 1
    assert det.confidence == pytest.approx(0.95)
    assert det.bbox.x_min == pytest.approx(0.1)
    assert det.bbox.y_max == pytest.approx(0.6)


def test_classify(httpserver, client):
    httpserver.expect_request(
        "/v1/classify",
        method="POST",
        query_string="model=speciesnet&top_k=3&store=true&halt_on_store_failure=true",
    ).respond_with_json(CLASSIFY_RESPONSE)
    result = client.classify(
        image=b"\xff\xd8fake-jpeg",
        model="speciesnet",
        top_k=3,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, ClassifyResult)
    assert result.model_id == "speciesnet"
    assert len(result.classifications) == 2
    assert result.classifications[0].label == "deer"
    assert result.classifications[0].confidence == pytest.approx(0.88)


def test_detect_audio(httpserver, client):
    httpserver.expect_request(
        "/v1/audio/detect",
        method="POST",
        query_string="model=md_audiobirds_v1&threshold=0.7&segment_duration=1.5&stride=0.5&store=true&halt_on_store_failure=true",
    ).respond_with_json(AUDIO_RESPONSE)
    result = client.detect_audio(
        audio=b"fake-wav-data",
        model="md_audiobirds_v1",
        threshold=0.7,
        segment_duration=1.5,
        stride=0.5,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, AudioResult)
    assert result.model_id == "md_audiobirds_v1"
    assert result.duration_s == pytest.approx(10.0)
    assert result.sample_rate == 48000
    assert len(result.segments) == 1
    assert result.segments[0].start_time_s == pytest.approx(1.5)


def test_detect_batch(httpserver, client):
    httpserver.expect_request(
        "/v1/detect/batch",
        method="POST",
        query_string="model=megadetector_v6&threshold=0.5&max_detections=2&store=true&halt_on_store_failure=true",
    ).respond_with_json(BATCH_DETECT_RESPONSE)
    results = client.detect_batch(
        images=[b"\xff\xd8img1", b"\xff\xd8img2"],
        model="megadetector_v6",
        threshold=0.5,
        max_detections=2,
        store=True,
        halt_on_store_failure=True,
    )
    assert len(results) == 2
    assert results[0].model_id == "megadetector_v6"
    assert results[0].image_size == (1920, 1080)
    assert len(results[0].detections) == 1
    assert results[1].image_size == (640, 480)
    assert len(results[1].detections) == 0


def test_pipeline(httpserver, client):
    httpserver.expect_request(
        "/v1/pipeline",
        method="POST",
        query_string="pipeline=md_speciesnet&top_k=2&threshold=0.3&max_detections=5&store=true&halt_on_store_failure=true",
    ).respond_with_json(PIPELINE_RESPONSE)
    result = client.pipeline(
        image=b"\xff\xd8fake",
        pipeline="md_speciesnet",
        threshold=0.3,
        top_k=2,
        max_detections=5,
        store=True,
        halt_on_store_failure=True,
    )
    assert isinstance(result, PipelineResult)
    assert result.pipeline_id == "md_speciesnet"
    assert len(result.detections) == 2
    # First detection has classification
    pd0 = result.detections[0]
    assert pd0.detection.label == "animal"
    assert pd0.classification is not None
    assert pd0.classification.label == "deer"
    # Second detection has no classification
    pd1 = result.detections[1]
    assert pd1.classification is None


def test_list_models(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        MODELS_RESPONSE
    )
    models = client.list_models()
    assert len(models) == 2
    assert isinstance(models[0], ModelInfo)
    assert models[0].id == "megadetector_v6"
    assert models[0].default is True
    assert models[0].version == "1.0.0"
    assert models[0].description == "MegaDetector v6 general wildlife detector"
    assert models[0].onnx_sha256 == "abcdef0123456789"
    assert models[0].onnx_size_bytes == 123456789
    assert models[1].model_type == "classifier"
    assert models[1].default is False
    assert models[1].version is None
    assert models[1].onnx_sha256 is None


def test_load_model(httpserver, client):
    httpserver.expect_request("/v1/models/load", method="POST").respond_with_json(
        LOAD_MODEL_RESPONSE
    )
    info = client.load_model("megadetector_v6")
    assert isinstance(info, ModelInfo)
    assert info.id == "megadetector_v6"
    assert info.model_type == "detector"
    assert info.default is True
    assert info.version == "1.0.0"
    assert info.onnx_sha256 == "abcdef0123456789"


def test_list_models_legacy_wire_format(httpserver, client):
    legacy_response = {
        "models": [
            {"id": "megadetector_v6", "model_type": "detector"},
            {"id": "speciesnet", "model_type": "classifier"},
        ]
    }
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        legacy_response
    )
    models = client.list_models()
    assert len(models) == 2
    assert models[0].id == "megadetector_v6"
    assert models[0].default is False
    assert models[0].version is None
    assert models[0].description is None
    assert models[0].onnx_sha256 is None
    assert models[0].onnx_size_bytes is None


def test_unload_model(httpserver, client):
    httpserver.expect_request(
        "/v1/models/megadetector_v6", method="DELETE"
    ).respond_with_data("", status=204)
    client.unload_model("megadetector_v6")  # should not raise


def test_unload_model_escapes_model_id():
    class StubClient:
        path: str | None = None

        def delete(self, path: str) -> httpx.Response:
            self.path = path
            return httpx.Response(204)

        def close(self) -> None:
            pass

    client = SparrowEngineClient()
    client._client.close()
    stub = StubClient()
    client._client = stub  # type: ignore[assignment]

    client.unload_model("model/with space")

    assert stub.path == "/v1/models/model%2Fwith%20space"


def test_health(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    h = client.health()
    assert h["status"] == "ready"
    assert h["models_loaded"] == 2
    assert h["version"] == "0.1.0"


def test_error_handling(httpserver, client):
    httpserver.expect_request("/v1/models", method="GET").respond_with_json(
        ERROR_RESPONSE, status=404
    )
    with pytest.raises(SparrowEngineClientError) as exc_info:
        client.list_models()
    err = exc_info.value
    assert err.code == "MODEL_NOT_LOADED"
    assert err.status == 404
    assert "foo" in err.message


def test_bbox_to_pixels():
    bbox = BBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
    px = bbox.to_pixels(1920, 1080)
    assert px == (192, 216, 960, 648)


def test_image_file_path(tmp_path):
    img_path = tmp_path / "photo.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    name, data, mime = SparrowEngineClient._image_file(img_path)
    assert name == "photo.jpg"
    assert data == b"\xff\xd8\xff\xe0fake-jpeg"
    assert mime == "image/jpeg"


def test_image_file_bytes():
    raw = b"\xff\xd8\xff\xe0raw-bytes"
    name, data, mime = SparrowEngineClient._image_file(raw)
    assert name == "image.jpg"
    assert data == raw
    assert mime == "image/jpeg"


def test_image_file_fileobj():
    raw = b"\xff\xd8\xff\xe0fileobj-data"
    name, data, mime = SparrowEngineClient._image_file(io.BytesIO(raw))
    assert name == "image.jpg"
    assert data == raw
    assert mime == "image/jpeg"


# ---------------------------------------------------------------------------
# wait_ready / is_ready regression tests (BUG-01 coverage)
# ---------------------------------------------------------------------------


def test_is_ready_true(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    assert client.is_ready() is True


def test_is_ready_false_non_ready_status(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        {"status": "loading", "models_loaded": 0, "pipelines_loaded": 0, "version": "0.1.0"}
    )
    assert client.is_ready() is False


def test_is_ready_false_connection_error():
    """is_ready returns False when server is unreachable."""
    c = SparrowEngineClient(base_url="http://127.0.0.1:1", timeout=0.1)
    try:
        assert c.is_ready() is False
    finally:
        c.close()


def test_wait_ready_success(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        HEALTH_RESPONSE
    )
    client.wait_ready(timeout=2.0, interval=0.1)  # should not raise


def test_wait_ready_timeout(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_json(
        {"status": "loading", "models_loaded": 0, "pipelines_loaded": 0, "version": "0.1.0"}
    )
    with pytest.raises(TimeoutError, match="not ready after"):
        client.wait_ready(timeout=0.3, interval=0.1)


def test_wait_ready_surfaces_malformed_success(httpserver, client):
    httpserver.expect_request("/v1/health", method="GET").respond_with_data(
        "not-json", status=200, content_type="application/json"
    )
    with pytest.raises(ValueError):
        client.wait_ready(timeout=0.3, interval=0.1)


# ---------------------------------------------------------------------------
# Error edge cases
# ---------------------------------------------------------------------------


def test_error_non_json_response(httpserver, client):
    """Non-JSON error (e.g., reverse proxy HTML 502) raises HTTPStatusError, not SparrowEngineClientError."""
    httpserver.expect_request("/v1/models", method="GET").respond_with_data(
        "<html>502 Bad Gateway</html>", status=502
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.list_models()
