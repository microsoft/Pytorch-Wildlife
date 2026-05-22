"""sparrow-engine-client: Python SDK for sparrow-engine-server HTTP API.

Install: pip install httpx  (only runtime dependency)
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx


# ---------------------------------------------------------------------------
# Dataclasses — field names match sparrow-engine-server JSON exactly
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized [0,1] bbox to pixel coordinates."""
        return (
            round(self.x_min * width),
            round(self.y_min * height),
            round(self.x_max * width),
            round(self.y_max * height),
        )


@dataclass
class Detection:
    label: str
    label_id: int
    confidence: float
    bbox: BBox


@dataclass
class Classification:
    label: str
    label_id: int
    confidence: float


@dataclass
class PipelineDetection:
    detection: Detection
    classification: Optional[Classification]


@dataclass
class DetectResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[Detection]


@dataclass
class ClassifyResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    classifications: list[Classification]


@dataclass
class PipelineResult:
    pipeline_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[PipelineDetection]


@dataclass
class AudioSegment:
    start_time_s: float
    end_time_s: float
    confidence: float


@dataclass
class AudioResult:
    model_id: str
    duration_s: float
    sample_rate: int
    processing_time_ms: float
    segments: list[AudioSegment]


@dataclass
class ModelInfo:
    id: str
    model_type: str
    default: bool = False
    version: Optional[str] = None
    description: Optional[str] = None
    onnx_sha256: Optional[str] = None
    onnx_size_bytes: Optional[int] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class SparrowEngineClientError(Exception):
    """Error returned by sparrow-engine-server."""

    def __init__(self, code: str, message: str, status: int) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(f"[{status}] {code}: {message}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


class SparrowEngineClient:
    """Synchronous Python client for sparrow-engine-server."""

    def __init__(
        self, base_url: str = "http://localhost:8080", timeout: float = 60.0
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SparrowEngineClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- Private helpers ---

    def _check(self, resp: httpx.Response) -> dict:
        """Parse response; raise SparrowEngineClientError on 4xx/5xx."""
        if resp.status_code >= 400:
            try:
                err = resp.json()["error"]
                raise SparrowEngineClientError(err["code"], err["message"], resp.status_code)
            except (KeyError, ValueError, TypeError):
                resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _add_store_params(
        params: dict[str, Any], store: bool, halt_on_store_failure: bool
    ) -> None:
        if store:
            params["store"] = True
        if halt_on_store_failure:
            params["halt_on_store_failure"] = True

    @staticmethod
    def _parse_detection(det: dict) -> Detection:
        """Parse a detection JSON dict into a Detection dataclass."""
        return Detection(
            label=det["label"],
            label_id=det["label_id"],
            confidence=det["confidence"],
            bbox=BBox(**det["bbox"]),
        )

    @staticmethod
    def _image_file(image: Any) -> tuple[str, bytes, str]:
        """Normalize image input to (filename, bytes, mimetype)."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            mime = _MIME_MAP.get(p.suffix.lower(), "application/octet-stream")
            return (p.name, p.read_bytes(), mime)
        if isinstance(image, bytes):
            return ("image.jpg", image, "image/jpeg")
        if hasattr(image, "read"):
            return ("image.jpg", image.read(), "image/jpeg")
        # Assume PIL Image
        if hasattr(image, "mode") and image.mode == "RGBA":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return ("image.jpg", buf.getvalue(), "image/jpeg")

    # --- Inference ---

    def detect(
        self,
        image: Any,
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> DetectResult:
        """Run single-image detection."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/detect",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return DetectResult(
            model_id=d["model_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            detections=[self._parse_detection(det) for det in d["detections"]],
        )

    def classify(
        self,
        image: Any,
        model: str,
        top_k: int = 5,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> ClassifyResult:
        """Run single-image classification."""
        params: dict[str, Any] = {"model": model, "top_k": top_k}
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/classify",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return ClassifyResult(
            model_id=d["model_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            classifications=[Classification(**c) for c in d["classifications"]],
        )

    def pipeline(
        self,
        image: Any,
        pipeline: str,
        threshold: Optional[float] = None,
        top_k: int = 5,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> PipelineResult:
        """Run detect+classify pipeline."""
        params: dict[str, Any] = {"pipeline": pipeline, "top_k": top_k}
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        fname, data, mime = self._image_file(image)
        resp = self._client.post(
            "/v1/pipeline",
            params=params,
            files={"image": (fname, data, mime)},
        )
        d = self._check(resp)
        return PipelineResult(
            pipeline_id=d["pipeline_id"],
            image_size=tuple(d["image_size"]),
            processing_time_ms=d["processing_time_ms"],
            detections=[
                PipelineDetection(
                    detection=self._parse_detection(det),
                    classification=(
                        Classification(**det["classification"])
                        if det.get("classification")
                        else None
                    ),
                )
                for det in d["detections"]
            ],
        )

    def detect_audio(
        self,
        audio: Any,
        model: str,
        threshold: Optional[float] = None,
        segment_duration: Optional[float] = None,
        stride: Optional[float] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> AudioResult:
        """Run audio detection."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if segment_duration is not None:
            params["segment_duration"] = segment_duration
        if stride is not None:
            params["stride"] = stride
        self._add_store_params(params, store, halt_on_store_failure)
        if isinstance(audio, (str, Path)):
            p = Path(audio)
            audio_data = p.read_bytes()
            fname = p.name
        elif isinstance(audio, bytes):
            audio_data = audio
            fname = "audio.wav"
        else:
            audio_data = audio.read()
            fname = "audio.wav"
        resp = self._client.post(
            "/v1/audio/detect",
            params=params,
            files={"audio": (fname, audio_data, "audio/wav")},
        )
        d = self._check(resp)
        return AudioResult(
            model_id=d["model_id"],
            duration_s=d["duration_s"],
            sample_rate=d["sample_rate"],
            processing_time_ms=d["processing_time_ms"],
            segments=[AudioSegment(**s) for s in d["segments"]],
        )

    def detect_batch(
        self,
        images: list[Any],
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        store: bool = False,
        halt_on_store_failure: bool = False,
    ) -> list[DetectResult]:
        """Run batch detection on multiple images."""
        params: dict[str, Any] = {"model": model}
        if threshold is not None:
            params["threshold"] = threshold
        if max_detections is not None:
            params["max_detections"] = max_detections
        self._add_store_params(params, store, halt_on_store_failure)
        files = [
            ("images", self._image_file(img)) for img in images
        ]
        resp = self._client.post("/v1/detect/batch", params=params, files=files)
        d = self._check(resp)
        model_id = d["model_id"]
        processing_time_ms = d["processing_time_ms"]
        return [
            DetectResult(
                model_id=model_id,
                image_size=tuple(item["image_size"]),
                processing_time_ms=processing_time_ms,
                detections=[
                    self._parse_detection(det) for det in item["detections"]
                ],
            )
            for item in d["results"]
        ]

    # --- Model management ---

    def list_models(self) -> list[ModelInfo]:
        resp = self._client.get("/v1/models")
        d = self._check(resp)
        return [ModelInfo(**m) for m in d["models"]]

    def load_model(self, model_id: str) -> ModelInfo:
        resp = self._client.post("/v1/models/load", json={"model_id": model_id})
        d = self._check(resp)
        return ModelInfo(**d)

    def unload_model(self, model_id: str) -> None:
        resp = self._client.delete(f"/v1/models/{quote(model_id, safe='')}")
        if resp.status_code == 204:
            return
        self._check(resp)

    # --- Health ---

    def health(self) -> dict:
        resp = self._client.get("/v1/health")
        return self._check(resp)

    def is_ready(self) -> bool:
        try:
            h = self.health()
            return h["status"] in ("ready", "no_models")
        except Exception:
            return False

    def wait_ready(self, timeout: float = 60.0, interval: float = 1.0) -> None:
        """Block until server is ready or timeout.

        Each health probe uses a short per-request timeout capped to
        ``min(interval, remaining)`` so the overall wall-clock time
        stays bounded by *timeout* even when the server is unreachable.
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                req_timeout = min(interval, remaining)
                resp = self._client.get("/v1/health", timeout=req_timeout)
                if resp.status_code < 400:
                    data = resp.json()
                    if data["status"] in ("ready", "no_models"):
                        return
            except httpx.TransportError:
                pass
            remaining = deadline - time.time()
            if remaining > interval:
                time.sleep(interval)
        raise TimeoutError(f"sparrow-engine-server not ready after {timeout}s")
