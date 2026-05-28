"""Type stubs for sparrow_engine._sparrow_engine_core native module."""
from typing import Callable, Optional

# Per-file progress callback: (index_0_based, total, filename) -> None.
# Invoked after each file's inference attempt resolves. See `detect` etc.
_ProgressCallback = Callable[[int, int, str], None]

class SparrowEngineError(Exception): ...

class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]: ...

class Detection:
    label: str
    label_id: int
    confidence: float
    bbox: BBox

class DetectResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[Detection]
    def __len__(self) -> int: ...

class Classification:
    label: str
    label_id: int
    confidence: float

class ClassifyResult:
    model_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    classifications: list[Classification]
    # First classification (highest-confidence) if any; None on empty result.
    # Convenience for the common `result.top1.label` idiom.
    top1: Optional[Classification]
    def __len__(self) -> int: ...

class PipelineDetection:
    detection: Detection
    classification: Optional[Classification]

class PipelineResult:
    pipeline_id: str
    image_size: tuple[int, int]
    processing_time_ms: float
    detections: list[PipelineDetection]
    def __len__(self) -> int: ...

class AudioClass:
    class_idx: int
    label: Optional[str]
    probability: float

class AudioSegment:
    start_time_s: float
    end_time_s: float
    confidence: float
    classes: list[AudioClass]

class AudioResult:
    model_id: str
    duration_s: float
    sample_rate: int
    window_s: float
    stride_s: float
    processing_time_ms: float
    segments: list[AudioSegment]
    def __len__(self) -> int: ...

class ModelInfo:
    id: str
    model_type: str
    # Manifest [model].subtype: "standard" for normal detectors, "overhead"
    # for top-down / drone-imagery detectors (HerdNet, OWL). Derived from
    # model_type when the native ModelInfo only carries the broader type.
    subtype: str
    default: bool
    version: Optional[str]
    description: Optional[str]
    onnx_sha256: Optional[str]
    onnx_size_bytes: Optional[int]

class PyEngine:
    def __init__(self, device: str, model_dir: str) -> None: ...
    def detect(
        self,
        paths: list[str],
        model: str,
        threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[DetectResult]: ...
    def classify(
        self,
        paths: list[str],
        model: str,
        top_k: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[ClassifyResult]: ...
    def detect_audio(
        self,
        paths: list[str],
        model: str,
        threshold: Optional[float] = None,
        stride_s: Optional[float] = None,
        segment_duration_s: Optional[float] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[AudioResult]: ...
    def pipeline(
        self,
        paths: list[str],
        detector: str,
        classifier: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        progress_callback: Optional[_ProgressCallback] = None,
    ) -> list[PipelineResult]: ...
    def list_models(self) -> list[ModelInfo]: ...
    def model_info(self, model_id: str) -> ModelInfo: ...
    def active_device(self) -> str: ...


def visualize_audio(
    engine: PyEngine,
    items: list[tuple[str, AudioResult]],
    output_dir: Optional[str] = None,
    smooth: bool = False,
    show_windows: bool = False,
    show_ranges: bool = True,
) -> list[list[bytes]]: ...
