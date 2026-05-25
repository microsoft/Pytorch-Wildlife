"""Integration tests for the Python audio visualization wrapper."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("sparrow_engine._sparrow_engine_core")

import sparrow_engine  # noqa: E402
from sparrow_engine import SparrowEngineError  # noqa: E402


AUDIO_MODEL_ID = "md-audiobirds-v1"
AUDIO_MODEL_FILES = ("MD_AudioBirds_V1.onnx", "MD_AudioBirds_V1_fp16.onnx")
PNG_MAGIC = b"\x89PNG"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _audio_fixture() -> Path:
    path = (
        _repo_root()
        / "sparrow-engine-core"
        / "tests"
        / "fixtures"
        / "audio"
        / "short_2s.wav"
    )
    if not path.is_file():
        pytest.skip(f"audio fixture not available: {path}")
    return path


def _audio_model_dir() -> Path:
    model_dir_env = os.environ.get("SPARROW_ENGINE_MODEL_DIR")
    if not model_dir_env:
        pytest.skip("SPARROW_ENGINE_MODEL_DIR is not set")

    model_dir = Path(model_dir_env)
    audio_model_dir = model_dir / AUDIO_MODEL_ID
    if not any((audio_model_dir / name).is_file() for name in AUDIO_MODEL_FILES):
        expected = ", ".join(str(audio_model_dir / name) for name in AUDIO_MODEL_FILES)
        pytest.skip(f"audio model file not available; expected one of: {expected}")
    if not (audio_model_dir / "manifest.toml").is_file():
        pytest.skip(f"audio manifest not available: {audio_model_dir / 'manifest.toml'}")
    return model_dir


def _init_with_model_dir(model_dir: Path) -> None:
    sparrow_engine.init(
        device=os.environ.get("SPARROW_ENGINE_DEVICE", "auto"),
        model_dir=str(model_dir),
    )


@pytest.fixture()
def audio_item() -> tuple[Path, sparrow_engine.AudioResult]:
    audio_path = _audio_fixture()
    model_dir = _audio_model_dir()
    _init_with_model_dir(model_dir)
    results = sparrow_engine.detect_audio(audio_path, model=AUDIO_MODEL_ID, threshold=0.0)
    assert len(results) == 1
    return audio_path, results[0]


def test_visualize_audio_returns_list_of_lists(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
) -> None:
    result = sparrow_engine.visualize_audio([audio_item])

    assert len(result) == 1
    assert len(result[0]) >= 3
    assert all(isinstance(layer, bytes) for layer in result[0])
    assert all(layer.startswith(PNG_MAGIC) for layer in result[0])


def test_visualize_audio_writes_files_when_output_dir_set(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
    tmp_path: Path,
) -> None:
    audio_path, _ = audio_item

    result = sparrow_engine.visualize_audio([audio_item], output_dir=tmp_path)

    assert len(result) == 1
    produced = {p.name for p in tmp_path.rglob("*.png")}
    expected = {
        f"{audio_path.stem}_01_spec.png",
        f"{audio_path.stem}_02_segments.png",
        f"{audio_path.stem}_03_heatmap.png",
        f"{audio_path.stem}_04_full.png",
    }
    assert expected.issubset(produced)


def test_visualize_audio_creates_missing_output_dir(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "viz"

    assert not target.exists()
    result = sparrow_engine.visualize_audio([audio_item], output_dir=target)

    assert len(result) == 1
    assert target.is_dir()
    assert any(target.rglob("*.png"))


def test_visualize_audio_show_windows_adds_layer(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
) -> None:
    default = sparrow_engine.visualize_audio([audio_item])
    with_windows = sparrow_engine.visualize_audio([audio_item], show_windows=True)

    assert len(with_windows) == len(default) == 1
    assert len(with_windows[0]) == len(default[0]) + 1


def test_visualize_audio_show_windows_writes_extra_file(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
    tmp_path: Path,
) -> None:
    audio_path, _ = audio_item

    sparrow_engine.visualize_audio([audio_item], output_dir=tmp_path, show_windows=True)

    produced = {p.name for p in tmp_path.rglob("*.png")}
    assert f"{audio_path.stem}_02_segments_windows.png" in produced


def test_visualize_audio_show_ranges_false_drops_full_layer(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
    tmp_path: Path,
) -> None:
    audio_path, _ = audio_item
    target = tmp_path / "ranges_off"

    result = sparrow_engine.visualize_audio(
        [audio_item], output_dir=target, show_ranges=False
    )

    assert len(result) == 1
    assert len(result[0]) == 3
    produced = {p.name for p in target.rglob("*.png")}
    assert f"{audio_path.stem}_04_full.png" not in produced
    assert {
        f"{audio_path.stem}_01_spec.png",
        f"{audio_path.stem}_02_segments.png",
        f"{audio_path.stem}_03_heatmap.png",
    }.issubset(produced)


def test_visualize_audio_smooth_changes_heatmap_layer(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
) -> None:
    default = sparrow_engine.visualize_audio([audio_item])
    smooth = sparrow_engine.visualize_audio([audio_item], smooth=True)

    assert len(default) == len(smooth) == 1
    assert len(default[0]) == len(smooth[0])
    assert default[0][0] == smooth[0][0]
    assert default[0][1] == smooth[0][1]
    assert default[0][2] != smooth[0][2]
    if len(default[0]) > 3:
        assert default[0][3] != smooth[0][3]


def test_visualize_audio_rejects_wrong_result_type(
    audio_item: tuple[Path, sparrow_engine.AudioResult],
) -> None:
    audio_path, _ = audio_item

    with pytest.raises(SparrowEngineError, match="visualization"):
        sparrow_engine.visualize_audio(
            [(audio_path, object())]  # type: ignore[list-item]
        )


def test_visualize_audio_empty_batch_does_not_initialize_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sparrow_engine, "_engine", None)
    monkeypatch.delenv("SPARROW_ENGINE_MODEL_DIR", raising=False)

    assert sparrow_engine.visualize_audio([]) == []
    assert sparrow_engine._engine is None
