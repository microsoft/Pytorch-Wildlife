"""Generate golden reference outputs for audio detection tests.

Implements sparrow-engine-cpu's EXACT mel spectrogram algorithm in Python so the JSON
golden file is independently verifiable. Does NOT use librosa — mel filterbank
and spectrogram are from scratch.

Phase 3.8 Step 2 Wave 0a parameters (post-Slaney corrective fix, 2026-05-04):
  - n_fft=2048 (Phase 3.5 fix; 1024 saturates the model on real audio)
  - Absolute dB: 10*log10(mel), then floor at max - 80
  - **Slaney mel scale** + **Slaney filter normalization** to match
    MD_AudioBirds_V1 training (PW Bioacoustics
    `mel_scale="slaney", norm="slaney"`)
  - Symmetric Hann window

Usage:
    uv run --no-project --with onnxruntime,soundfile,numpy,pillow \\
        tools/generate_audio_golden.py [--synthetic-only] [--first-real]

Output:
    test_outputs/golden/audio_birds_v1/*.json
    test_outputs/golden/audio_birds_v1/*_spectrogram.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # sparrow-engine/
TEST_FILES = REPO_ROOT.parent.parent / "test_files"  # PW_refactor/test_files/
ONNX_PATH = TEST_FILES / "onnx" / "MD_AudioBirds_V1.onnx"
AUDIO_DIR = TEST_FILES / "test_audio"
FIXTURE_PATH = REPO_ROOT / "test_outputs" / "fixtures" / "synthetic_audio_10s.wav"
OUTPUT_DIR = REPO_ROOT / "test_outputs" / "golden" / "audio_birds_v1"

# ---------------------------------------------------------------------------
# Confirmed parameters (R3 — all empirically verified)
# ---------------------------------------------------------------------------
SR = 48000
N_FFT = 2048
HOP = 512
N_MELS = 224
FMIN = 0.0
FMAX = 24000.0
TOP_DB = 80.0
SEGMENT_SEC = 1.0
OVERLAP_SEC = 0.7
STRIDE_SEC = SEGMENT_SEC - OVERLAP_SEC  # 0.3s
SEGMENT_SAMPLES = int(SEGMENT_SEC * SR)  # 48000
STRIDE_SAMPLES = int(STRIDE_SEC * SR)    # 14400
THRESHOLD = 0.5

# Derived: time_steps = 1 + (48000 - 1024) / 512 = 92
TIME_STEPS = 1 + (SEGMENT_SAMPLES - N_FFT) // HOP

# ---------------------------------------------------------------------------
# Symmetric Hann window (NOT periodic — matches Sparrow/MathNet)
# w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
# ---------------------------------------------------------------------------
WINDOW = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))

# ---------------------------------------------------------------------------
# Slaney mel scale (matches torchaudio _hz_to_mel("slaney") and
# librosa.filters.mel(htk=False); Phase 3.8 Step 2 Wave 0a switch from HTK).
# ---------------------------------------------------------------------------
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = (_MIN_LOG_HZ - 0.0) / _F_SP  # = 15.0
_LOGSTEP = np.log(6.4) / 27.0

def hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    out = np.where(
        hz < _MIN_LOG_HZ,
        hz / _F_SP,
        _MIN_LOG_MEL + np.log(np.maximum(hz, 1e-12) / _MIN_LOG_HZ) / _LOGSTEP,
    )
    return out

def mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    out = np.where(
        mel < _MIN_LOG_MEL,
        _F_SP * mel,
        _MIN_LOG_HZ * np.exp((mel - _MIN_LOG_MEL) * _LOGSTEP),
    )
    return out

# ---------------------------------------------------------------------------
# Mel filterbank — Slaney mel scale + Slaney filter normalization.
# ---------------------------------------------------------------------------
def build_mel_filterbank() -> np.ndarray:
    """Build [n_mels, n_fft//2+1] mel filterbank with Slaney normalization.

    Slaney normalization: each filter divided by 2/(hz_centers[i+2] - hz_centers[i])
    so equal-loudness filters get equivalent energy weighting (librosa eq. 6).

    Phase 3.8 Step 2 Wave 0a (F0.8 corrective fix, 2026-05-04): switched
    HTK + area normalization → Slaney + Slaney normalization to match
    MD_AudioBirds_V1 training.
    """
    n_bins = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SR / 2, n_bins)

    mel_min = hz_to_mel(FMIN)
    mel_max = hz_to_mel(FMAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = mel_to_hz(mel_points)

    filters = np.zeros((N_MELS, n_bins))
    for m in range(N_MELS):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k in range(n_bins):
            f = fft_freqs[k]
            if left <= f < center and center > left:
                filters[m, k] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                filters[m, k] = (right - f) / (right - center)

        # Slaney normalization: 2 / Hz bandwidth (left edge to right edge).
        enorm = 2.0 / (hz_points[m + 2] - hz_points[m])
        filters[m] *= enorm

    return filters

# ---------------------------------------------------------------------------
# Mel spectrogram (Sparrow-exact algorithm)
# ---------------------------------------------------------------------------
def compute_mel_spectrogram(samples: np.ndarray,
                            filters: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram for one segment.

    Steps (matching Sparrow AudioProcessingWindow.xaml.cs):
    1. Frame with symmetric Hann window
    2. FFT (unnormalized forward)
    3. Power spectrum: |X|^2
    4. Apply mel filterbank, clamp min 1e-10
    5. 10*log10 dB conversion (ABSOLUTE — ref=1.0, not ref=np.max)
    6. top_db floor: max(S, max(S) - 80)

    Returns: [n_mels, n_frames] float64 array (dB scale)
    """
    n_frames = 1 + max(0, (len(samples) - N_FFT) // HOP)
    n_bins = N_FFT // 2 + 1
    power = np.zeros((n_bins, n_frames))

    for t in range(n_frames):
        frame = samples[t * HOP : t * HOP + N_FFT].copy()
        if len(frame) < N_FFT:
            frame = np.pad(frame, (0, N_FFT - len(frame)))
        frame *= WINDOW
        spectrum = np.fft.rfft(frame)
        power[:, t] = np.real(spectrum) ** 2 + np.imag(spectrum) ** 2

    # Apply mel filterbank
    mel = filters @ power  # [n_mels, n_frames]
    mel = np.maximum(mel, 1e-10)

    # Power to dB — ABSOLUTE (ref=1.0)
    mel_db = 10.0 * np.log10(mel)

    # top_db floor — clip to [max - 80, max]
    max_db = mel_db.max()
    mel_db = np.maximum(mel_db, max_db - TOP_DB)

    return mel_db  # [n_mels, n_frames]

# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------------------------------------------------------
# Load WAV via soundfile
# ---------------------------------------------------------------------------
def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV file. Returns (mono float32 samples in [-1,1], sample_rate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # Take first channel if stereo/multi-channel
    samples = data[:, 0]
    return samples, sr

# ---------------------------------------------------------------------------
# Resample (integer decimation for 192kHz → 48kHz)
# ---------------------------------------------------------------------------
def resample_to_target(samples: np.ndarray, from_sr: int) -> np.ndarray:
    """Resample to SR via integer decimation. Raises if non-integer ratio."""
    if from_sr == SR:
        return samples
    if from_sr % SR == 0:
        factor = from_sr // SR
        resampled = samples[::factor]
        print(f"  Resampled {from_sr} -> {SR} Hz (decimation by {factor})")
        return resampled
    raise ValueError(
        f"Non-integer resampling ratio: {from_sr}/{SR}. "
        f"Only integer decimation supported in golden script."
    )

# ---------------------------------------------------------------------------
# Spectrogram visualization (Pillow — no matplotlib)
# ---------------------------------------------------------------------------
def save_spectrogram_viz(segments: list[dict], wav_name: str,
                         mel_spectrograms: list[np.ndarray],
                         output_dir: Path) -> None:
    """Save a spectrogram PNG with detected segments highlighted.

    Top panel: confidence timeline (bar per segment).
    Bottom panel: concatenated mel spectrogram with detection overlay.
    """
    if not segments:
        return

    n_seg = len(segments)
    bar_h = 100          # height for confidence bar panel
    spec_h = N_MELS      # height for spectrogram panel (224 px)
    gap = 4
    bar_w = max(n_seg * 4, 400)  # at least 400px wide
    total_h = bar_h + gap + spec_h
    total_w = bar_w

    img = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    pixels = img.load()

    # --- Confidence bar panel (top) ---
    seg_w = total_w / n_seg
    for i, seg in enumerate(segments):
        conf = seg["confidence"]
        is_det = conf >= THRESHOLD
        bar_height = int(conf * (bar_h - 2))
        x_start = int(i * seg_w)
        x_end = int((i + 1) * seg_w) - 1

        # Bar color: green if detected, gray if not
        color = (0, 180, 0) if is_det else (100, 100, 100)
        for y in range(bar_h - 1 - bar_height, bar_h - 1):
            for x in range(x_start, min(x_end, total_w)):
                pixels[x, y] = color

        # Threshold line at 0.5
        thresh_y = bar_h - 1 - int(THRESHOLD * (bar_h - 2))
        for x in range(x_start, min(x_end, total_w)):
            pixels[x, thresh_y] = (255, 100, 100)

    # --- Spectrogram panel (bottom) ---
    # Concatenate mel spectrograms into a wide image
    if mel_spectrograms:
        # Use first segment's shape for reference
        frames_per_seg = mel_spectrograms[0].shape[1]
        total_frames = sum(m.shape[1] for m in mel_spectrograms)

        # Build concatenated spectrogram
        concat = np.concatenate(mel_spectrograms, axis=1)  # [n_mels, total_frames]

        # Normalize to [0, 255] for display
        vmin, vmax = concat.min(), concat.max()
        if vmax > vmin:
            norm = (concat - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(concat)

        # Map to viridis-like colormap (simple approximation)
        spec_w_actual = min(total_frames, total_w)
        for y in range(spec_h):
            mel_row = spec_h - 1 - y  # flip vertically (low freq at bottom)
            for x in range(spec_w_actual):
                frame_idx = int(x * total_frames / spec_w_actual)
                if frame_idx >= total_frames:
                    frame_idx = total_frames - 1
                v = norm[mel_row, frame_idx]
                # Simple blue→green→yellow colormap
                r = int(min(1.0, max(0, 2 * v - 0.5)) * 255)
                g = int(min(1.0, max(0, 2 * v if v < 0.5 else 1.0)) * 255)
                b = int(min(1.0, max(0, 1.0 - 2 * v)) * 255)
                pixels[x, bar_h + gap + y] = (r, g, b)

        # Overlay detection boundaries (red vertical lines for non-detected segments)
        for i, seg in enumerate(segments):
            if seg["confidence"] < THRESHOLD:
                x_start = int(i * frames_per_seg * spec_w_actual / total_frames)
                x_end = int((i + 1) * frames_per_seg * spec_w_actual / total_frames)
                for x in range(max(0, x_start), min(x_end, spec_w_actual)):
                    pixels[x, bar_h + gap] = (255, 60, 60)
                    pixels[x, bar_h + gap + spec_h - 1] = (255, 60, 60)
                # Vertical edges
                for y in range(spec_h):
                    if 0 <= x_start < spec_w_actual:
                        pixels[x_start, bar_h + gap + y] = (255, 60, 60)
                    if 0 <= x_end - 1 < spec_w_actual:
                        pixels[x_end - 1, bar_h + gap + y] = (255, 60, 60)

    stem = Path(wav_name).stem
    out_path = output_dir / f"{stem}_spectrogram.png"
    img.save(out_path)
    print(f"  Spectrogram: {out_path}")

# ---------------------------------------------------------------------------
# Process one audio file
# ---------------------------------------------------------------------------
def process_file(wav_path: Path, session: ort.InferenceSession,
                 filters: np.ndarray) -> tuple[dict, list[np.ndarray]]:
    """Load WAV, segment, compute mel spectrogram, run inference.

    Returns: (golden_dict, list_of_mel_spectrograms)
    """
    samples, file_sr = load_wav(wav_path)
    samples = resample_to_target(samples, file_sr)
    duration_s = len(samples) / SR

    # Sliding window — zero-pad last segment if shorter than SEGMENT_SAMPLES
    segments = []
    mel_spectrograms = []
    offset = 0
    idx = 0
    while offset < len(samples):
        remaining = len(samples) - offset
        if remaining < SEGMENT_SAMPLES:
            # Zero-pad the last partial segment (matches Sparrow behavior)
            chunk = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
            chunk[:remaining] = samples[offset:offset + remaining]
        else:
            chunk = samples[offset:offset + SEGMENT_SAMPLES].copy()

        mel_db = compute_mel_spectrogram(chunk.astype(np.float64), filters)
        mel_spectrograms.append(mel_db)

        # Build tensor [1, 1, n_mels, time_steps]
        tensor = mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)

        # Run inference
        logit = float(session.run(None, {"input": tensor})[0][0, 0])
        conf = float(sigmoid(logit))

        start_s = offset / SR
        end_s = min(start_s + SEGMENT_SEC, duration_s)

        segments.append({
            "index": idx,
            "start_s": round(start_s, 6),
            "end_s": round(end_s, 6),
            "logit": round(logit, 6),
            "confidence": round(conf, 6),
        })

        offset += STRIDE_SAMPLES
        idx += 1

        # Stop after processing the last (possibly partial) segment
        if remaining <= SEGMENT_SAMPLES:
            break

    result = {
        "file": wav_path.name,
        "model": "md-audiobirds-v1",
        "sample_rate": SR,
        "duration_s": round(duration_s, 6),
        "n_fft": N_FFT,
        "time_steps_per_segment": TIME_STEPS,
        "segment_duration_s": SEGMENT_SEC,
        "segment_overlap_s": OVERLAP_SEC,
        "num_segments": len(segments),
        "preprocessing": {
            "n_fft": N_FFT,
            "hop_length": HOP,
            "n_mels": N_MELS,
            "fmin": FMIN,
            "fmax": FMAX,
            "power": 2.0,
            "window": "hann_symmetric",
            "mel_scale": "slaney",
            "filter_norm": "slaney",
            "top_db": TOP_DB,
            "db_reference": "absolute (ref=1.0)",
        },
        "segments": segments,
    }

    return result, mel_spectrograms

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filters = build_mel_filterbank()

    print(f"Parameters: n_fft={N_FFT}, hop={HOP}, n_mels={N_MELS}, "
          f"time_steps/segment={TIME_STEPS}")
    print(f"Filterbank shape: {filters.shape}")
    print(f"df = {SR/N_FFT:.4f} Hz per FFT bin")

    session = ort.InferenceSession(str(ONNX_PATH))

    synthetic_only = "--synthetic-only" in sys.argv
    first_real = "--first-real" in sys.argv

    # Always process synthetic fixture
    if FIXTURE_PATH.exists():
        print(f"\nProcessing synthetic fixture: {FIXTURE_PATH.name}")
        result, mels = process_file(FIXTURE_PATH, session, filters)
        out = OUTPUT_DIR / "synthetic_10s_audio.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote {out}")
        print(f"  {result['num_segments']} segments, "
              f"time_steps={result['time_steps_per_segment']}")

        confs = [s["confidence"] for s in result["segments"]]
        detections = sum(1 for c in confs if c >= THRESHOLD)
        print(f"  Detections (>={THRESHOLD}): {detections}/{len(confs)}")
        print(f"  Confidence range: [{min(confs):.4f}, {max(confs):.4f}]")
        print(f"  Mean confidence: {np.mean(confs):.4f}")

        # Save spectrogram visualization
        save_spectrogram_viz(result["segments"], FIXTURE_PATH.name, mels,
                             OUTPUT_DIR)
    else:
        print(f"WARNING: synthetic fixture not found at {FIXTURE_PATH}")
        print("Run tools/generate_synthetic_audio.py first.")

    if synthetic_only:
        return

    # Process real WAV files
    if AUDIO_DIR.exists():
        wavs = sorted(AUDIO_DIR.glob("*.wav"))
        if first_real:
            wavs = wavs[:1]

        for wav in wavs:
            print(f"\nProcessing {wav.name} ({wav.stat().st_size / 1e6:.1f} MB)...")
            t0 = time.time()
            result, mels = process_file(wav, session, filters)
            elapsed = time.time() - t0
            out = OUTPUT_DIR / f"{wav.stem}_audio.json"
            with open(out, "w") as f:
                json.dump(result, f, indent=2)

            confs = [s["confidence"] for s in result["segments"]]
            detections = sum(1 for c in confs if c >= THRESHOLD)
            print(f"  Wrote {out}")
            print(f"  {result['num_segments']} segments in {elapsed:.1f}s")
            print(f"  Detections (>={THRESHOLD}): {detections}/{len(confs)}")
            print(f"  Confidence range: [{min(confs):.4f}, {max(confs):.4f}]")
            print(f"  Mean confidence: {np.mean(confs):.4f}")

            # Save spectrogram visualization (only first 100 segments to keep size reasonable)
            viz_mels = mels[:100] if len(mels) > 100 else mels
            viz_segs = result["segments"][:100] if len(result["segments"]) > 100 else result["segments"]
            save_spectrogram_viz(viz_segs, wav.name, viz_mels, OUTPUT_DIR)

if __name__ == "__main__":
    main()
