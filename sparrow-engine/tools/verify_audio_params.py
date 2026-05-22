"""Empirical verification of n_fft and dB reference choices.

Compares:
1. n_fft=1024 (correct) vs n_fft=2048 (Sparrow default, wrong for this model)
2. Absolute dB (ref=1.0) vs Relative dB (ref=np.max)

Runs on synthetic fixture first, then first real WAV file.

Usage:
    uv run --no-project --with onnxruntime,numpy,scipy \
        tools/verify_audio_params.py
"""
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal.windows import hann as scipy_hann

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_FILES = REPO_ROOT.parent / "test_files"
ONNX_PATH = TEST_FILES / "onnx" / "MD_AudioBirds_V1.onnx"
AUDIO_DIR = TEST_FILES / "test_audio"
FIXTURE_PATH = REPO_ROOT / "test_outputs" / "fixtures" / "synthetic_audio_10s.wav"

SR = 48000
HOP = 512
N_MELS = 224
FMIN = 0.0
FMAX = 24000.0
TOP_DB = 80.0
SEGMENT_SEC = 1.0
OVERLAP_SEC = 0.7
STRIDE_SEC = 0.3
SEGMENT_SAMPLES = int(SEGMENT_SEC * SR)
STRIDE_SAMPLES = int(STRIDE_SEC * SR)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(n_fft: int) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, SR / 2, n_bins)
    df = SR / n_fft
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
        area = np.sum(filters[m] * df)
        if area > 0:
            filters[m] /= area
    return filters


def compute_mel_spectrogram(samples, filters, n_fft, db_mode="absolute"):
    window = scipy_hann(n_fft, sym=True)
    n_frames = 1 + max(0, (len(samples) - n_fft) // HOP)
    n_bins = n_fft // 2 + 1
    power = np.zeros((n_bins, n_frames))
    for t in range(n_frames):
        frame = samples[t * HOP : t * HOP + n_fft].copy()
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame *= window
        spectrum = np.fft.rfft(frame)
        power[:, t] = np.real(spectrum) ** 2 + np.imag(spectrum) ** 2
    mel = filters @ power
    mel = np.maximum(mel, 1e-10)

    if db_mode == "absolute":
        # ref=1.0: 10*log10(mel)
        mel_db = 10.0 * np.log10(mel)
        max_db = mel_db.max()
        mel_db = np.maximum(mel_db, max_db - TOP_DB)
    elif db_mode == "relative":
        # ref=np.max: 10*log10(mel/max(mel)) = 10*log10(mel) - 10*log10(max(mel))
        mel_db = 10.0 * np.log10(mel)
        mel_db -= mel_db.max()  # subtract max → peak = 0
        mel_db = np.maximum(mel_db, -TOP_DB)  # floor at -80
    return mel_db


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_wav_pcm16(path):
    with open(path, "rb") as f:
        f.read(4)  # RIFF
        f.read(4)
        f.read(4)  # WAVE
        sample_rate = None
        raw = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
            elif chunk_id == b"data":
                raw = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if sample_rate != SR and sample_rate % SR == 0:
        factor = sample_rate // SR
        pcm = pcm[::factor]
    return pcm.astype(np.float32) / 32768.0


def run_config(samples, session, n_fft, db_mode, label, n_segments=10):
    """Run first n_segments through model with given config."""
    filters = build_mel_filterbank(n_fft)
    time_steps = 1 + (SEGMENT_SAMPLES - n_fft) // HOP
    results = []
    offset = 0
    for i in range(min(n_segments, 1 + (len(samples) - SEGMENT_SAMPLES) // STRIDE_SAMPLES)):
        chunk = samples[offset:offset + SEGMENT_SAMPLES].astype(np.float64)
        if len(chunk) < SEGMENT_SAMPLES:
            break
        mel_db = compute_mel_spectrogram(chunk, filters, n_fft, db_mode)
        tensor = mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)
        logit = float(session.run(None, {"input": tensor})[0][0, 0])
        conf = sigmoid(logit)
        results.append((i, offset / SR, logit, conf))
        offset += STRIDE_SAMPLES

    detections = sum(1 for _, _, _, c in results if c >= 0.5)
    mean_conf = np.mean([c for _, _, _, c in results])
    print(f"\n  [{label}] n_fft={n_fft}, dB={db_mode}, time_steps={time_steps}")
    print(f"  Detections: {detections}/{len(results)}, mean conf: {mean_conf:.4f}")
    print(f"  {'Seg':>3} {'Start':>6} {'Logit':>10} {'Conf':>8}")
    for idx, start, logit, conf in results:
        marker = "*" if conf >= 0.5 else " "
        print(f"  {idx:3d} {start:6.1f}s {logit:10.4f} {conf:8.4f} {marker}")
    return results


def main():
    session = ort.InferenceSession(str(ONNX_PATH))

    # --- Synthetic fixture ---
    print("=" * 60)
    print("SYNTHETIC FIXTURE (10s, bird-frequency tones + silence)")
    print("=" * 60)
    samples = load_wav_pcm16(FIXTURE_PATH)
    print(f"Loaded: {len(samples)} samples, {len(samples)/SR:.1f}s")

    # Test all 4 combinations on first 20 segments
    n_segs = 22  # enough to cover silence region (segments 17-20)
    run_config(samples, session, 1024, "absolute", "CORRECT", n_segs)
    run_config(samples, session, 2048, "absolute", "WRONG n_fft", n_segs)
    run_config(samples, session, 1024, "relative", "WRONG dB ref", n_segs)
    run_config(samples, session, 2048, "relative", "BOTH WRONG", n_segs)

    # --- First real file ---
    if AUDIO_DIR.exists():
        wavs = sorted(AUDIO_DIR.glob("*.wav"))
        if wavs:
            print("\n" + "=" * 60)
            print(f"REAL FILE: {wavs[0].name}")
            print("=" * 60)
            samples = load_wav_pcm16(wavs[0])
            print(f"Loaded: {len(samples)} samples, {len(samples)/SR:.1f}s")
            run_config(samples, session, 1024, "absolute", "CORRECT", 20)
            run_config(samples, session, 2048, "absolute", "WRONG n_fft", 20)
            run_config(samples, session, 1024, "relative", "WRONG dB ref", 20)


if __name__ == "__main__":
    main()
