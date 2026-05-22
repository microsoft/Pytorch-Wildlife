"""Generate deterministic 10s synthetic WAV for CI audio tests.

Usage:
    uv run --no-project --with numpy tools/generate_synthetic_audio.py

Output:
    test_outputs/fixtures/synthetic_audio_10s.wav
"""
import struct
import numpy as np
from pathlib import Path

SR = 48000
DURATION = 10.0
N_SAMPLES = int(SR * DURATION)


def sine_tone(freq_hz: float, start_s: float, end_s: float,
              amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine tone within a time window, zero elsewhere."""
    t = np.arange(N_SAMPLES) / SR
    mask = (t >= start_s) & (t < end_s)
    signal = np.zeros(N_SAMPLES, dtype=np.float64)
    signal[mask] = amplitude * np.sin(2 * np.pi * freq_hz * t[mask])
    return signal


def generate():
    signal = np.zeros(N_SAMPLES, dtype=np.float64)
    signal += sine_tone(2000.0, 0.0, 3.0, amplitude=0.5)
    signal += sine_tone(4000.0, 2.0, 5.0, amplitude=0.3)
    # 5.0-7.0s: silence (already zero)
    signal += sine_tone(1000.0, 7.0, 9.0, amplitude=0.4)
    signal += sine_tone(6000.0, 7.0, 9.0, amplitude=0.3)
    signal += sine_tone(8000.0, 9.0, 10.0, amplitude=0.4)

    # Clip to [-1, 1] and convert to 16-bit PCM
    signal = np.clip(signal, -1.0, 1.0)
    pcm16 = (signal * 32767).astype(np.int16)

    # Write WAV (no dependency beyond struct)
    out_path = Path(__file__).resolve().parent.parent / "test_outputs" / "fixtures"
    out_path.mkdir(parents=True, exist_ok=True)
    wav_path = out_path / "synthetic_audio_10s.wav"

    with open(wav_path, "wb") as f:
        n_bytes = len(pcm16) * 2
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n_bytes))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", n_bytes))
        f.write(pcm16.tobytes())

    print(f"Wrote {wav_path} ({wav_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate()
