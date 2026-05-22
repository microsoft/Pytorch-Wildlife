// Phase 3.8 Step 2 Wave 2 — Window-frame kernel.
//
// Replaces sparrow-engine-cpu's per-frame Hann-multiply loop:
//   for each (frame, i):
//     windowed[frame, i] = samples[frame_start + i] * hann[i]
//
// Producing the row-major `[total_frames * n_fft]` f32 buffer the cuFFT
// R2C plan consumes. Per-frame absolute sample offset is supplied via
// `frame_starts[total_frames]` (i32 per-frame); the GPU never has to
// derive segment / hop arithmetic.
//
// Layouts:
//   samples       : [total_samples] f32
//   frame_starts  : [total_frames] i32 (non-negative; out-of-range reads → 0.0)
//   hann          : [n_fft] f32
//   windowed_out  : [total_frames * n_fft] f32 row-major
//
// Grid: total_frames blocks × BLOCK threads. Block-stride loop covers
// n_fft samples per frame. Out-of-range samples are zero-padded to match
// sparrow-engine-cpu's `padded.resize(segment_samples, 0.0)` tail-handling
// (`sparrow-engine-cpu/src/detect_audio.rs:243-245` post-Wave 0).

extern "C" __global__ void window_frame_kernel(
    const float* __restrict__ samples,
    const int*   __restrict__ frame_starts,
    const float* __restrict__ hann,
    float*       __restrict__ windowed_out,
    int n_fft,
    int total_samples
) {
    int f = blockIdx.x;
    int tid = threadIdx.x;
    int start = frame_starts[f];
    float* out_frame = windowed_out + (size_t)f * n_fft;
    for (int i = tid; i < n_fft; i += blockDim.x) {
        int s = start + i;
        float v = (s >= 0 && s < total_samples) ? samples[s] : 0.0f;
        out_frame[i] = v * hann[i];
    }
}
