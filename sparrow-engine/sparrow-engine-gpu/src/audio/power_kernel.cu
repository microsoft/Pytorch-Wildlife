// CUDA kernel: complex → power spectrum, fused `re*re + im*im`.
//
// Replaces sparrow-engine-cpu's per-frame loop:
//   for each (frame, k):
//     power[frame, k] = output[frame, k].re^2 + output[frame, k].im^2
//
// (`sparrow-engine-core/src/preprocess_audio.rs:422-425` post-Wave-0a.)
//
// Layouts (row-major):
//   complex_in : [total_frames, n_freqs] float2   (cuFFT R2C output)
//   power_out  : [total_frames, n_freqs] float    (real power spectrum)
//
// Grid: 1D over `total = total_frames * n_freqs`. Bounds-checked.

extern "C" __global__ void power_kernel(
    const float2* __restrict__ complex_in,
    float* __restrict__ power_out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float2 c = complex_in[idx];
    power_out[idx] = c.x * c.x + c.y * c.y;
}
