// CUDA kernel pair: power → dB with per-segment max-relative floor.
//
// Mirrors `sparrow-engine-core/src/preprocess_audio.rs::power_to_db`
// (`sparrow-engine-core/src/preprocess_audio.rs:570-582` post-Wave-0a):
//
//   for each x in mel:
//     x = 10 * log10(max(x, 1e-10))     # log_kernel
//   max_db = max(x for x in mel)         # CPU does this serially
//   floor  = max_db - top_db
//   for each x in mel:
//     x = max(x, floor)                  # clamp_kernel
//
// `sparrow-engine-cpu` invokes this PER SEGMENT of mel output (each `[n_mels=224,
// n_frames=90]` block): the per-segment max determines the floor for that
// segment only. We replicate that semantic by partitioning the
// `[n_mels, total_frames]` GPU mel tensor into per-segment slices of
// `n_mels * frames_per_segment` and running the reduction + clamp per
// slice.
//
// For the manifest's 1 s segments at 48 kHz with hop 512 and n_fft 2048,
// frames_per_segment = 90 and n_mels = 224 → 20,160 elements per segment.
//
// Layouts:
//   mel : [n_mels, total_frames] column-major f32 (output of cuBLAS sgemm).
//         `mel[m, t] = mel_raw[m + n_mels * t]`. The reduction is over the
//         per-segment slab `mel_raw[seg_start * n_mels .. (seg_start +
//         frames_per_segment) * n_mels]`. Both axes flatten to a contiguous
//         range of `n_mels * frames_per_segment` floats.
//
// Kernel structure: launch one block per segment with block_dim = 256
// threads. Block-stride loop over the segment's elements:
//   - Apply `10 * log10(max(x, 1e-10))` in place.
//   - Block-level max reduction → shared-memory `seg_max`.
// Sync + write `floor = seg_max - top_db` into a per-segment float.
// Second pass (block-stride loop) clamps `x = max(x, floor)`.

#define EPS 1e-10f
#define BLOCK 256

extern "C" __global__ void power_to_db_kernel(
    float* __restrict__ mel,
    int n_mels,
    int frames_per_segment,
    float top_db
) {
    int seg = blockIdx.x;
    int tid = threadIdx.x;
    int seg_size = n_mels * frames_per_segment;
    int seg_off  = seg * seg_size;
    float* mel_seg = mel + seg_off;

    // Pass 1: in-place dB conversion + per-thread local max accumulation.
    //
    // Sentinel NEG_LARGE = -1.0e30f instead of -INFINITY because NVRTC
    // (used for runtime PTX compilation here) does not preprocess
    // <math.h> by default — the `INFINITY` macro is undefined. -1.0e30f
    // is safely below any possible post-log10 mel-dB value: the input
    // domain is post-GEMM mel power (max ~4 in unit-amplitude tone tests,
    // ~1e6 in real audio); 10*log10 of 1e6 is +60 dB, 10*log10 of 1e-10
    // (the EPS floor) is -100 dB, both well within ±1e30.
    __shared__ float s_max[BLOCK];
    const float NEG_LARGE = -1.0e30f;
    float local_max = NEG_LARGE;
    for (int i = tid; i < seg_size; i += BLOCK) {
        float v = mel_seg[i];
        if (v < EPS) v = EPS;
        v = 10.0f * log10f(v);
        mel_seg[i] = v;
        if (v > local_max) local_max = v;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Block-level max reduction (power-of-2 BLOCK).
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = s_max[tid];
            float b = s_max[tid + s];
            s_max[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    float seg_max = s_max[0];
    float floor_db = seg_max - top_db;

    // Pass 2: clamp.
    for (int i = tid; i < seg_size; i += BLOCK) {
        float v = mel_seg[i];
        if (v < floor_db) v = floor_db;
        mel_seg[i] = v;
    }
}
