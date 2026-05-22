// Phase 3.8 Step 2 Wave 2 — Per-segment col-major to row-major transpose.
//
// cuBLAS sgemm in mel_gemm.rs writes the mel output column-major
// `[n_mels, total_frames]` (in storage: `mel_col[m + n_mels * t_global]`).
// Per-segment slabs of size `n_mels * frames_per_seg` are byte-contiguous
// (segment s spans `mel_col[s * n_mels * frames_per_seg : ..]`), but
// within each slab the layout is column-major `[n_mels, frames_per_seg]`.
//
// ORT consumes NCHW row-major input `[batch, 1, n_mels, time_steps]`,
// per-segment row-major `[n_mels, frames_per_seg]`. This kernel does an
// out-of-place per-segment transpose so the ORT input buffer is correctly
// laid out:
//
//   in_col [seg, m, t] = in_buf[seg * n_mels * frames_per_seg
//                                + m + n_mels * t]
//   out_row[seg, m, t] = out_buf[seg * n_mels * frames_per_seg
//                                + m * frames_per_seg + t]
//
// Grid: total_segments × BLOCK threads. Block-stride loop covers
// n_mels * frames_per_segment elements per segment.

extern "C" __global__ void transpose_per_segment_kernel(
    const float* __restrict__ in_col_major,
    float*       __restrict__ out_row_major,
    int n_mels,
    int frames_per_seg
) {
    int seg = blockIdx.x;
    int seg_size = n_mels * frames_per_seg;
    const float* in_seg  = in_col_major  + (size_t)seg * seg_size;
    float*       out_seg = out_row_major + (size_t)seg * seg_size;

    int tid = threadIdx.x;
    for (int i = tid; i < seg_size; i += blockDim.x) {
        // i runs over the col-major slab linearly: i = m + n_mels * t.
        int t = i / n_mels;
        int m = i % n_mels;
        // Write to row-major position [m, t] = m * frames_per_seg + t.
        out_seg[m * frames_per_seg + t] = in_seg[i];
    }
}
