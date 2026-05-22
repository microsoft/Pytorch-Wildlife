// CUDA kernel: convolutional bilinear resize + per-channel normalize + NCHW
// transpose. Bit-tight against `fast_image_resize::Resizer` with
// `ResizeAlg::Convolution(FilterType::Bilinear)` — the exact algorithm
// `sparrow-engine-cpu/src/preprocess.rs::resize_simd` uses. Replaces the earlier
// 2-tap texture-style bilinear (which differed at the bilinear-impl LSB
// from the CPU baseline and flipped 1/10 borderline-confidence top-1
// labels in the SpeciesNet parity test).
//
// Algorithm (matches `fast_image_resize-5.5.0/src/convolution/mod.rs::
// precompute_coefficients` + `bilinear_filter`):
//   scale       = in_size  / out_size       (per axis)
//   filter_scale = max(1.0, scale)           (adaptive AA, on for Convolution())
//   filter_radius = 1.0 * filter_scale       (Bilinear support = 1.0)
//   For each out pixel:
//     in_center  = (out + 0.5) * scale
//     center     = in_center - 0.5
//     window     = [floor(in_center - radius), ceil(in_center + radius))
//                  clamped to [0, in_size)
//     For each x in window:
//       t = (x - center) / filter_scale
//       w = max(0, 1 - |t|)          // Bilinear filter (triangle / tent)
//     Normalize weights so sum == 1.0.
//   Output pixel = Σ_x Σ_y w_x[x] * w_y[y] * src[y, x] (separable conv)
//
// Window size:
//   window_size = ceil(filter_radius) * 2 + 1
//   For SpeciesNet 1280×960 → 480×480 (~2.67× downsample): window = 7
//   For 4× downsample: window = 9. We size the on-stack weight arrays
//   to 16 (covers up to ~7× downsample) and clamp via WMAX to keep the
//   kernel branch-free.
//
// Normalization (post-resize):
//   out = (px / 255 - mean[c]) / std[c]
//   Unit (SpeciesNet):     mean = [0, 0, 0], std = [1, 1, 1] — identity.
//   ImageNet (Amazon CTV2): mean = [0.485, 0.456, 0.406],
//                            std = [0.229, 0.224, 0.225].
//
// Unit-identity guard: an explicit `unit_norm` flag (computed in resize.rs
// from the NormalizeStats values) selects a fast path that emits `px/255`
// directly without the subtract+divide step. Two reasons:
//   1. Belt-and-braces against any FMA / contraction-rewrite the NVRTC
//      backend might introduce — IEEE 754 says `x-0=x` and `x/1=x` are
//      exact ops, but the fast path is a true byte-identity skip rather
//      than relying on optimizer fidelity.
//   2. Skips two redundant arithmetic ops in the hot loop for the common
//      Unit case (SpeciesNet today, possibly more classifiers later).
// The result for the Unit case is bit-tight against the pre-2026-05-03
// `/255`-only kernel and against sparrow-engine-cpu's `resize_direct` reference.
//
// Inputs:
//   src       — HWC u8 RGB device buffer of size (src_h * src_w * 3).
//                Always RGB — channel-order swap happens at output write.
//   src_w     — source image width  (pixels).
//   src_h     — source image height (pixels).
//   dst       — NCHW f32 output of size (3 * tgt_h * tgt_w).
//   tgt_w     — target tensor width.
//   tgt_h     — target tensor height.
//   mean_*    — per-channel mean for normalization in [0,1] space.
//   std_*     — per-channel std  for normalization.
//   unit_norm — 1 → Unit fast path (out = px/255, byte-identity skip).
//                0 → general path (out = (px/255 - mean) / std).
//                The wrapper `resize_gpu` sets this when stats == UNIT.
//   bgr       — 0 → RGB plane order. 1 → BGR plane order.

#define WMAX 16

extern "C" __global__ void resize_kernel(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    int unit_norm,
    int bgr
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= tgt_w || oy >= tgt_h) return;

    int plane_size = tgt_w * tgt_h;
    int idx = oy * tgt_w + ox;

    float scale_x = (float)src_w / (float)tgt_w;
    float scale_y = (float)src_h / (float)tgt_h;
    float fscale_x = fmaxf(1.0f, scale_x);
    float fscale_y = fmaxf(1.0f, scale_y);
    float radius_x = fscale_x; // Bilinear support is 1.0
    float radius_y = fscale_y;
    float recip_x = 1.0f / fscale_x;
    float recip_y = 1.0f / fscale_y;

    float in_cx = ((float)ox + 0.5f) * scale_x;
    float in_cy = ((float)oy + 0.5f) * scale_y;
    float center_x = in_cx - 0.5f;
    float center_y = in_cy - 0.5f;

    int xmin = (int)floorf(in_cx - radius_x);
    int xmax = (int)ceilf(in_cx + radius_x);
    int ymin = (int)floorf(in_cy - radius_y);
    int ymax = (int)ceilf(in_cy + radius_y);
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > src_w) xmax = src_w;
    if (ymax > src_h) ymax = src_h;

    int n_x = xmax - xmin;
    int n_y = ymax - ymin;
    if (n_x <= 0 || n_y <= 0) {
        // Degenerate: no source pixels in range. Should never happen for
        // any non-empty image, but emit normalized zero so the tensor is
        // well-formed (zero-pad path). Unit fast-path: 0/255 = 0.
        // General path: (0/255 - mean)/std = -mean/std per channel.
        float pr, pg, pb;
        if (unit_norm) {
            pr = 0.0f;
            pg = 0.0f;
            pb = 0.0f;
        } else {
            pr = (0.0f - mean_r) / std_r;
            pg = (0.0f - mean_g) / std_g;
            pb = (0.0f - mean_b) / std_b;
        }
        if (bgr == 0) {
            dst[0 * plane_size + idx] = pr;
            dst[1 * plane_size + idx] = pg;
            dst[2 * plane_size + idx] = pb;
        } else {
            dst[0 * plane_size + idx] = pb;
            dst[1 * plane_size + idx] = pg;
            dst[2 * plane_size + idx] = pr;
        }
        return;
    }
    if (n_x > WMAX) n_x = WMAX;
    if (n_y > WMAX) n_y = WMAX;

    // X weights — Bilinear (triangle) filter, normalized so they sum to 1.
    float wx[WMAX];
    float wx_sum = 0.0f;
    for (int i = 0; i < n_x; i++) {
        float t = ((float)(xmin + i) - center_x) * recip_x;
        float w = 1.0f - fabsf(t);
        if (w < 0.0f) w = 0.0f;
        wx[i] = w;
        wx_sum += w;
    }
    if (wx_sum > 0.0f) {
        float inv = 1.0f / wx_sum;
        for (int i = 0; i < n_x; i++) wx[i] *= inv;
    }

    // Y weights — same.
    float wy[WMAX];
    float wy_sum = 0.0f;
    for (int j = 0; j < n_y; j++) {
        float t = ((float)(ymin + j) - center_y) * recip_y;
        float w = 1.0f - fabsf(t);
        if (w < 0.0f) w = 0.0f;
        wy[j] = w;
        wy_sum += w;
    }
    if (wy_sum > 0.0f) {
        float inv = 1.0f / wy_sum;
        for (int j = 0; j < n_y; j++) wy[j] *= inv;
    }

    // Separable convolution. n_x * n_y reads per output pixel.
    float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
    int row_stride = src_w * 3;
    for (int j = 0; j < n_y; j++) {
        int sy = ymin + j;
        const unsigned char* row = src + sy * row_stride;
        float wj = wy[j];
        for (int i = 0; i < n_x; i++) {
            int sx = xmin + i;
            const unsigned char* p = row + sx * 3;
            float w = wx[i] * wj;
            r_acc += w * (float)p[0];
            g_acc += w * (float)p[1];
            b_acc += w * (float)p[2];
        }
    }

    // /255 + per-channel (mean, std).
    // Unit fast path (`unit_norm == 1`): out = px / 255 directly. This is
    // a true byte-identity skip vs the pre-2026-05-03 kernel — no FMA /
    // contraction concerns, no reliance on optimizer fidelity. SpeciesNet
    // uses this branch (manifest normalization = "unit").
    // General path: out = (px / 255 - mean) / std. ImageNet-normalized
    // classifiers (Amazon CTV2 today) take this branch.
    float r, g, b;
    if (unit_norm) {
        r = r_acc / 255.0f;
        g = g_acc / 255.0f;
        b = b_acc / 255.0f;
    } else {
        r = (r_acc / 255.0f - mean_r) / std_r;
        g = (g_acc / 255.0f - mean_g) / std_g;
        b = (b_acc / 255.0f - mean_b) / std_b;
    }

    if (bgr == 0) {
        dst[0 * plane_size + idx] = r;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = b;
    } else {
        dst[0 * plane_size + idx] = b;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = r;
    }
}
