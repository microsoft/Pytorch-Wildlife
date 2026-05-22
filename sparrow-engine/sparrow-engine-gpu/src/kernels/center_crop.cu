// CUDA kernel: square center-crop + bilinear resize + /255 normalize +
// NCHW transpose. Classifier preprocess (SpeciesNet, etc.).
//
// Inputs:
//   src       — HWC u8 RGB device buffer of size (src_h * src_w * 3).
//                Always RGB — channel-order swap happens at output write time.
//   src_w     — source image width  (pixels).
//   src_h     — source image height (pixels).
//   dst       — NCHW f32 output of size (3 * tgt_h * tgt_w).
//   tgt_w     — target tensor width  (e.g. 224 for SpeciesNet).
//   tgt_h     — target tensor height (e.g. 224).
//   crop_x    — left x of the square crop in src-space.
//   crop_y    — top  y of the square crop in src-space.
//   crop_size — side length of the square crop in src-space (= min(src_w, src_h)).
//   bgr       — 0 → emit RGB plane order (plane 0 = R).
//                1 → emit BGR plane order (plane 0 = B).
//
// Threading model:
//   1 thread per output canvas pixel; grid sized for (tgt_w, tgt_h).

extern "C" __global__ void center_crop_kernel(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    int crop_x,
    int crop_y,
    int crop_size,
    int bgr
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tgt_w || y >= tgt_h) return;

    int plane_size = tgt_w * tgt_h;
    int idx = y * tgt_w + x;

    // Map (x, y) in target-space to src-space inside the crop window.
    // Half-pixel convention so taps are centered on output pixels.
    float scale_x = (float)crop_size / (float)tgt_w;
    float scale_y = (float)crop_size / (float)tgt_h;
    float sx = ((float)x + 0.5f) * scale_x - 0.5f + (float)crop_x;
    float sy = ((float)y + 0.5f) * scale_y - 0.5f + (float)crop_y;

    if (sx < (float)crop_x) sx = (float)crop_x;
    if (sy < (float)crop_y) sy = (float)crop_y;
    float maxx = (float)(crop_x + crop_size - 1);
    float maxy = (float)(crop_y + crop_size - 1);
    if (sx > maxx) sx = maxx;
    if (sy > maxy) sy = maxy;

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1; if (x1 > src_w - 1) x1 = src_w - 1;
    int y1 = y0 + 1; if (y1 > src_h - 1) y1 = src_h - 1;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    int rs = src_w * 3;
    const unsigned char* p00 = src + y0 * rs + x0 * 3;
    const unsigned char* p01 = src + y0 * rs + x1 * 3;
    const unsigned char* p10 = src + y1 * rs + x0 * 3;
    const unsigned char* p11 = src + y1 * rs + x1 * 3;

    float fr = w00 * (float)p00[0] + w01 * (float)p01[0] + w10 * (float)p10[0] + w11 * (float)p11[0];
    float fg = w00 * (float)p00[1] + w01 * (float)p01[1] + w10 * (float)p10[1] + w11 * (float)p11[1];
    float fb = w00 * (float)p00[2] + w01 * (float)p01[2] + w10 * (float)p10[2] + w11 * (float)p11[2];

    float r = fr / 255.0f;
    float g = fg / 255.0f;
    float b = fb / 255.0f;

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
