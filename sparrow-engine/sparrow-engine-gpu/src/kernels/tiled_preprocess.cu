// CUDA kernel: per-tile crop + zero-pad + per-channel normalize + NCHW transpose.
//
// Phase 3.8 Step 1 Wave 4 follow-up — closes the GPU-pipeline gap left by the
// initial Wave 4 MVP (which did CPU preprocess + GPU inference). This kernel
// runs the per-tile preprocess on GPU directly out of an nvjpeg-decoded
// HWC u8 RGB buffer, so the only HtoD remaining is ORT's implicit re-upload
// of our FP32 NCHW tile (a known follow-up — see models/yolo.rs IoBinding
// commentary).
//
// Compiled at runtime via cudarc::nvrtc::Ptx (no nvcc at build time).
//
// # Inputs
//   src       — full image HWC u8 RGB device buffer, length = img_h * img_w * 3.
//                Always RGB-ordered; channel-order swap happens at output write
//                time, mirroring letterbox.cu's convention.
//   img_w     — full image width  (pixels).
//   img_h     — full image height (pixels).
//   tile_x    — top-left X of the crop in source pixels.
//   tile_y    — top-left Y of the crop in source pixels.
//   crop_w    — crop width  in pixels (≤ tgt_w; smaller for right-edge tiles).
//   crop_h    — crop height in pixels (≤ tgt_h; smaller for bottom-edge tiles).
//   dst       — NCHW f32 output of size (3 * tgt_h * tgt_w). Caller
//                pre-zeroes; the kernel always writes every output slot.
//   tgt_w  — model input width  (e.g. 512 for HerdNet / OWL-T).
//   tgt_h  — model input height (e.g. 512 for HerdNet / OWL-T).
//   mean_*    — per-channel mean for normalization in [0,1] space.
//   std_*     — per-channel std  for normalization.
//                For Unit normalization (OWL-T):  mean = [0,0,0], std = [1,1,1].
//                For ImageNet     (HerdNet):      mean = [0.485, 0.456, 0.406],
//                                                  std  = [0.229, 0.224, 0.225].
//   bgr       — 0 → emit RGB plane order (plane 0 = R).
//                1 → emit BGR plane order (plane 0 = B).
//                Mirrors the ChannelOrder::{Rgb,Bgr} dispatch in letterbox.cu.
//
// # Edge-tile behavior (zero-pad, NOT stretch)
//
// For (x, y) inside the crop region (x < crop_w && y < crop_h): bilinear-style
// 1-tap sampling at (tile_x + x, tile_y + y) in src.
//
// For (x, y) outside the crop region (x >= crop_w || y >= crop_h): emit the
// pixel as if it were 0 (black), then push it through the same normalize
// formula so the output matches the CPU reference exactly:
//
//   out_pad_r = (0 / 255 - mean_r) / std_r   (= -2.118 for ImageNet R, 0 for Unit)
//   out_pad_g = (0 / 255 - mean_g) / std_g   (= -2.036 for ImageNet G, 0 for Unit)
//   out_pad_b = (0 / 255 - mean_b) / std_b   (= -1.804 for ImageNet B, 0 for Unit)
//
// This is byte-equivalent to the CPU path which builds a black 512×512
// `RgbImage::new(...)` (R=G=B=0) for the edge case, copy_from's the small
// crop into the top-left, and then runs the same normalize_pixel formula.
//
// # Why no resize?
//
// HerdNet + OWL-T manifests both have `tile_size == input_size == [512, 512]`.
// Cropping a 512×512 patch from the source then sampling it 1-to-1 to a 512×512
// model input is identity — there is no resize. If a future tiled manifest
// declares `tile_size != input_size`, this kernel must be extended (the sparrow-engine-cpu
// reference path doesn't currently handle that case either; both crash early at
// the manifest validator).
//
// # Threading model
//
// 1 thread per output canvas pixel; grid sized for (tgt_w, tgt_h).

extern "C" __global__ void tiled_preprocess_kernel(
    const unsigned char* __restrict__ src,
    int img_w,
    int img_h,
    int tile_x,
    int tile_y,
    int crop_w,
    int crop_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    int bgr
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tgt_w || y >= tgt_h) return;

    int plane_size = tgt_w * tgt_h;
    int idx = y * tgt_w + x;

    float r, g, b;
    if (x < crop_w && y < crop_h) {
        int sx = tile_x + x;
        int sy = tile_y + y;
        // Defensive bounds clamp. Caller already ensures
        // tile_x + crop_w <= img_w and tile_y + crop_h <= img_h, but float
        // rounding upstream is not worth chasing — clamp here.
        if (sx < 0) sx = 0;
        if (sy < 0) sy = 0;
        if (sx > img_w - 1) sx = img_w - 1;
        if (sy > img_h - 1) sy = img_h - 1;

        const unsigned char* p = src + sy * img_w * 3 + sx * 3;
        float pr = (float)p[0] / 255.0f;
        float pg = (float)p[1] / 255.0f;
        float pb = (float)p[2] / 255.0f;
        r = (pr - mean_r) / std_r;
        g = (pg - mean_g) / std_g;
        b = (pb - mean_b) / std_b;
    } else {
        // Zero-pad. Push 0/255 = 0 through the same normalize formula so the
        // output is byte-equivalent to the CPU build_nchw_tensor path.
        r = (0.0f - mean_r) / std_r;
        g = (0.0f - mean_g) / std_g;
        b = (0.0f - mean_b) / std_b;
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
