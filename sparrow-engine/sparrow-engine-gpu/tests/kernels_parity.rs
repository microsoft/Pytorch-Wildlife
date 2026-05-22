//! Kernel parity tests: GPU letterbox + center-crop vs a CPU 2-tap
//! bilinear reference that mirrors the kernels' algorithm exactly.
//!
//! Why mirror the algorithm rather than compare against
//! `fast_image_resize::Resizer` (Bilinear)?
//!
//! - `fast_image_resize::Bilinear` is *convolutional* bilinear (multi-tap
//!   anti-aliased), and for >2× downsampling it samples 4–10+ source
//!   pixels per output pixel. That's a different algorithm from the
//!   GPU kernels' 2-tap bilinear (texture-style). Comparing them at
//!   ε=1e-3 in normalized space fails for downsampling-heavy paths
//!   (e.g., 1080×1080 → 224×224 SpeciesNet center crop = 4.82×).
//! - The Step 1 design's Gate 1 ("ε=1e-3 on bbox coords on 100-image
//!   corpus") evaluates parity at the POST-postprocess detection level,
//!   not at raw tensor level. Wave 2/3 owns the end-to-end model parity
//!   check (Gate 2 / Gate 3); Wave 1's job is verifying the kernel
//!   implements 2-tap bilinear correctly.
//! - Mirroring the algorithm exactly turns this into a unit test: it
//!   catches kernel bugs (off-by-one, wrong plane order, wrong
//!   normalization) without entangling algorithm choice.
//!
//! For each kernel an "informational" comparison against
//! `fast_image_resize` is also printed to stderr — useful when bench
//! parity diverges in Wave 2/3 — but does not assert.

use std::path::Path;

use sparrow_engine::decode::GpuImage;
use sparrow_engine::kernels::{
    center_crop::{center_crop_gpu, CenterCropKernel},
    letterbox::{letterbox_gpu, LetterboxKernel},
};
use sparrow_engine_types::manifest::ChannelOrder;
use cudarc::driver::CudaContext;
use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{FilterType as FirFilter, PixelType, ResizeAlg, ResizeOptions, Resizer};

const CORPUS: &str = "/home/miao/repos/PW_refactor/test_files/test_cameratrap";

// ε at the kernel level: tight, because the CPU reference mirrors the
// kernel's 2-tap bilinear math exactly. Any divergence > 1e-4 mean-abs
// in normalized [0,1] space indicates a kernel implementation bug
// (memory access, plane order, normalization, or sub-pixel offset).
//
// `final_design §8 Gate 1`'s ε=1e-3 lives at the bbox-coord level
// post-postprocess (Wave 2/3 territory) where rounding compounds across
// resize → letterbox → normalize → ONNX → postprocess; Wave 1's
// kernel-only test runs against a Rust reference that mirrors the GPU
// 2-tap bilinear math exactly, so the achievable ε floor is tighter.
const EPSILON_MEAN: f32 = 1e-4;
// Allow individual pixels to drift up to ~1 LSB in u8 space (= 1/255 ≈ 4e-3)
// to absorb f32 rounding differences between Rust and PTX.
const EPSILON_MAX: f32 = 5e-3;

fn gpu_tests_enabled() -> bool {
    !matches!(std::env::var("SPARROW_ENGINE_GPU_TESTS").as_deref(), Ok("0"))
}

fn corpus_jpegs() -> Vec<std::path::PathBuf> {
    let dir = Path::new(CORPUS);
    if !dir.exists() {
        return Vec::new();
    }
    let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry: std::io::Result<std::fs::DirEntry>| {
            entry.ok().map(|entry: std::fs::DirEntry| entry.path())
        })
        .filter(|p: &std::path::PathBuf| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg"))
        })
        .collect();
    v.sort();
    v
}

// ---------------------------------------------------------------------
// CPU 2-tap bilinear references (mirror the .cu kernel math exactly).
// ---------------------------------------------------------------------

#[inline]
fn bilinear_sample(src: &image::RgbImage, sx: f32, sy: f32) -> [f32; 3] {
    let (sw, sh) = (src.width() as i32, src.height() as i32);
    // Match the kernel's clamp.
    let sx = sx.clamp(0.0, (sw - 1) as f32);
    let sy = sy.clamp(0.0, (sh - 1) as f32);
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;
    let x1 = (x0 + 1).min(sw - 1);
    let y1 = (y0 + 1).min(sh - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    let p00 = src.get_pixel(x0 as u32, y0 as u32);
    let p01 = src.get_pixel(x1 as u32, y0 as u32);
    let p10 = src.get_pixel(x0 as u32, y1 as u32);
    let p11 = src.get_pixel(x1 as u32, y1 as u32);
    let s = |i: usize| -> f32 {
        w00 * p00[i] as f32 + w01 * p01[i] as f32 + w10 * p10[i] as f32 + w11 * p11[i] as f32
    };
    [s(0), s(1), s(2)]
}

fn cpu_letterbox_2tap(
    img: &image::RgbImage,
    tgt_w: u32,
    tgt_h: u32,
    pad_value: f32,
    bgr: bool,
) -> Vec<f32> {
    let (img_w, img_h) = (img.width() as f32, img.height() as f32);
    let scale = (tgt_w as f32 / img_w).min(tgt_h as f32 / img_h);
    let new_w = (img_w * scale).round().max(1.0).min(tgt_w as f32) as u32;
    let new_h = (img_h * scale).round().max(1.0).min(tgt_h as f32) as u32;
    let pad_x_left = ((tgt_w as f32 - new_w as f32) / 2.0).floor() as i32;
    let pad_y_top = ((tgt_h as f32 - new_h as f32) / 2.0).ceil() as i32;
    let plane = (tgt_w * tgt_h) as usize;
    let mut nchw = vec![pad_value; 3 * plane];
    for y in 0..tgt_h as i32 {
        for x in 0..tgt_w as i32 {
            let xi = x - pad_x_left;
            let yi = y - pad_y_top;
            if xi < 0 || yi < 0 || xi >= new_w as i32 || yi >= new_h as i32 {
                continue; // pad
            }
            // Identical formula to letterbox.cu.
            let sx = (xi as f32 + 0.5) / scale - 0.5;
            let sy = (yi as f32 + 0.5) / scale - 0.5;
            let p = bilinear_sample(img, sx, sy);
            let idx = (y as u32 * tgt_w + x as u32) as usize;
            let r = p[0] / 255.0;
            let g = p[1] / 255.0;
            let b = p[2] / 255.0;
            if bgr {
                nchw[idx] = b;
                nchw[plane + idx] = g;
                nchw[2 * plane + idx] = r;
            } else {
                nchw[idx] = r;
                nchw[plane + idx] = g;
                nchw[2 * plane + idx] = b;
            }
        }
    }
    nchw
}

fn cpu_center_crop_2tap(img: &image::RgbImage, tgt_w: u32, tgt_h: u32, bgr: bool) -> Vec<f32> {
    let crop_size = img.width().min(img.height());
    let crop_x = ((img.width() - crop_size) / 2) as f32;
    let crop_y = ((img.height() - crop_size) / 2) as f32;
    let scale_x = crop_size as f32 / tgt_w as f32;
    let scale_y = crop_size as f32 / tgt_h as f32;
    let plane = (tgt_w * tgt_h) as usize;
    let mut nchw = vec![0.0f32; 3 * plane];
    let max_x = crop_x + crop_size as f32 - 1.0;
    let max_y = crop_y + crop_size as f32 - 1.0;
    for y in 0..tgt_h {
        for x in 0..tgt_w {
            let mut sx = (x as f32 + 0.5) * scale_x - 0.5 + crop_x;
            let mut sy = (y as f32 + 0.5) * scale_y - 0.5 + crop_y;
            if sx < crop_x {
                sx = crop_x;
            }
            if sy < crop_y {
                sy = crop_y;
            }
            if sx > max_x {
                sx = max_x;
            }
            if sy > max_y {
                sy = max_y;
            }
            let p = bilinear_sample(img, sx, sy);
            let idx = (y * tgt_w + x) as usize;
            let r = p[0] / 255.0;
            let g = p[1] / 255.0;
            let b = p[2] / 255.0;
            if bgr {
                nchw[idx] = b;
                nchw[plane + idx] = g;
                nchw[2 * plane + idx] = r;
            } else {
                nchw[idx] = r;
                nchw[plane + idx] = g;
                nchw[2 * plane + idx] = b;
            }
        }
    }
    nchw
}

// ---------------------------------------------------------------------
// Informational fast_image_resize comparison (does NOT assert).
// Useful when Wave 2/3 detection-level parity diverges, to localise
// "is the gap from kernel algorithm or model code?"
// ---------------------------------------------------------------------

fn fir_resize_informational(img: &image::RgbImage, w: u32, h: u32) -> image::RgbImage {
    let src = FirImage::from_vec_u8(
        img.width(),
        img.height(),
        img.as_raw().to_vec(),
        PixelType::U8x3,
    )
    .expect("FirImage::from_vec_u8");
    let mut dst = FirImage::new(w, h, PixelType::U8x3);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FirFilter::Bilinear));
    let mut r = Resizer::new();
    r.resize(&src, &mut dst, &opts).expect("FIR resize");
    image::RgbImage::from_raw(w, h, dst.into_vec()).expect("RgbImage::from_raw")
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        s += (*x as f64 - *y as f64).abs();
    }
    (s / a.len() as f64) as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn upload_cpu_image(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    img: &image::RgbImage,
) -> GpuImage {
    let buf = img.as_raw().to_vec();
    let dev = stream.clone_htod(buf.as_slice()).expect("clone_htod");
    GpuImage {
        data: dev,
        width: img.width(),
        height: img.height(),
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[test]
fn letterbox_gpu_vs_cpu_2tap_parity() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping letterbox parity test");
        return;
    }
    let jpegs = corpus_jpegs();
    if jpegs.is_empty() {
        eprintln!("Corpus {CORPUS} missing → skipping letterbox parity test");
        return;
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let kernel = LetterboxKernel::new(&ctx).expect("compile letterbox kernel");

    const TGT: u32 = 1280; // MDv6 input size
    const PAD: f32 = 114.0 / 255.0;

    let n_check = jpegs.len().min(5);
    for path in jpegs.iter().take(n_check) {
        let bytes = std::fs::read(path).expect("read jpeg");
        let cpu_img = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .expect("guess format")
            .decode()
            .expect("decode")
            .to_rgb8();
        let gpu_img = upload_cpu_image(&stream, &cpu_img);
        let (gpu_dst, _meta) =
            letterbox_gpu(&stream, &kernel, &gpu_img, TGT, TGT, PAD, ChannelOrder::Rgb)
                .expect("letterbox_gpu");
        let gpu_buf: Vec<f32> = stream.clone_dtoh(&gpu_dst).expect("clone_dtoh");
        stream.synchronize().expect("stream synchronize");
        let cpu_buf = cpu_letterbox_2tap(&cpu_img, TGT, TGT, PAD, false);

        let m = mean_abs(&gpu_buf, &cpu_buf);
        let x = max_abs(&gpu_buf, &cpu_buf);
        eprintln!("letterbox 2-tap parity {path:?}: mean_abs={m:.6}, max_abs={x:.6}");
        assert!(
            m < EPSILON_MEAN,
            "letterbox mean-abs {m} > ε_mean={EPSILON_MEAN} for {path:?}"
        );
        assert!(
            x < EPSILON_MAX,
            "letterbox max-abs {x} > ε_max={EPSILON_MAX} for {path:?}"
        );

        // Informational: how does the kernel compare to fast_image_resize
        // convolutional Bilinear? Helpful diagnostic when downstream
        // model parity diverges. Does NOT assert.
        let fir_resized = fir_resize_informational(&cpu_img, TGT, TGT);
        let mut fir_nchw = vec![PAD; (3 * TGT * TGT) as usize];
        // For an exact-match letterbox vs fir, we'd need to align tile
        // sizes — keep this purely informational by computing only on
        // the unpadded bbox.
        let plane = (TGT * TGT) as usize;
        let (img_w, img_h) = (cpu_img.width() as f32, cpu_img.height() as f32);
        let scale = (TGT as f32 / img_w).min(TGT as f32 / img_h);
        let new_w = (img_w * scale).round().max(1.0).min(TGT as f32) as u32;
        let new_h = (img_h * scale).round().max(1.0).min(TGT as f32) as u32;
        let pad_x = ((TGT - new_w) / 2) as usize;
        let pad_y = ((TGT - new_h) / 2) as usize;
        let fir_resized_inner = fir_resize_informational(&cpu_img, new_w, new_h);
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let p = fir_resized_inner.get_pixel(x as u32, y as u32);
                let idx = (pad_y + y) * TGT as usize + (pad_x + x);
                fir_nchw[idx] = p[0] as f32 / 255.0;
                fir_nchw[plane + idx] = p[1] as f32 / 255.0;
                fir_nchw[2 * plane + idx] = p[2] as f32 / 255.0;
            }
        }
        let info_m = mean_abs(&gpu_buf, &fir_nchw);
        eprintln!("  vs fast_image_resize Bilinear: mean_abs={info_m:.6} (info only)");
        let _ = fir_resized;
    }
}

#[test]
fn letterbox_gpu_metadata_uses_fractional_padding() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping letterbox metadata parity test");
        return;
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let kernel = LetterboxKernel::new(&ctx).expect("compile letterbox kernel");
    let cpu_img = image::RgbImage::from_pixel(5, 3, image::Rgb([127, 64, 32]));
    let gpu_img = upload_cpu_image(&stream, &cpu_img);

    let (_gpu_dst, meta) = letterbox_gpu(
        &stream,
        &kernel,
        &gpu_img,
        8,
        8,
        114.0 / 255.0,
        ChannelOrder::Rgb,
    )
    .expect("letterbox_gpu");

    // 5x3 -> 8x5 under an 8x8 target, leaving 3 vertical pad pixels.
    // CUDA placement still uses ceil(top)=2 for PW compatibility, but
    // postprocess metadata must mirror CPU/shared fractional padding (1.5).
    assert!((meta.pad_x - 0.0).abs() < f32::EPSILON);
    assert!(
        (meta.pad_y - 1.5).abs() < f32::EPSILON,
        "odd vertical padding must be stored fractionally for postprocess parity, got {}",
        meta.pad_y
    );
    assert!(
        (meta.scale - 1.6).abs() < 1e-6,
        "unexpected scale {}",
        meta.scale
    );
}

#[test]
fn center_crop_gpu_vs_cpu_2tap_parity() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping center_crop parity test");
        return;
    }
    let jpegs = corpus_jpegs();
    if jpegs.is_empty() {
        eprintln!("Corpus {CORPUS} missing → skipping center_crop parity test");
        return;
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let kernel = CenterCropKernel::new(&ctx).expect("compile center_crop kernel");

    const TGT: u32 = 224;

    let n_check = jpegs.len().min(5);
    for path in jpegs.iter().take(n_check) {
        let bytes = std::fs::read(path).expect("read jpeg");
        let cpu_img = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .expect("guess format")
            .decode()
            .expect("decode")
            .to_rgb8();
        let gpu_img = upload_cpu_image(&stream, &cpu_img);
        let gpu_dst = center_crop_gpu(&stream, &kernel, &gpu_img, TGT, TGT, ChannelOrder::Rgb)
            .expect("center_crop_gpu");
        let gpu_buf: Vec<f32> = stream.clone_dtoh(&gpu_dst).expect("clone_dtoh");
        stream.synchronize().expect("stream synchronize");
        let cpu_buf = cpu_center_crop_2tap(&cpu_img, TGT, TGT, false);

        let m = mean_abs(&gpu_buf, &cpu_buf);
        let x = max_abs(&gpu_buf, &cpu_buf);
        eprintln!("center_crop 2-tap parity {path:?}: mean_abs={m:.6}, max_abs={x:.6}");
        assert!(
            m < EPSILON_MEAN,
            "center_crop mean-abs {m} > ε_mean={EPSILON_MEAN} for {path:?}"
        );
        assert!(
            x < EPSILON_MAX,
            "center_crop max-abs {x} > ε_max={EPSILON_MAX} for {path:?}"
        );
    }
}

#[test]
fn channel_order_bgr_swaps_planes() {
    // Tight unit test: verify the kernel correctly emits BGR plane order
    // when channel_order=Bgr. Constructs a 1×1 source pixel with known
    // (R,G,B), runs through letterbox at tgt=1×1 (identity scale), and
    // asserts plane0=B, plane1=G, plane2=R after /255.
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping channel order test");
        return;
    }
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let kernel = LetterboxKernel::new(&ctx).expect("compile letterbox kernel");

    let img = image::RgbImage::from_pixel(1, 1, image::Rgb([200u8, 100u8, 50u8]));
    let gpu_img = upload_cpu_image(&stream, &img);

    let (gpu_dst, _meta) = letterbox_gpu(&stream, &kernel, &gpu_img, 1, 1, 0.0, ChannelOrder::Bgr)
        .expect("letterbox_gpu");
    let gpu_buf: Vec<f32> = stream.clone_dtoh(&gpu_dst).expect("clone_dtoh");
    stream.synchronize().expect("stream synchronize");

    assert_eq!(gpu_buf.len(), 3);
    assert!((gpu_buf[0] - 50.0 / 255.0).abs() < 1e-4, "plane0 != B");
    assert!((gpu_buf[1] - 100.0 / 255.0).abs() < 1e-4, "plane1 != G");
    assert!((gpu_buf[2] - 200.0 / 255.0).abs() < 1e-4, "plane2 != R");
}
