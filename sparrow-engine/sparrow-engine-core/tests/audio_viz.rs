use image::{DynamicImage, Rgba, RgbaImage};
use sparrow_engine_core::viz::{
    render_audio_layers, render_window_lanes, AudioLayersOpts, WindowLanesOpts,
};
use sparrow_engine_types::{AudioRange, AudioSegment};

fn synthetic_spectrogram(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        let v = (y as f32 / height as f32 * 128.0) as u8 + 32;
        for x in 0..width {
            img.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    DynamicImage::ImageRgba8(img)
}

fn segments() -> Vec<AudioSegment> {
    vec![
        AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.2,
            classes: Vec::new(),
        },
        AudioSegment {
            start_time_s: 0.3,
            end_time_s: 1.3,
            confidence: 0.8,
            classes: Vec::new(),
        },
        AudioSegment {
            start_time_s: 1.2,
            end_time_s: 2.0,
            confidence: 0.5,
            classes: Vec::new(),
        },
    ]
}

fn ranges() -> Vec<AudioRange> {
    vec![AudioRange {
        start_time_s: 0.3,
        end_time_s: 1.6,
        max_confidence: 0.8,
        class: None,
    }]
}

fn layer_names(layers: &[(&'static str, DynamicImage)]) -> Vec<&'static str> {
    layers.iter().map(|(name, _)| *name).collect()
}

fn find_layer<'a>(layers: &'a [(&'static str, DynamicImage)], name: &str) -> &'a DynamicImage {
    layers
        .iter()
        .find_map(|(layer_name, img)| (*layer_name == name).then_some(img))
        .unwrap_or_else(|| panic!("missing layer {name}"))
}

fn layer_bytes(layer: &DynamicImage) -> Vec<u8> {
    layer.to_rgba8().into_raw()
}

fn assert_dimensions_match(layers: &[(&'static str, DynamicImage)], width: u32, height: u32) {
    for (name, img) in layers {
        assert_eq!(img.width(), width, "{name} width must match spectrogram");
        assert_eq!(img.height(), height, "{name} height must match spectrogram");
    }
}

fn assert_window_layer_geometry(img: &DynamicImage, spec: &DynamicImage) {
    assert_eq!(img.width(), spec.width(), "window layer width");
    assert_eq!(
        img.height(),
        spec.height() + 1 + (4 * 2 + 3 * 2),
        "default window lanes geometry changed"
    );
}

#[test]
fn render_audio_layers_default_emits_three_layers() {
    let spec = synthetic_spectrogram(100, 20);
    let layers = render_audio_layers(&spec, &segments(), None, 2.0, &AudioLayersOpts::default());

    assert_eq!(
        layer_names(&layers),
        ["01_spec", "02_segments", "03_heatmap"]
    );
    assert_dimensions_match(&layers, spec.width(), spec.height());
    assert_eq!(
        layer_bytes(find_layer(&layers, "02_segments")),
        layer_bytes(find_layer(&layers, "03_heatmap")),
        "smooth=false should make 03_heatmap equal 02_segments"
    );
}

#[test]
fn render_audio_layers_with_ranges_emits_four_layers() {
    let spec = synthetic_spectrogram(100, 20);
    let ranges = ranges();
    let layers = render_audio_layers(
        &spec,
        &segments(),
        Some(&ranges),
        2.0,
        &AudioLayersOpts::default(),
    );

    assert_eq!(
        layer_names(&layers),
        ["01_spec", "02_segments", "03_heatmap", "04_full"]
    );
    assert_dimensions_match(&layers, spec.width(), spec.height());
    assert_ne!(
        layer_bytes(find_layer(&layers, "03_heatmap")),
        layer_bytes(find_layer(&layers, "04_full")),
        "non-empty ranges should draw 04_full overlays"
    );
}

#[test]
fn render_audio_layers_with_show_windows() {
    let spec = synthetic_spectrogram(100, 20);
    let opts = AudioLayersOpts {
        show_windows: true,
        ..AudioLayersOpts::default()
    };
    let layers = render_audio_layers(&spec, &segments(), None, 2.0, &opts);

    assert_eq!(
        layer_names(&layers),
        [
            "01_spec",
            "02_segments",
            "02_segments_windows",
            "03_heatmap",
        ]
    );

    for (name, img) in &layers {
        if *name == "02_segments_windows" {
            assert_window_layer_geometry(img, &spec);
        } else {
            assert_eq!(
                img.width(),
                spec.width(),
                "{name} width must match spectrogram"
            );
            assert_eq!(
                img.height(),
                spec.height(),
                "{name} height must match spectrogram"
            );
        }
    }
}

#[test]
fn render_audio_layers_with_windows_and_ranges_emits_five_layers() {
    let spec = synthetic_spectrogram(100, 20);
    let ranges = ranges();
    let layers = render_audio_layers(
        &spec,
        &segments(),
        Some(&ranges),
        2.0,
        &AudioLayersOpts {
            show_windows: true,
            ..AudioLayersOpts::default()
        },
    );

    assert_eq!(
        layer_names(&layers),
        [
            "01_spec",
            "02_segments",
            "02_segments_windows",
            "03_heatmap",
            "04_full",
        ]
    );
    assert_window_layer_geometry(find_layer(&layers, "02_segments_windows"), &spec);
}

#[test]
fn render_audio_layers_with_empty_ranges_emits_noop_full_layer() {
    let spec = synthetic_spectrogram(100, 20);
    let layers = render_audio_layers(
        &spec,
        &segments(),
        Some(&[]),
        2.0,
        &AudioLayersOpts::default(),
    );

    assert_eq!(
        layer_names(&layers),
        ["01_spec", "02_segments", "03_heatmap", "04_full"]
    );
    assert_eq!(
        layer_bytes(find_layer(&layers, "03_heatmap")),
        layer_bytes(find_layer(&layers, "04_full")),
        "empty ranges should emit a 04_full layer identical to 03_heatmap"
    );
}

#[test]
fn render_audio_layers_empty_segments_paints_no_overlays() {
    let spec = synthetic_spectrogram(100, 20);
    let layers = render_audio_layers(&spec, &[], None, 2.0, &AudioLayersOpts::default());
    let spec_bytes = layer_bytes(&spec);

    assert_eq!(
        layer_names(&layers),
        ["01_spec", "02_segments", "03_heatmap"]
    );
    assert_eq!(layer_bytes(find_layer(&layers, "02_segments")), spec_bytes);
    assert_eq!(layer_bytes(find_layer(&layers, "03_heatmap")), spec_bytes);
}

#[test]
fn render_audio_layers_non_finite_or_non_positive_duration_no_overlays() {
    let spec = synthetic_spectrogram(60, 16);
    let spec_bytes = layer_bytes(&spec);
    for duration_s in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let layers = render_audio_layers(
            &spec,
            &segments(),
            None,
            duration_s,
            &AudioLayersOpts::default(),
        );
        assert_eq!(
            layer_names(&layers),
            ["01_spec", "02_segments", "03_heatmap"],
            "duration_s={duration_s:?}"
        );
        assert_eq!(layer_bytes(find_layer(&layers, "02_segments")), spec_bytes);
        assert_eq!(layer_bytes(find_layer(&layers, "03_heatmap")), spec_bytes);
    }
}

#[test]
fn render_audio_layers_skips_non_finite_segments_and_ranges() {
    let spec = synthetic_spectrogram(60, 16);
    let bad_segments = vec![
        AudioSegment {
            start_time_s: f32::NAN,
            end_time_s: 1.0,
            confidence: 1.0,
            classes: Vec::new(),
        },
        AudioSegment {
            start_time_s: 0.0,
            end_time_s: f32::INFINITY,
            confidence: 1.0,
            classes: Vec::new(),
        },
        AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: f32::NAN,
            classes: Vec::new(),
        },
    ];
    let bad_ranges = vec![AudioRange {
        start_time_s: f32::NAN,
        end_time_s: 1.0,
        max_confidence: 1.0,
        class: None,
    }];
    let layers = render_audio_layers(
        &spec,
        &bad_segments,
        Some(&bad_ranges),
        2.0,
        &AudioLayersOpts::default(),
    );
    let spec_bytes = layer_bytes(&spec);

    assert_eq!(
        layer_names(&layers),
        ["01_spec", "02_segments", "03_heatmap", "04_full"]
    );
    assert_eq!(layer_bytes(find_layer(&layers, "02_segments")), spec_bytes);
    assert_eq!(layer_bytes(find_layer(&layers, "03_heatmap")), spec_bytes);
    assert_eq!(
        layer_bytes(find_layer(&layers, "03_heatmap")),
        layer_bytes(find_layer(&layers, "04_full")),
        "non-finite ranges should not draw false origin/full-width overlays"
    );
}

#[test]
fn render_audio_layers_bounds_window_lane_count() {
    let spec = synthetic_spectrogram(40, 12);
    let layers = render_audio_layers(
        &spec,
        &segments(),
        None,
        2.0,
        &AudioLayersOpts {
            show_windows: true,
            stride_s: 1.0e-9,
            ..AudioLayersOpts::default()
        },
    );
    let windows = find_layer(&layers, "02_segments_windows");

    assert_eq!(windows.width(), spec.width());
    assert!(
        windows.height() <= spec.height() + 1 + (128 * 2 + 127 * 2),
        "window lane clamp failed; height={} spec_height={}",
        windows.height(),
        spec.height()
    );
}

#[test]
fn render_window_lanes_clamps_direct_lane_count() {
    let spec = synthetic_spectrogram(40, 12);
    let img = render_window_lanes(
        &spec,
        &segments(),
        2.0,
        &WindowLanesOpts {
            n_lanes: u32::MAX,
            ..WindowLanesOpts::default()
        },
    );

    assert_eq!(img.width(), spec.width());
    assert_eq!(img.height(), spec.height() + 1 + (128 * 2 + 127 * 2));
}

#[test]
fn render_audio_layers_show_windows_preserves_zero_size_spectrograms() {
    for (width, height) in [(0, 10), (10, 0), (0, 0)] {
        let spec = synthetic_spectrogram(width, height);
        let layers = render_audio_layers(
            &spec,
            &segments(),
            None,
            2.0,
            &AudioLayersOpts {
                show_windows: true,
                ..AudioLayersOpts::default()
            },
        );

        for (name, img) in &layers {
            assert_eq!(img.width(), width, "{name} width");
            assert_eq!(img.height(), height, "{name} height");
        }
    }
}

#[test]
fn render_audio_layers_smooth_vs_unsmoothed() {
    let spec = synthetic_spectrogram(100, 20);
    let unsmoothed =
        render_audio_layers(&spec, &segments(), None, 2.0, &AudioLayersOpts::default());
    let smoothed = render_audio_layers(
        &spec,
        &segments(),
        None,
        2.0,
        &AudioLayersOpts {
            smooth: true,
            ..AudioLayersOpts::default()
        },
    );

    for name in ["01_spec", "02_segments"] {
        assert_eq!(
            layer_bytes(find_layer(&unsmoothed, name)),
            layer_bytes(find_layer(&smoothed, name)),
            "smooth flag must not affect {name}"
        );
    }
    let unsmoothed_heatmap = find_layer(&unsmoothed, "03_heatmap").to_rgba8();
    let smoothed_heatmap = find_layer(&smoothed, "03_heatmap").to_rgba8();
    assert_eq!(
        unsmoothed_heatmap.dimensions(),
        (spec.width(), spec.height())
    );
    assert_eq!(smoothed_heatmap.dimensions(), (spec.width(), spec.height()));
    assert_ne!(
        unsmoothed_heatmap.as_raw(),
        smoothed_heatmap.as_raw(),
        "smooth=true should change 03_heatmap pixel content"
    );
}
