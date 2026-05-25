use image::{DynamicImage, Rgba, RgbaImage};
use sparrow_engine_core::viz::{render_audio_layers, AudioLayersOpts};
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

fn assert_dimensions_match(layers: &[(&'static str, DynamicImage)], width: u32, height: u32) {
    for (name, img) in layers {
        assert_eq!(img.width(), width, "{name} width must match spectrogram");
        assert_eq!(img.height(), height, "{name} height must match spectrogram");
    }
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
        assert_eq!(
            img.width(),
            spec.width(),
            "{name} width must match spectrogram"
        );
        if *name == "02_segments_windows" {
            assert!(
                img.height() > spec.height(),
                "window lanes layer should append a lanes band below the spectrogram"
            );
        } else {
            assert_eq!(
                img.height(),
                spec.height(),
                "{name} height must match spectrogram"
            );
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
