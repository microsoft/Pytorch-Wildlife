//! Export detection results to MegaDet v1.5 JSON, COCO JSON, and CSV.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use sparrow_engine_types::{BBox, DetectResult, Detection, PipelineResult, Result};

// ---------------------------------------------------------------------------
// Bbox conversion helpers
// ---------------------------------------------------------------------------

/// Convert normalized xyxy bbox to xywh (normalized).
fn bbox_to_xywh(bbox: &BBox) -> [f32; 4] {
    [
        bbox.x_min,
        bbox.y_min,
        bbox.x_max - bbox.x_min,
        bbox.y_max - bbox.y_min,
    ]
}

/// Convert normalized xyxy bbox to pixel xywh.
fn bbox_to_pixels(bbox: &BBox, w: u32, h: u32) -> [f32; 4] {
    [
        bbox.x_min * w as f32,
        bbox.y_min * h as f32,
        (bbox.x_max - bbox.x_min) * w as f32,
        (bbox.y_max - bbox.y_min) * h as f32,
    ]
}

// ---------------------------------------------------------------------------
// MegaDet v1.5 JSON
// ---------------------------------------------------------------------------

/// Export as MegaDetector v1.5 JSON format.
///
/// Each entry is `(file_path, DetectResult)`.
/// Bbox: `[x_min, y_min, width, height]` normalized [0,1].
/// Category IDs: "1"=animal, "2"=person, "3"=vehicle for standard labels, sequential for others.
pub fn to_megadet(
    results: &[(&Path, &DetectResult)],
    model_id: &str,
    writer: &mut impl Write,
) -> Result<()> {
    let mut images = Vec::new();
    for (path, result) in results {
        let mut detections = Vec::new();
        for d in &result.detections {
            let xywh = bbox_to_xywh(&d.bbox);
            let cat = megadet_category(&d.label);
            detections.push(serde_json::json!({
                "category": cat,
                "conf": d.confidence,
                "bbox": xywh,
            }));
        }
        images.push(serde_json::json!({
            "file": path.display().to_string(),
            "max_detection_conf": result.detections.iter().map(|d| d.confidence).fold(0.0f32, f32::max),
            "detections": detections,
        }));
    }

    let output = serde_json::json!({
        "info": {
            "detector": model_id,
            "format_version": "1.5",
        },
        "detection_categories": {
            "1": "animal",
            "2": "person",
            "3": "vehicle",
        },
        "images": images,
    });

    serde_json::to_writer_pretty(writer, &output)?;
    Ok(())
}

fn megadet_category(label: &str) -> &str {
    match label.to_lowercase().as_str() {
        "animal" => "1",
        "person" | "human" => "2",
        "vehicle" | "car" => "3",
        // MegaDet convention: unknown labels map to category "1" (animal).
        _ => "1",
    }
}

// ---------------------------------------------------------------------------
// COCO JSON
// ---------------------------------------------------------------------------

/// Export as COCO JSON format.
///
/// Bbox: `[x, y, width, height]` in pixels.
///
/// # Invariant: `label_id >= 1`
///
/// COCO category IDs are 1-indexed by convention (the official COCO dataset
/// reserves 0 as "background" / no-object). Detectors that feed `to_coco`
/// must emit `Detection::label_id >= 1`. A `label_id` of `0` is not rejected
/// here — it is written through as-is and produces a COCO file that some
/// downstream tools (e.g. pycocotools) treat as background and silently drop.
///
/// Models built on the sparrow-engine manifest map category names to 1-indexed IDs in
/// `manifest.toml`; this function relies on that convention rather than
/// re-mapping at export time.
///
/// ## Cross-namespace collisions (pipeline export)
///
/// When exporting pipeline results (detector + classifier), the two label
/// spaces are typically both 1-indexed and can collide (e.g. MD v6
/// `animal=1` and SpeciesNet `species_A=1`). On collision, `to_coco` keeps
/// the first-seen `(label_id, label)` pair, emits a one-shot warning to
/// stderr (deduped per `label_id`), and continues. Downstream consumers
/// see one COCO category per distinct `label_id` with the first-seen name;
/// subsequent annotations with the same `label_id` but a different `label`
/// reference the first-seen category.
///
/// A full namespace strategy (offset / prefix / separate category spaces)
/// is tracked as a Phase 3.5 item (`docs/ideas.md`).
pub fn to_coco(results: &[(&Path, &DetectResult)], writer: &mut impl Write) -> Result<()> {
    use std::collections::btree_map::Entry;

    let mut images = Vec::new();
    let mut annotations = Vec::new();
    let mut ann_id = 1u64;
    let mut seen_categories = std::collections::BTreeMap::new();
    // Dedup: emit at most one collision warning per label_id across the entire
    // export batch. See function doc "Cross-namespace collisions" for details.
    let mut warned_ids: HashSet<u32> = HashSet::new();

    for (img_id, (path, result)) in results.iter().enumerate() {
        let image_id = img_id + 1;
        images.push(serde_json::json!({
            "id": image_id,
            "file_name": path.display().to_string(),
            "width": result.image_width,
            "height": result.image_height,
        }));

        for d in &result.detections {
            match seen_categories.entry(d.label_id) {
                Entry::Vacant(v) => {
                    v.insert(d.label.clone());
                }
                Entry::Occupied(o) if o.get() != &d.label => {
                    if warned_ids.insert(d.label_id) {
                        eprintln!(
                            "warning: to_coco label_id={} collision — first-seen label {:?} kept; \
                             subsequent label {:?} dropped. This typically indicates mixing \
                             detector and classifier namespaces in pipeline output; see to_coco \
                             doc for the namespace invariant.",
                            d.label_id,
                            o.get(),
                            d.label
                        );
                    }
                }
                Entry::Occupied(_) => {}
            }
            let px = bbox_to_pixels(&d.bbox, result.image_width, result.image_height);
            let area = px[2] * px[3];
            annotations.push(serde_json::json!({
                "id": ann_id,
                "image_id": image_id,
                "category_id": d.label_id,
                "bbox": px,
                "area": area,
                "score": d.confidence,
                "iscrowd": 0,
            }));
            ann_id += 1;
        }
    }

    let categories: Vec<serde_json::Value> = seen_categories
        .iter()
        .map(|(id, name)| {
            serde_json::json!({
                "id": id,
                "name": name,
            })
        })
        .collect();

    let output = serde_json::json!({
        "images": images,
        "annotations": annotations,
        "categories": categories,
    });

    serde_json::to_writer_pretty(writer, &output)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// RFC 4180 CSV escaping: quote fields containing `,`, `"`, or newlines.
pub fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Export as CSV.
///
/// Header: `file,label,confidence,x_min,y_min,x_max,y_max` (normalized xyxy).
pub fn to_csv(results: &[(&Path, &DetectResult)], writer: &mut impl Write) -> Result<()> {
    writeln!(writer, "file,label,confidence,x_min,y_min,x_max,y_max")?;
    for (path, result) in results {
        let path_str = csv_escape(&path.display().to_string());
        for d in &result.detections {
            let label_str = csv_escape(&d.label);
            writeln!(
                writer,
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                path_str,
                label_str,
                d.confidence,
                d.bbox.x_min,
                d.bbox.y_min,
                d.bbox.x_max,
                d.bbox.y_max,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pipeline → DetectResult converter
// ---------------------------------------------------------------------------

/// Convert pipeline results into detection-shaped entries for export.
///
/// Classification label is preferred when present; falls back to detection label.
/// Confidence follows the same rule: classification confidence is used when a
/// classification exists, otherwise detection confidence. Downstream consumers
/// filtering on `max_detection_conf` in MegaDet JSON will therefore see
/// classification-confidence values for classified detections — these are drawn
/// from a different distribution than raw detection confidence.
/// This lets `to_megadet`, `to_coco`, and `to_csv` work with pipeline output.
///
/// Note: the output mixes detector and classifier `label_id` spaces. See
/// [`to_coco`] "Cross-namespace collisions" for how COCO export handles
/// colliding `label_id`s (first-seen wins, one-shot stderr warning).
pub fn pipeline_results_to_detect_entries(
    results: &[(&Path, &PipelineResult)],
) -> Vec<(PathBuf, DetectResult)> {
    results
        .iter()
        .map(|(path, pr)| {
            let detections = pr
                .detections
                .iter()
                .map(|pd| {
                    let (label, label_id, confidence) = match &pd.classification {
                        Some(cls) => (cls.label.clone(), cls.label_id, cls.confidence),
                        None => (
                            pd.detection.label.clone(),
                            pd.detection.label_id,
                            pd.detection.confidence,
                        ),
                    };
                    Detection {
                        bbox: pd.detection.bbox,
                        label,
                        label_id,
                        confidence,
                    }
                })
                .collect();
            (
                path.to_path_buf(),
                DetectResult {
                    detections,
                    image_width: pr.image_width,
                    image_height: pr.image_height,
                    processing_time_ms: pr.processing_time_ms,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::Detection;

    fn make_det(label: &str, conf: f32, bbox: [f32; 4]) -> Detection {
        Detection {
            bbox: BBox {
                x_min: bbox[0],
                y_min: bbox[1],
                x_max: bbox[2],
                y_max: bbox[3],
            },
            label: label.to_string(),
            // COCO convention: category_id starts at 1 (0 is reserved for
            // background). See to_coco() invariant.
            label_id: 1,
            confidence: conf,
        }
    }

    fn make_result(dets: Vec<Detection>) -> DetectResult {
        DetectResult {
            detections: dets,
            image_width: 1920,
            image_height: 1080,
            processing_time_ms: 50.0,
        }
    }

    #[test]
    fn bbox_xywh_conversion() {
        let b = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.5,
            y_max: 0.6,
        };
        let xywh = bbox_to_xywh(&b);
        assert!((xywh[0] - 0.1).abs() < 1e-6);
        assert!((xywh[1] - 0.2).abs() < 1e-6);
        assert!((xywh[2] - 0.4).abs() < 1e-6);
        assert!((xywh[3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn bbox_pixels_conversion() {
        let b = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.5,
            y_max: 1.0,
        };
        let px = bbox_to_pixels(&b, 200, 100);
        assert!((px[0] - 0.0).abs() < 1e-6);
        assert!((px[1] - 0.0).abs() < 1e-6);
        assert!((px[2] - 100.0).abs() < 1e-6);
        assert!((px[3] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn megadet_json_structure() {
        let result = make_result(vec![make_det("animal", 0.95, [0.1, 0.2, 0.5, 0.6])]);
        let path = Path::new("test.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_megadet(&entries, "mdv6", &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        assert_eq!(json["info"]["format_version"], "1.5");
        assert_eq!(json["info"]["detector"], "mdv6");
        assert_eq!(json["images"].as_array().unwrap().len(), 1);
        let img = &json["images"][0];
        assert_eq!(img["detections"].as_array().unwrap().len(), 1);
        assert_eq!(img["detections"][0]["category"], "1");
    }

    #[test]
    fn coco_json_structure() {
        let result = make_result(vec![make_det("animal", 0.9, [0.0, 0.0, 0.5, 0.5])]);
        let path = Path::new("img.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_coco(&entries, &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        assert_eq!(json["images"].as_array().unwrap().len(), 1);
        assert_eq!(json["annotations"].as_array().unwrap().len(), 1);
        let ann = &json["annotations"][0];
        assert_eq!(ann["image_id"], 1);
    }

    #[test]
    fn csv_header_and_rows() {
        let result = make_result(vec![
            make_det("animal", 0.9, [0.1, 0.2, 0.3, 0.4]),
            make_det("person", 0.8, [0.5, 0.6, 0.7, 0.8]),
        ]);
        let path = Path::new("test.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_csv(&entries, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "file,label,confidence,x_min,y_min,x_max,y_max");
        assert_eq!(lines.len(), 3); // header + 2 detections
        assert!(lines[1].starts_with("test.jpg,animal,"));
    }

    #[test]
    fn empty_results_csv() {
        let entries: Vec<(&Path, &DetectResult)> = vec![];
        let mut buf = Vec::new();
        to_csv(&entries, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1); // header only
    }

    #[test]
    fn empty_results_megadet() {
        let entries: Vec<(&Path, &DetectResult)> = vec![];
        let mut buf = Vec::new();
        to_megadet(&entries, "mdv6", &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(json["images"].as_array().unwrap().len(), 0);
    }

    // Regression: CSV fields with commas must be escaped (RFC 4180).
    #[test]
    fn csv_escapes_commas_in_path() {
        let result = make_result(vec![make_det("animal", 0.9, [0.1, 0.2, 0.3, 0.4])]);
        let path = Path::new("Smith, John/camera trap/img.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_csv(&entries, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        // Path with comma must be quoted.
        assert!(lines[1].starts_with("\"Smith, John/camera trap/img.jpg\""));
    }

    // Regression: CSV fields with quotes must be double-quoted.
    #[test]
    fn csv_escapes_quotes_in_label() {
        let result = make_result(vec![make_det("\"bird\"", 0.8, [0.1, 0.2, 0.3, 0.4])]);
        let path = Path::new("test.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_csv(&entries, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Label with quotes must be escaped: "bird" -> """bird"""
        assert!(output.contains("\"\"\"bird\"\"\""));
    }

    // Regression (MI-2, R4): when two detections share label_id but have
    // different labels (e.g. pipeline output mixing detector + classifier
    // 1-indexed namespaces), to_coco must keep the first-seen (label_id, label)
    // pair and not silently drop either category. The collision-branch code
    // path also emits a one-shot stderr warning per colliding label_id.
    //
    // Warn emission is verified by code inspection; stderr capture is not
    // asserted here. JSON state below confirms the collision branch was
    // entered (first-wins preserved, not silent drop of both annotations).
    #[test]
    fn coco_first_seen_label_wins_on_collision() {
        let mut det_a = make_det("species_A", 0.9, [0.0, 0.0, 0.4, 0.4]);
        det_a.label_id = 1;
        let mut det_b = make_det("animal", 0.8, [0.5, 0.5, 0.9, 0.9]);
        det_b.label_id = 1; // same id, different label -> collision
        let result = make_result(vec![det_a, det_b]);
        let path = Path::new("img.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_coco(&entries, &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        let cats = json["categories"].as_array().unwrap();
        assert_eq!(
            cats.len(),
            1,
            "collision must collapse to a single COCO category"
        );
        assert_eq!(cats[0]["id"], 1);
        // First-seen wins: "species_A" was inserted first.
        assert_eq!(cats[0]["name"], "species_A");

        // Both annotations must still be emitted, both referencing category_id=1.
        let anns = json["annotations"].as_array().unwrap();
        assert_eq!(anns.len(), 2, "both annotations preserved despite collision");
        assert_eq!(anns[0]["category_id"], 1);
        assert_eq!(anns[1]["category_id"], 1);
    }

    // Regression: COCO categories array must be populated from detections.
    #[test]
    fn coco_categories_populated() {
        let mut det = make_det("animal", 0.9, [0.0, 0.0, 0.5, 0.5]);
        det.label_id = 1;
        let result = make_result(vec![det]);
        let path = Path::new("img.jpg");
        let entries: Vec<(&Path, &DetectResult)> = vec![(path, &result)];
        let mut buf = Vec::new();
        to_coco(&entries, &mut buf).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        let cats = json["categories"].as_array().unwrap();
        assert!(!cats.is_empty(), "categories array must not be empty");
        assert_eq!(cats[0]["name"], "animal");
        assert_eq!(cats[0]["id"], 1);
    }

    // Regression: csv_escape function correctness.
    #[test]
    fn csv_escape_cases() {
        assert_eq!(csv_escape("simple"), "simple");
        assert_eq!(csv_escape("has,comma"), "\"has,comma\"");
        assert_eq!(csv_escape("has\"quote"), "\"has\"\"quote\"");
        assert_eq!(csv_escape("has\nnewline"), "\"has\nnewline\"");
        assert_eq!(csv_escape("clean_path.jpg"), "clean_path.jpg");
    }

    // --- pipeline_results_to_detect_entries tests ---

    use sparrow_engine_types::{Classification, PipelineDetection, PipelineResult};

    fn make_pipeline_detection(
        det_label: &str,
        det_conf: f32,
        bbox: [f32; 4],
        cls: Option<(&str, f32)>,
    ) -> PipelineDetection {
        PipelineDetection {
            detection: Detection {
                bbox: BBox {
                    x_min: bbox[0],
                    y_min: bbox[1],
                    x_max: bbox[2],
                    y_max: bbox[3],
                },
                label: det_label.to_string(),
                // COCO convention: category_id starts at 1 (see to_coco invariant).
                label_id: 1,
                confidence: det_conf,
            },
            classification: cls.map(|(label, conf)| Classification {
                label: label.to_string(),
                label_id: 42,
                confidence: conf,
            }),
        }
    }

    #[test]
    fn test_pipeline_to_detect_with_classification() {
        let pr = PipelineResult {
            pipeline_id: "test".to_string(),
            detections: vec![make_pipeline_detection(
                "animal",
                0.9,
                [0.1, 0.2, 0.3, 0.4],
                Some(("deer", 0.85)),
            )],
            image_width: 1920,
            image_height: 1080,
            processing_time_ms: 100.0,
        };
        let path = Path::new("img.jpg");
        let entries = pipeline_results_to_detect_entries(&[(path, &pr)]);
        assert_eq!(entries.len(), 1);
        let (p, dr) = &entries[0];
        assert_eq!(p, Path::new("img.jpg"));
        assert_eq!(dr.detections.len(), 1);
        assert_eq!(dr.detections[0].label, "deer");
        assert_eq!(dr.detections[0].label_id, 42);
        assert!((dr.detections[0].confidence - 0.85).abs() < 1e-6);
        assert!((dr.detections[0].bbox.x_min - 0.1).abs() < 1e-6);
        assert_eq!(dr.image_width, 1920);
        assert_eq!(dr.processing_time_ms, 100.0);
    }

    #[test]
    fn test_pipeline_to_detect_fallback_label() {
        let pr = PipelineResult {
            pipeline_id: "test".to_string(),
            detections: vec![make_pipeline_detection(
                "animal",
                0.9,
                [0.1, 0.2, 0.3, 0.4],
                None,
            )],
            image_width: 640,
            image_height: 480,
            processing_time_ms: 50.0,
        };
        let path = Path::new("img2.jpg");
        let entries = pipeline_results_to_detect_entries(&[(path, &pr)]);
        assert_eq!(entries.len(), 1);
        let det = &entries[0].1.detections[0];
        assert_eq!(det.label, "animal");
        // Fixture label_id updated to 1 for COCO 1-indexed convention (EX2).
        assert_eq!(det.label_id, 1);
        assert!((det.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_to_detect_empty() {
        let entries: Vec<(&Path, &PipelineResult)> = vec![];
        let result = pipeline_results_to_detect_entries(&entries);
        assert!(result.is_empty());
    }
}

#[cfg(test)]
mod phase_a_r1_export {
    use super::*;
    use sparrow_engine_types::{BBox, DetectResult, Detection};

    fn d(label: &str, label_id: u32, conf: f32, bbox: [f32; 4]) -> Detection {
        Detection {
            bbox: BBox {
                x_min: bbox[0],
                y_min: bbox[1],
                x_max: bbox[2],
                y_max: bbox[3],
            },
            label: label.to_string(),
            label_id,
            confidence: conf,
        }
    }

    fn r(dets: Vec<Detection>) -> DetectResult {
        DetectResult {
            detections: dets,
            image_width: 640,
            image_height: 480,
            processing_time_ms: 10.0,
        }
    }

    /// Plain string round-trip — RFC 4180 says fields without comma, quote, or
    /// newline are emitted verbatim. The existing `csv_escape_cases` covers
    /// this but pins it again here as a separate, single-purpose assertion.
    #[test]
    fn csv_escape_plain_passthrough() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape(""), "");
    }

    /// Comma forces quoting. Pins the literal output shape so a regression
    /// that changes the wrapping rule (e.g., to backslash-escape) breaks here.
    #[test]
    fn csv_escape_quotes_field_with_comma() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
    }

    /// Embedded double-quote is escaped by doubling per RFC 4180, AND the
    /// whole field is wrapped in quotes. Existing `csv_escape_cases` covers
    /// `has"quote` → `"has""quote"` but this pins the slightly-trickier
    /// edge of leading + trailing quotes (`"quote"` → `"""quote"""`).
    #[test]
    fn csv_escape_doubles_internal_quotes_rfc4180() {
        // Input: "quote"
        // Output: "" "quote"" "" wrapped → """quote"""
        assert_eq!(csv_escape("\"quote\""), "\"\"\"quote\"\"\"");
    }

    /// Embedded newline forces quoting (multi-line CSV cell, RFC 4180 §2.6).
    #[test]
    fn csv_escape_quotes_field_with_newline() {
        assert_eq!(csv_escape("line\nbreak"), "\"line\nbreak\"");
        // Carriage-return path also covered by csv_escape (line 207).
        assert_eq!(csv_escape("line\rbreak"), "\"line\rbreak\"");
    }

    /// `to_megadet` round-trip: the JSON the function emits must parse back
    /// to a JSON object whose `info.detector` matches the model_id we passed,
    /// `info.format_version` is "1.5" (the spec we promised), and each bbox
    /// is xywh (width = x_max - x_min, height = y_max - y_min).
    #[test]
    fn to_megadet_round_trip_schema_and_bbox_xywh() {
        let res = r(vec![d("animal", 1, 0.7, [0.1, 0.2, 0.5, 0.6])]);
        let path = std::path::Path::new("img.jpg");
        let entries: Vec<(&std::path::Path, &DetectResult)> = vec![(path, &res)];
        let mut buf = Vec::new();
        to_megadet(&entries, "mdv6", &mut buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        assert_eq!(parsed["info"]["detector"], "mdv6");
        assert_eq!(parsed["info"]["format_version"], "1.5");

        let bbox = &parsed["images"][0]["detections"][0]["bbox"];
        // Bbox is `[x_min, y_min, width, height]` per Phase 3 design decision.
        // Allow ~1e-6 slop on f32→f64 JSON encoding.
        let x = bbox[0].as_f64().unwrap();
        let y = bbox[1].as_f64().unwrap();
        let w = bbox[2].as_f64().unwrap();
        let h = bbox[3].as_f64().unwrap();
        assert!((x - 0.1).abs() < 1e-6);
        assert!((y - 0.2).abs() < 1e-6);
        assert!((w - 0.4).abs() < 1e-6, "width must be x_max-x_min, got {w}");
        assert!((h - 0.4).abs() < 1e-6, "height must be y_max-y_min, got {h}");
    }

    /// COCO category collision: existing `coco_first_seen_label_wins_on_collision`
    /// covers the within-image case. This widens to cross-image — same
    /// `label_id` reused across two different image entries with conflicting
    /// labels still produces ONE category, first-seen wins.
    #[test]
    fn to_coco_category_collision_first_seen_wins_across_images() {
        let mut da = d("species_X", 1, 0.9, [0.0, 0.0, 0.4, 0.4]);
        let mut db = d("animal", 1, 0.8, [0.5, 0.5, 0.9, 0.9]);
        da.label_id = 1;
        db.label_id = 1;
        let res_a = r(vec![da]);
        let res_b = r(vec![db]);
        let path_a = std::path::Path::new("a.jpg");
        let path_b = std::path::Path::new("b.jpg");
        let entries: Vec<(&std::path::Path, &DetectResult)> =
            vec![(path_a, &res_a), (path_b, &res_b)];
        let mut buf = Vec::new();
        to_coco(&entries, &mut buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&buf).unwrap();

        let cats = parsed["categories"].as_array().unwrap();
        assert_eq!(
            cats.len(),
            1,
            "cross-image collision must collapse to one COCO category"
        );
        assert_eq!(cats[0]["id"], 1);
        assert_eq!(
            cats[0]["name"], "species_X",
            "first-seen label must win across images"
        );
        assert_eq!(
            parsed["annotations"].as_array().unwrap().len(),
            2,
            "both annotations preserved across images"
        );
    }

    /// CSV header schema pin — `to_csv` must emit an exact header row matching
    /// the documented schema (file,label,confidence,x_min,y_min,x_max,y_max).
    /// Any change to column order or naming breaks downstream consumers.
    #[test]
    fn to_csv_header_matches_schema() {
        let res = r(vec![]);
        let path = std::path::Path::new("img.jpg");
        let entries: Vec<(&std::path::Path, &DetectResult)> = vec![(path, &res)];
        let mut buf = Vec::new();
        to_csv(&entries, &mut buf).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(
            out.starts_with("file,label,confidence,x_min,y_min,x_max,y_max\n"),
            "header line must match exact schema, got {out:?}"
        );
    }
}
