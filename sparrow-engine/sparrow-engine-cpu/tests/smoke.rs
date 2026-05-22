// Smoke test: verifies the common test module compiles and golden files are accessible.
mod common;

#[test]
fn golden_dir_exists() {
    let dir = common::golden_dir();
    assert!(dir.exists(), "Golden dir not found: {:?}", dir);
}

#[test]
fn test_cameratrap_dir_exists() {
    let dir = common::test_cameratrap_dir();
    assert!(dir.exists(), "Camera trap images dir not found: {:?}", dir);
}

#[test]
fn can_load_golden_detection() {
    let paths = common::image_paths_from(&common::test_cameratrap_dir(), 1);
    assert!(!paths.is_empty(), "No test images found");
    let image_name = paths[0].file_name().unwrap().to_str().unwrap();
    let golden = common::load_golden_detections("mdv6", image_name);
    assert!(!golden.detections.is_empty(), "Golden detections empty");
}

#[test]
fn can_load_golden_classification() {
    let paths = common::image_paths_from(&common::test_cameratrap_dir(), 1);
    assert!(!paths.is_empty(), "No test images found");
    let image_name = paths[0].file_name().unwrap().to_str().unwrap();
    let golden = common::load_golden_classifications("speciesnet", image_name);
    assert!(
        !golden.classifications.is_empty(),
        "Golden classifications empty",
    );
}
