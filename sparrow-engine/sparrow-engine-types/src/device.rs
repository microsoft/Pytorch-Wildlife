//! Compute device selection: Auto, CPU, or CUDA.
//!
//! Moved from the legacy monolithic engine crate for Phase 3.8 Phase A crate split.
//! Pure data + trait impls; zero ORT/CUDA deps.

/// Compute device selection.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Device {
    /// Auto-detect: try CUDA first, fall back to CPU.
    #[default]
    Auto,
    Cpu,
    Cuda(u32),
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Auto => write!(f, "auto"),
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(0) => write!(f, "cuda:0"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            "gpu" | "cuda" | "cuda:0" => Ok(Device::Cuda(0)),
            s if s.starts_with("cuda:") => {
                let idx = s[5..]
                    .parse::<u32>()
                    .map_err(|e| format!("invalid CUDA device index: {e}"))?;
                Ok(Device::Cuda(idx))
            }
            "directml" => Err(
                "DirectML is not yet supported. Use 'cpu' or 'cuda'.".to_string(),
            ),
            other => Err(format!(
                "unknown device: '{other}'. Valid: auto, cpu, gpu, cuda, cuda:N"
            )),
        }
    }
}

#[cfg(test)]
mod phase_a_r1_device_tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn device_parse_auto_lowercase() {
        assert_eq!(Device::from_str("auto").unwrap(), Device::Auto);
    }

    #[test]
    fn device_parse_auto_mixed_case() {
        // FromStr lowercases internally — uppercase + mixed case must work.
        assert_eq!(Device::from_str("AUTO").unwrap(), Device::Auto);
        assert_eq!(Device::from_str("Auto").unwrap(), Device::Auto);
    }

    #[test]
    fn device_parse_cpu_and_aliases() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("CPU").unwrap(), Device::Cpu);
    }

    #[test]
    fn device_parse_gpu_aliases_resolve_to_cuda_zero() {
        // "gpu", "cuda", "cuda:0" all collapse to Cuda(0) per device.rs:33.
        assert_eq!(Device::from_str("gpu").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("GPU").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("CUDA").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("cuda:0").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn device_parse_cuda_with_index() {
        assert_eq!(Device::from_str("cuda:7").unwrap(), Device::Cuda(7));
        assert_eq!(Device::from_str("cuda:1").unwrap(), Device::Cuda(1));
        assert_eq!(Device::from_str("CUDA:3").unwrap(), Device::Cuda(3));
    }

    #[test]
    fn device_parse_invalid_cuda_index_returns_err() {
        // "cuda:not_a_number" must fail with helpful message mentioning the index.
        let err = Device::from_str("cuda:not_a_number").unwrap_err();
        assert!(
            err.contains("invalid CUDA device index"),
            "expected error mentioning 'invalid CUDA device index', got: {err}"
        );
    }

    #[test]
    fn device_parse_directml_helpful_message() {
        // DirectML is reserved for future use; current path returns a guiding error.
        let err = Device::from_str("directml").unwrap_err();
        assert!(
            err.contains("DirectML"),
            "expected error mentioning DirectML, got: {err}"
        );
        assert!(
            err.contains("cpu") || err.contains("cuda"),
            "expected error suggesting cpu/cuda alternative, got: {err}"
        );
    }

    #[test]
    fn device_parse_garbage_string_returns_err() {
        let err = Device::from_str("garbage").unwrap_err();
        assert!(
            err.contains("unknown device") && err.contains("garbage"),
            "expected error naming the bad input, got: {err}"
        );
    }

    #[test]
    fn device_format_round_trip_auto_cpu_cuda() {
        // For each canonical surface form, parse -> Display -> parse must be stable.
        for canonical in &["auto", "cpu", "cuda:0", "cuda:7"] {
            let parsed: Device = canonical.parse().unwrap();
            let rendered = parsed.to_string();
            assert_eq!(
                &rendered, canonical,
                "round-trip mismatch for {canonical}: rendered as {rendered}"
            );
            // Re-parse the rendered form: must equal the original parsed value.
            let reparsed: Device = rendered.parse().unwrap();
            assert_eq!(reparsed, parsed);
        }
    }

    #[test]
    fn device_default_is_auto() {
        let d: Device = Default::default();
        assert_eq!(d, Device::Auto);
    }
}
