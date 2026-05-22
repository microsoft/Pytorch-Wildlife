use std::net::SocketAddr;
use std::path::PathBuf;

/// Server configuration parsed from `SPARROW_ENGINE_*` environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub model_dir: PathBuf,
    pub log_format: LogFormat,
    pub log_level: String,
    pub max_body_size: usize,
    pub max_concurrent_inference: usize,
    pub max_batch_size: usize,
    pub request_timeout_secs: u64,
    pub drain_timeout_secs: u64,
    pub device: String,
    pub inter_threads: Option<u32>,
    pub intra_threads: Option<u32>,
    /// Idle-unload background reaper period. 0 disables the feature.
    /// Default 1800 sec (30 min). Configurable via `SPARROW_ENGINE_IDLE_UNLOAD_SEC`.
    pub idle_unload_seconds: u64,
    /// Number of most-recently-used models to always keep loaded, regardless
    /// of idle age. Default 1 (protect the hot model). Configurable via
    /// `SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N`.
    pub idle_unload_keep_last_n: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogFormat {
    Json,
    Pretty,
}

impl Config {
    /// Parse configuration from environment variables. Panics on invalid values.
    pub fn from_env() -> Self {
        let bind_addr = env_or("SPARROW_ENGINE_BIND_ADDR", "0.0.0.0:8080")
            .parse::<SocketAddr>()
            .expect("SPARROW_ENGINE_BIND_ADDR must be a valid socket address");

        let model_dir = PathBuf::from(env_or("SPARROW_ENGINE_MODEL_DIR", "/models"));

        let log_format_str = env_or("SPARROW_ENGINE_LOG_FORMAT", "json");
        let log_format = match log_format_str.as_str() {
            "json" => LogFormat::Json,
            "pretty" => LogFormat::Pretty,
            other => panic!("SPARROW_ENGINE_LOG_FORMAT must be 'json' or 'pretty', got '{other}'"),
        };

        let log_level = env_or("SPARROW_ENGINE_LOG_LEVEL", "info");

        let max_body_size = parse_size(&env_or("SPARROW_ENGINE_MAX_BODY_SIZE", "100mb"));
        assert!(
            max_body_size > 0,
            "SPARROW_ENGINE_MAX_BODY_SIZE must be > 0"
        );
        let max_concurrent_inference: usize =
            env_or("SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE", "32")
                .parse()
                .expect("SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE must be a positive integer");
        assert!(
            max_concurrent_inference > 0,
            "SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE must be > 0"
        );
        let max_batch_size: usize = env_or("SPARROW_ENGINE_MAX_BATCH_SIZE", "64")
            .parse()
            .expect("SPARROW_ENGINE_MAX_BATCH_SIZE must be a positive integer");
        assert!(
            max_batch_size > 0,
            "SPARROW_ENGINE_MAX_BATCH_SIZE must be > 0"
        );
        let request_timeout_secs: u64 = env_or("SPARROW_ENGINE_REQUEST_TIMEOUT", "120")
            .parse()
            .expect("SPARROW_ENGINE_REQUEST_TIMEOUT must be a number of seconds");
        assert!(
            request_timeout_secs > 0,
            "SPARROW_ENGINE_REQUEST_TIMEOUT must be > 0"
        );
        let drain_timeout_secs: u64 = env_or("SPARROW_ENGINE_DRAIN_TIMEOUT", "10")
            .parse()
            .expect("SPARROW_ENGINE_DRAIN_TIMEOUT must be a number of seconds");
        assert!(
            drain_timeout_secs > 0,
            "SPARROW_ENGINE_DRAIN_TIMEOUT must be > 0"
        );
        let device = env_or("SPARROW_ENGINE_DEVICE", "auto");
        let inter_threads = std::env::var("SPARROW_ENGINE_INTER_THREADS")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().expect("SPARROW_ENGINE_INTER_THREADS must be u32"));
        let intra_threads = std::env::var("SPARROW_ENGINE_INTRA_THREADS")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().expect("SPARROW_ENGINE_INTRA_THREADS must be u32"));

        // Idle-unload reaper. Default 1800s (30 min). Set to 0 to disable the
        // background task entirely. `keep_last_n` defaults to 1 — the most
        // recently used model always stays loaded so the hot path doesn't
        // pay a cold-load tax during normal operation.
        let idle_unload_seconds: u64 = env_or("SPARROW_ENGINE_IDLE_UNLOAD_SEC", "1800")
            .parse()
            .expect("SPARROW_ENGINE_IDLE_UNLOAD_SEC must be a non-negative integer (seconds)");
        let idle_unload_keep_last_n: usize = env_or("SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N", "1")
            .parse()
            .expect("SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N must be a non-negative integer");

        Self {
            bind_addr,
            model_dir,
            log_format,
            log_level,
            max_body_size,
            max_concurrent_inference,
            max_batch_size,
            request_timeout_secs,
            drain_timeout_secs,
            device,
            inter_threads,
            intra_threads,
            idle_unload_seconds,
            idle_unload_keep_last_n,
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Parse a human-readable size string (e.g., "100mb") into bytes.
fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("gb") {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("mb") {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("kb") {
        (n, 1024)
    } else {
        (s.as_str(), 1)
    };
    num_str.trim().parse::<usize>().expect("invalid size") * multiplier
}
