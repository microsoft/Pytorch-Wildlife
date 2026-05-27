use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::os::raw::{c_int, c_uchar};
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::bindings::*;

const EXPECTED_NVJPEG_MAJOR: u32 = 12;
const NVJPEG_ENV_VAR: &str = "SPARROW_ENGINE_NVJPEG_LIBRARY_PATH";

pub struct NvjpegLib {
    _lib: libloading::Library,
    pub nvjpegCreateSimple: unsafe extern "C" fn(handle: *mut nvjpegHandle_t) -> nvjpegStatus_t,
    pub nvjpegJpegStateCreate: unsafe extern "C" fn(
        handle: nvjpegHandle_t,
        jpeg_handle: *mut nvjpegJpegState_t,
    ) -> nvjpegStatus_t,
    pub nvjpegGetImageInfo: unsafe extern "C" fn(
        handle: nvjpegHandle_t,
        data: *const c_uchar,
        length: usize,
        nComponents: *mut c_int,
        subsampling: *mut nvjpegChromaSubsampling_t,
        widths: *mut c_int,
        heights: *mut c_int,
    ) -> nvjpegStatus_t,
    pub nvjpegDecode: unsafe extern "C" fn(
        handle: nvjpegHandle_t,
        jpeg_handle: nvjpegJpegState_t,
        data: *const c_uchar,
        length: usize,
        output_format: nvjpegOutputFormat_t,
        destination: *mut nvjpegImage_t,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t,
    pub nvjpegJpegStateDestroy:
        unsafe extern "C" fn(jpeg_handle: nvjpegJpegState_t) -> nvjpegStatus_t,
    pub nvjpegDestroy: unsafe extern "C" fn(handle: nvjpegHandle_t) -> nvjpegStatus_t,
    pub nvjpegGetProperty:
        unsafe extern "C" fn(type_: libraryPropertyType, value: *mut c_int) -> nvjpegStatus_t,
}

#[derive(Debug)]
pub enum NvjpegInitError {
    LibraryNotFound {
        dlerror: String,
        tried_paths: Vec<PathBuf>,
    },
    IncompatibleMajor {
        found: u32,
        expected: u32,
    },
    SymbolMissing(String),
}

impl fmt::Display for NvjpegInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound { dlerror, .. } => {
                if cfg!(target_os = "windows") {
                    write!(
                        f,
                        "nvjpeg64_12.dll could not be loaded: {dlerror}. The sparrow-engine-gpu wheel requires the CUDA 12 nvjpeg runtime. Install one of: (a) the NVIDIA CUDA Toolkit 12.x for Windows (https://developer.nvidia.com/cuda-downloads — provides %CUDA_PATH%\\bin\\nvjpeg64_12.dll), (b) use the CPU wheel: pip install sparrow-engine. Override the search path with SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=C:\\abs\\path\\nvjpeg64_12.dll."
                    )
                } else {
                    write!(
                        f,
                        "libnvjpeg.so.12 could not be loaded: {dlerror}. The sparrow-engine-gpu wheel requires the CUDA 12 nvjpeg runtime. Install one of: (a) pip install nvidia-nvjpeg-cu12, (b) apt install libnvjpeg-12-6 (Debian/Ubuntu), (c) use the CPU wheel: pip install sparrow-engine. Override the search path with SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/abs/path/libnvjpeg.so.12."
                    )
                }
            }
            Self::IncompatibleMajor { found, expected } => write!(
                f,
                "libnvjpeg major version {found} found; sparrow-engine-gpu requires CUDA {expected}. Install the CUDA {expected} toolkit (or nvidia-nvjpeg-cu{expected} on Linux)."
            ),
            Self::SymbolMissing(name) => write!(
                f,
                "libnvjpeg loaded but missing symbol '{name}'; CUDA installation appears corrupt or pre-CUDA-12.0."
            ),
        }
    }
}

impl Error for NvjpegInitError {}

static NVJPEG: OnceLock<Result<NvjpegLib, NvjpegInitError>> = OnceLock::new();

pub fn nvjpeg() -> Result<&'static NvjpegLib, &'static NvjpegInitError> {
    match NVJPEG.get_or_init(load_nvjpeg) {
        Ok(lib) => Ok(lib),
        Err(err) => Err(err),
    }
}

fn load_nvjpeg() -> Result<NvjpegLib, NvjpegInitError> {
    let candidates = candidate_paths();
    let mut errors = Vec::new();
    let mut tried_paths = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        tried_paths.push(candidate.clone());
        // SAFETY: Loading a process-global CUDA runtime library is the intended
        // boundary for this sys crate. Symbols are resolved immediately below
        // and the Library is retained for process lifetime by OnceLock.
        match unsafe { libloading::Library::new(candidate.as_os_str()) } {
            Ok(lib) => {
                return NvjpegLib::from_library(lib).map_err(|err| match err {
                    NvjpegInitError::LibraryNotFound { dlerror, .. } => {
                        NvjpegInitError::LibraryNotFound {
                            dlerror,
                            tried_paths: vec![candidate],
                        }
                    }
                    other => other,
                });
            }
            Err(err) => errors.push(format!("{}: {err}", candidate.display())),
        }
    }

    Err(NvjpegInitError::LibraryNotFound {
        dlerror: if errors.is_empty() {
            "no candidate paths generated".to_owned()
        } else {
            errors.join("; ")
        },
        tried_paths,
    })
}

fn candidate_paths() -> Vec<PathBuf> {
    if let Some(path) = env::var_os(NVJPEG_ENV_VAR).filter(|value| !value.as_os_str().is_empty()) {
        return vec![PathBuf::from(path)];
    }

    if cfg!(target_os = "windows") {
        let mut candidates = vec![PathBuf::from("nvjpeg64_12.dll")];
        candidates.extend(windows_cuda_known_paths());
        candidates
    } else {
        let mut candidates = vec![
            PathBuf::from("libnvjpeg.so.12"),
            PathBuf::from("libnvjpeg.so"),
        ];
        candidates.extend(linux_cuda_known_paths());
        candidates
    }
}

fn linux_cuda_known_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(entries) = fs::read_dir("/usr/local") else {
        return paths;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("cuda") {
            continue;
        }
        let candidate = entry.path().join("lib64").join("libnvjpeg.so.12");
        if candidate.is_file() {
            paths.push(candidate);
        }
    }
    paths.sort();
    paths
}

/// Windows nvjpeg candidate paths.
///
/// Order: explicit env vars (`CUDA_PATH`, `CUDA_PATH_V12_*`) first so a user
/// pointing at a specific toolkit overrides the default-install probe, then
/// the typical `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin\`
/// locations. The bare `nvjpeg64_12.dll` candidate above falls back to the
/// Windows DLL search path (app dir, System32, PATH).
fn windows_cuda_known_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Some(root) = env::var_os(var) {
            let candidate = PathBuf::from(&root).join("bin").join("nvjpeg64_12.dll");
            if candidate.is_file() {
                paths.push(candidate);
            }
        }
    }

    for (key, value) in env::vars_os() {
        let key_str = key.to_string_lossy();
        if !key_str.starts_with("CUDA_PATH_V12_") {
            continue;
        }
        let candidate = PathBuf::from(&value).join("bin").join("nvjpeg64_12.dll");
        if candidate.is_file() {
            paths.push(candidate);
        }
    }

    // Probe the default-install layout in case CUDA_PATH is unset.
    if let Ok(entries) =
        fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.starts_with("v12") {
                continue;
            }
            let candidate = entry.path().join("bin").join("nvjpeg64_12.dll");
            if candidate.is_file() {
                paths.push(candidate);
            }
        }
    }

    paths.sort();
    paths.dedup();
    paths
}

impl NvjpegLib {
    fn from_library(lib: libloading::Library) -> Result<Self, NvjpegInitError> {
        let nvjpegCreateSimple = load_symbol(&lib, "nvjpegCreateSimple")?;
        let nvjpegJpegStateCreate = load_symbol(&lib, "nvjpegJpegStateCreate")?;
        let nvjpegGetImageInfo = load_symbol(&lib, "nvjpegGetImageInfo")?;
        let nvjpegDecode = load_symbol(&lib, "nvjpegDecode")?;
        let nvjpegJpegStateDestroy = load_symbol(&lib, "nvjpegJpegStateDestroy")?;
        let nvjpegDestroy = load_symbol(&lib, "nvjpegDestroy")?;
        let nvjpegGetProperty = load_symbol(&lib, "nvjpegGetProperty")?;

        let loaded = Self {
            _lib: lib,
            nvjpegCreateSimple,
            nvjpegJpegStateCreate,
            nvjpegGetImageInfo,
            nvjpegDecode,
            nvjpegJpegStateDestroy,
            nvjpegDestroy,
            nvjpegGetProperty,
        };
        loaded.validate_major_version()?;
        Ok(loaded)
    }

    fn validate_major_version(&self) -> Result<(), NvjpegInitError> {
        let mut major: c_int = 0;
        // SAFETY: nvjpegGetProperty is a resolved function pointer from the
        // retained library; `major` is a valid out pointer.
        let status =
            unsafe { (self.nvjpegGetProperty)(libraryPropertyType_t_MAJOR_VERSION, &mut major) };
        if status != nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
            return Err(NvjpegInitError::LibraryNotFound {
                dlerror: format!("nvjpegGetProperty(MAJOR_VERSION) returned status {status}"),
                tried_paths: Vec::new(),
            });
        }

        let found = u32::try_from(major).unwrap_or_default();
        if found != EXPECTED_NVJPEG_MAJOR {
            return Err(NvjpegInitError::IncompatibleMajor {
                found,
                expected: EXPECTED_NVJPEG_MAJOR,
            });
        }
        Ok(())
    }
}

fn load_symbol<T>(lib: &libloading::Library, name: &'static str) -> Result<T, NvjpegInitError>
where
    T: Copy,
{
    let mut symbol = Vec::with_capacity(name.len() + 1);
    symbol.extend_from_slice(name.as_bytes());
    symbol.push(0);
    unsafe { lib.get::<T>(symbol.as_slice()) }
        .map(|loaded| *loaded)
        .map_err(|_| NvjpegInitError::SymbolMissing(name.to_owned()))
}
