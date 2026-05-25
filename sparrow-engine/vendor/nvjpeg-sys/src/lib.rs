//! Raw FFI Rust bindings to nvJPEG.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod bindings;
mod dynamic;

pub use bindings::*;
pub use dynamic::{nvjpeg, NvjpegInitError, NvjpegLib};

#[macro_export]
macro_rules! check {
    ($status:ident, $err:literal) => {
        if $status != 0 {
            Err(format!("{}. Error occured with code: {}", $err, $status))?
        }
    };
}

pub unsafe fn nvjpegCreateSimple(handle: *mut nvjpegHandle_t) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe { (lib.nvjpegCreateSimple)(handle) },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegJpegStateCreate(
    handle: nvjpegHandle_t,
    jpeg_handle: *mut nvjpegJpegState_t,
) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe { (lib.nvjpegJpegStateCreate)(handle, jpeg_handle) },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegGetImageInfo(
    handle: nvjpegHandle_t,
    data: *const std::os::raw::c_uchar,
    length: usize,
    nComponents: *mut std::os::raw::c_int,
    subsampling: *mut nvjpegChromaSubsampling_t,
    widths: *mut std::os::raw::c_int,
    heights: *mut std::os::raw::c_int,
) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe {
            (lib.nvjpegGetImageInfo)(
                handle,
                data,
                length,
                nComponents,
                subsampling,
                widths,
                heights,
            )
        },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegDecode(
    handle: nvjpegHandle_t,
    jpeg_handle: nvjpegJpegState_t,
    data: *const std::os::raw::c_uchar,
    length: usize,
    output_format: nvjpegOutputFormat_t,
    destination: *mut nvjpegImage_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe {
            (lib.nvjpegDecode)(
                handle,
                jpeg_handle,
                data,
                length,
                output_format,
                destination,
                stream,
            )
        },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegJpegStateDestroy(jpeg_handle: nvjpegJpegState_t) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe { (lib.nvjpegJpegStateDestroy)(jpeg_handle) },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegDestroy(handle: nvjpegHandle_t) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe { (lib.nvjpegDestroy)(handle) },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}

pub unsafe fn nvjpegGetProperty(
    type_: libraryPropertyType,
    value: *mut std::os::raw::c_int,
) -> nvjpegStatus_t {
    match nvjpeg() {
        Ok(lib) => unsafe { (lib.nvjpegGetProperty)(type_, value) },
        Err(_) => nvjpegStatus_t_NVJPEG_STATUS_NOT_INITIALIZED,
    }
}
