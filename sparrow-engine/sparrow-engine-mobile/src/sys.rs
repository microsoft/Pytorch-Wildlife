//! Raw FFI bindings to the Google AI Edge LiteRT C API.
//!
//! Generated at build time by `build.rs` from headers vendored under
//! `sparrow-engine-mobile/vendor/litert`.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/litert_bindings.rs"));
