//! Raw FFI bindings to the Google AI Edge LiteRT C API.
//!
//! Generated at build time by `build.rs` from headers vendored under
//! `sparrow-engine-mobile/vendor/litert`.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unnecessary_transmutes)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::transmute_int_to_bool)]
#![allow(clippy::useless_transmute)]

include!(concat!(env!("OUT_DIR"), "/litert_bindings.rs"));
