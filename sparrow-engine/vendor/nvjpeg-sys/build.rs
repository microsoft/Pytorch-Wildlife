fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rustc-link-lib=dylib=nvjpeg");
}
