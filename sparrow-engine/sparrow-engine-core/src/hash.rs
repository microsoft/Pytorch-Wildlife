//! SHA-256 file hashing (chunked reads).

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use sparrow_engine_types::Result;

/// Compute SHA-256 hash of a file. Returns lowercase hex string (64 chars).
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_hex_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"hello world").unwrap();
        let hash = hash_file(&path).unwrap();
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"deterministic content").unwrap();
        let h1 = hash_file(&path).unwrap();
        let h2 = hash_file(&path).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_known_content() {
        // SHA-256 of empty file is well-known.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty");
        std::fs::File::create(&path).unwrap();
        let hash = hash_file(&path).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hash_file_not_found() {
        let result = hash_file(Path::new("/nonexistent/file.bin"));
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod phase_a_r1_hash {
    use super::*;
    use sha2::{Digest, Sha256};

    /// SHA-256 of the canonical FIPS PUB 180-2 test vector "abc".
    /// Bit-for-bit pin: any change to the chunked-read loop (boundary off-by-one,
    /// wrong digest, accidental update of zero-length tail) breaks this immediately.
    #[test]
    fn hash_known_vector_abc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc.bin");
        std::fs::write(&path, b"abc").unwrap();
        let got = hash_file(&path).unwrap();
        assert_eq!(
            got,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256(\"abc\") must match the FIPS 180-2 reference vector"
        );
    }

    /// File larger than the 65 536-byte internal read buffer (hash.rs:14).
    /// Drives the `loop { read; if n == 0 break; update; }` across multiple
    /// non-trivial chunks and confirms the streaming path matches a one-shot
    /// digest of the same bytes — guards against accumulation bugs (e.g.,
    /// `hasher.update(&buf[..n])` accidentally written as `&buf` would only
    /// mismatch when `n < buf.len()`, which never happens for small inputs).
    #[test]
    fn hash_large_file_matches_one_shot_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");
        // 200 KiB > 3 × 65 536 internal chunks. Use a deterministic non-zero
        // pattern so we'd notice a missed slice.
        let bytes: Vec<u8> = (0..200 * 1024).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();

        let got = hash_file(&path).unwrap();

        // One-shot reference: feed the entire buffer in a single `update()`.
        let mut h = Sha256::new();
        h.update(&bytes);
        let expected = format!("{:x}", h.finalize());

        assert_eq!(
            got, expected,
            "streaming hash must equal one-shot SHA-256 across buffer boundaries"
        );
    }

    /// Avalanche / sanity: a single-bit change in the input must change the
    /// digest. Catches `hasher.update(&[]); ` style bypass bugs that pass the
    /// length test but silently digest zero bytes.
    #[test]
    fn hash_differs_on_single_byte_change() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.bin");
        let b = dir.path().join("b.bin");
        std::fs::write(&a, b"hello world\x00").unwrap();
        std::fs::write(&b, b"hello world\x01").unwrap(); // last byte differs by 1 bit
        let ha = hash_file(&a).unwrap();
        let hb = hash_file(&b).unwrap();
        assert_ne!(
            ha, hb,
            "single-byte change must produce a different SHA-256 digest"
        );
    }
}
