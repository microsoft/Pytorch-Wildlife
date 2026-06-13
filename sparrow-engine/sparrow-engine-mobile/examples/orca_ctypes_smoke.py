"""Generic-FFI ctypes smoke for the sparrow-engine mobile orca cascade (RP-25-FU-1).

Exercises the C ABI exactly as a native consumer (water-sparrow) would:
  engine_new -> load_pipeline_by_id("orca-cascade") -> run_pipeline(samples) ->
  read cascade result -> pipeline_result_free -> engine_free.

Env (all optional):
  SPE_MOBILE_LIB         path to libsparrow_engine.so (default: target/debug build)
  SPE_MOBILE_MODEL_DIR   model catalog dir with orca-cascade/pipeline.toml
  SPE_MOBILE_FIXTURES    fixtures dir (seg_000 .. with ecotype_audio.npy)
  LD_LIBRARY_PATH        must contain the x86_64 libLiteRt.so on host
"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

REPO = Path("/home/miao/repos/PW_refactor/Pytorch-Wildlife/sparrow-engine")
DEV = Path("/home/miao/repos/PW_refactor/sparrow-engine-dev")
LIB = Path(os.environ.get("SPE_MOBILE_LIB", REPO / "target/debug/libsparrow_engine.so"))
MODEL_DIR = Path(
    os.environ.get("SPE_MOBILE_MODEL_DIR", DEV / ".zenodo-staging/sparrow-engine-models-v0.6.0")
)
FIXTURES = Path(os.environ.get("SPE_MOBILE_FIXTURES", DEV / "bench-binaries/artifacts/fixtures"))
PIPELINE_ID = b"orca-cascade"


class CascadeSegment(ctypes.Structure):
    _fields_ = [
        ("start_s", ctypes.c_float),
        ("end_s", ctypes.c_float),
        ("detector_logit", ctypes.c_float),
        ("detector_probability", ctypes.c_float),
        ("is_detected", ctypes.c_uint8),
        ("stage2_ran", ctypes.c_uint8),
        ("stage2_argmax", ctypes.c_int32),
        ("stage2_confidence", ctypes.c_float),
    ]


class CascadeResult(ctypes.Structure):
    _fields_ = [
        ("pipeline_id", ctypes.c_void_p),
        ("data", ctypes.POINTER(CascadeSegment)),
        ("len", ctypes.c_size_t),
        ("num_stage2_classes", ctypes.c_size_t),
        ("stage2_probabilities", ctypes.POINTER(ctypes.c_float)),
        ("duration_s", ctypes.c_float),
        ("sample_rate", ctypes.c_uint32),
        ("processing_time_ms", ctypes.c_float),
    ]


def expected_argmax(seg_dir: Path) -> int:
    logits = json.loads((seg_dir / "expected_logits.json").read_text())["ecotype"]["fp32"]
    return max(range(len(logits)), key=logits.__getitem__)


def last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.sparrow_engine_last_error()
    if not ptr:
        return "<no last error>"
    return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")


def main() -> None:
    lib = ctypes.CDLL(str(LIB))
    lib.sparrow_engine_engine_new.argtypes = [ctypes.c_char_p]
    lib.sparrow_engine_engine_new.restype = ctypes.c_void_p
    lib.sparrow_engine_engine_free.argtypes = [ctypes.c_void_p]
    lib.sparrow_engine_load_pipeline_by_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.sparrow_engine_load_pipeline_by_id.restype = ctypes.c_int
    lib.sparrow_engine_run_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.sparrow_engine_run_pipeline.restype = ctypes.POINTER(CascadeResult)
    lib.sparrow_engine_pipeline_result_free.argtypes = [ctypes.POINTER(CascadeResult)]
    lib.sparrow_engine_last_error.argtypes = []
    lib.sparrow_engine_last_error.restype = ctypes.c_void_p

    config = json.dumps({"model_dir": str(MODEL_DIR), "intra_threads": 0}).encode("utf-8")
    engine = lib.sparrow_engine_engine_new(config)
    if not engine:
        raise RuntimeError(f"engine_new failed: {last_error(lib)}")

    try:
        if lib.sparrow_engine_load_pipeline_by_id(engine, PIPELINE_ID) != 0:
            raise RuntimeError(f"load_pipeline failed: {last_error(lib)}")

        matches = 0
        gated = 0
        for idx in range(10):
            seg = FIXTURES / f"seg_{idx:03d}"
            audio = np.load(seg / "ecotype_audio.npy").astype(np.float32).ravel()
            sr = int(np.load(seg / "ecotype_sample_rate.npy").ravel()[0])
            res_ptr = lib.sparrow_engine_run_pipeline(
                engine,
                PIPELINE_ID,
                audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                audio.size,
                sr,
                None,
            )
            if not res_ptr:
                raise RuntimeError(f"run_pipeline {seg.name} failed: {last_error(lib)}")
            try:
                res = res_ptr.contents
                if res.len < 1:
                    raise RuntimeError(f"{seg.name}: no windows returned")
                w = res.data[0]
                n = res.num_stage2_classes
                probs = [round(float(res.stage2_probabilities[c]), 6) for c in range(n)] if (
                    w.stage2_ran and res.stage2_probabilities
                ) else []
                exp = expected_argmax(seg)
                ok = w.stage2_argmax == exp
                matches += int(ok)
                gated += int(bool(w.stage2_ran))
                print(
                    f"{seg.name}: detector_logit={w.detector_logit:.6f} "
                    f"detector_prob={w.detector_probability:.6f} is_detected={w.is_detected} "
                    f"stage2_ran={w.stage2_ran} stage2_argmax={w.stage2_argmax} "
                    f"expected={exp} probs={probs}"
                )
            finally:
                lib.sparrow_engine_pipeline_result_free(res_ptr)

        # The detector gate skips stage 2 for non-orca windows (matching the
        # proven OrcaCascade); assert every window where stage 2 RAN got the
        # expected ecotype argmax. Gating itself is covered by the Rust parity test.
        print(f"CTYPES_SMOKE generic-FFI gated_argmax_matches={matches}/{gated} gated_segments={gated}/10")
        if gated < 1 or matches != gated:
            raise SystemExit(f"FAIL: {matches}/{gated} gated ecotype argmax matched expected")
    finally:
        lib.sparrow_engine_engine_free(engine)


if __name__ == "__main__":
    main()
