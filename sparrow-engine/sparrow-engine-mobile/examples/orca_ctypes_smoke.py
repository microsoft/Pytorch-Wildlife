import ctypes
import json
import os
from pathlib import Path

import numpy as np


REPO = Path("/home/miao/repos/PW_refactor/Pytorch-Wildlife/sparrow-engine")
DEFAULT_ARTIFACTS = Path("/home/miao/repos/PW_refactor/sparrow-engine-dev/bench-binaries/artifacts")
LIB = Path(os.environ.get("SPE_MOBILE_LIB", REPO / "target/debug/libsparrow_engine.so"))
MODELS = Path(os.environ.get("SPE_MOBILE_MODELS", DEFAULT_ARTIFACTS))
FIXTURES = Path(os.environ.get("SPE_MOBILE_FIXTURES", DEFAULT_ARTIFACTS / "fixtures"))
DETECTOR = Path(os.environ.get("SPE_MOBILE_DETECTOR", MODELS / "orca-detector-fp32.tflite"))
ECOTYPE = Path(os.environ.get("SPE_MOBILE_ECOTYPE", MODELS / "orca-ecotype-melinput-fp32.tflite"))


class SparrowOrcaResult(ctypes.Structure):
    _fields_ = [
        ("detector_logit", ctypes.c_float),
        ("detector_probability", ctypes.c_float),
        ("is_orca", ctypes.c_uint8),
        ("ecotype_ran", ctypes.c_uint8),
        ("ecotype_argmax", ctypes.c_int32),
        ("ecotype_probabilities", ctypes.c_float * 5),
    ]


def expected_argmax(seg_dir: Path) -> int:
    data = json.loads((seg_dir / "expected_logits.json").read_text())
    logits = data["ecotype"]["fp32"]
    return max(range(len(logits)), key=logits.__getitem__)


def last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.sparrow_engine_orca_last_error()
    if not ptr:
        return "<no last error>"
    return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")


def main() -> None:
    lib = ctypes.CDLL(str(LIB))
    lib.sparrow_engine_orca_cascade_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.sparrow_engine_orca_cascade_new.restype = ctypes.c_void_p
    lib.sparrow_engine_orca_cascade_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(SparrowOrcaResult),
    ]
    lib.sparrow_engine_orca_cascade_run.restype = ctypes.c_int
    lib.sparrow_engine_orca_result_init.argtypes = [ctypes.POINTER(SparrowOrcaResult)]
    lib.sparrow_engine_orca_result_init.restype = ctypes.c_int
    lib.sparrow_engine_orca_cascade_free.argtypes = [ctypes.c_void_p]
    lib.sparrow_engine_orca_cascade_free.restype = None
    lib.sparrow_engine_orca_last_error.argtypes = []
    lib.sparrow_engine_orca_last_error.restype = ctypes.c_void_p

    handle = lib.sparrow_engine_orca_cascade_new(
        str(DETECTOR).encode("utf-8"),
        str(ECOTYPE).encode("utf-8"),
        0,
    )
    if not handle:
        raise RuntimeError(f"cascade_new failed: {last_error(lib)}")

    try:
        gated_matches = 0
        gated_count = 0
        total_returned_matches = 0
        for idx in range(10):
            seg = FIXTURES / f"seg_{idx:03d}"
            audio = np.load(seg / "ecotype_audio.npy").astype(np.float32).ravel()
            sample_rate = int(np.load(seg / "ecotype_sample_rate.npy").ravel()[0])
            out = SparrowOrcaResult()
            rc = lib.sparrow_engine_orca_result_init(ctypes.byref(out))
            if rc != 0:
                raise RuntimeError(f"result_init failed: {last_error(lib)}")
            rc = lib.sparrow_engine_orca_cascade_run(
                handle,
                audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                audio.size,
                sample_rate,
                ctypes.byref(out),
            )
            if rc != 0:
                raise RuntimeError(f"cascade_run {seg.name} failed: {last_error(lib)}")

            expected = expected_argmax(seg)
            returned_match = out.ecotype_argmax == expected
            total_returned_matches += int(returned_match)
            if out.ecotype_ran:
                gated_count += 1
                gated_matches += int(returned_match)
            print(
                f"{seg.name}: detector_logit={out.detector_logit:.6f} "
                f"detector_probability={out.detector_probability:.6f} "
                f"is_orca={out.is_orca} ecotype_ran={out.ecotype_ran} "
                f"ecotype_argmax={out.ecotype_argmax} expected_argmax={expected} "
                f"probabilities={[round(float(x), 6) for x in out.ecotype_probabilities]}"
            )

        print(
            f"CTYPES_SMOKE returned_argmax_matches={total_returned_matches}/10 "
            f"gated_argmax_matches={gated_matches}/{gated_count} "
            f"gated_segments={gated_count}/10"
        )
    finally:
        lib.sparrow_engine_orca_cascade_free(handle)


if __name__ == "__main__":
    main()
