#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Orca detector manifest preprocessing constants.
 *
 * Source: `.zenodo-staging/orca-dclde2026-onboarding-workdir/
 * orca-detector-dclde2026-v1/manifest.toml`, `[preprocessing]`.
 */
#define ORCA_SAMPLE_RATE 24000

#define ORCA_SEGMENT_SAMPLES 72000

#define ORCA_N_FFT 1024

#define ORCA_HOP_LENGTH 128

#define ORCA_N_MELS 256

#define ORCA_FMIN 200.0

#define ORCA_FMAX 12000.0

#define ORCA_TOP_DB 80.0

#define ORCA_THRESHOLD 0.5

/**
 * Opaque orca cascade handle. Consumers must not inspect or dereference.
 */
typedef void SparrowOrcaCascade;

typedef struct SparrowOrcaResult {
  float detector_logit;
  float detector_probability;
  uint8_t is_orca;
  uint8_t ecotype_ran;
  /**
   * Ecotype class index, or -1 when the detector gate skipped ecotype.
   */
  int32_t ecotype_argmax;
  /**
   * Five ecotype probabilities. All zeros when ecotype did not run.
   */
  float ecotype_probabilities[5];
} SparrowOrcaResult;

/**
 * Create a focused orca cascade handle from detector and ecotype TFLite paths.
 *
 * Returns null on error. Call `sparrow_engine_orca_last_error` for details.
 *
 * # Safety
 * `detector_path` and `ecotype_path` must be valid, non-null, null-terminated
 * UTF-8 strings.
 */
SparrowOrcaCascade *sparrow_engine_orca_cascade_new(const char *detector_path,
                                                    const char *ecotype_path,
                                                    uintptr_t num_threads);

/**
 * Run one raw-audio segment through the orca cascade.
 *
 * Returns 0 on success and nonzero on error. On error, call
 * `sparrow_engine_orca_last_error` for details.
 * A handle is single-owner and not thread-safe: do not call `run` concurrently
 * on the same handle, and free each handle exactly once.
 *
 * # Safety
 * - `handle` must be a valid pointer returned by `sparrow_engine_orca_cascade_new`.
 * - `samples` must point to `n_samples` finite `f32` samples.
 * - `out` must be a valid writable pointer.
 */
int32_t sparrow_engine_orca_cascade_run(SparrowOrcaCascade *handle,
                                        const float *samples,
                                        uintptr_t n_samples,
                                        uint32_t sample_rate,
                                        struct SparrowOrcaResult *out);

/**
 * Initialize an orca result struct to its empty/default value.
 *
 * Returns 0 on success and nonzero on error. This helper lets ctypes callers
 * reset a stack-allocated result before reuse without knowing Rust defaults.
 *
 * # Safety
 * `out` must be a valid writable pointer.
 */
int32_t sparrow_engine_orca_result_init(struct SparrowOrcaResult *out);

/**
 * Free an orca cascade handle. Null-safe.
 *
 * # Safety
 * `handle` must be a pointer returned by `sparrow_engine_orca_cascade_new`, or null.
 * Each non-null handle must be freed exactly once.
 */
void sparrow_engine_orca_cascade_free(SparrowOrcaCascade *handle);

/**
 * Return the last orca FFI error message for this thread, or null if no error.
 * The returned pointer is valid until the next orca FFI call on the same thread.
 *
 * # Safety
 * Thread-safe. Returned pointer must not be freed by the caller.
 */
const char *sparrow_engine_orca_last_error(void);
