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
 * Opaque engine handle. Consumers must not inspect or dereference.
 */
typedef void SparrowEngine;

/**
 * Opaque model handle returned by `sparrow_engine_load_model_by_id`.
 */
typedef void SparrowEngineModel;

/**
 * Detection bounding box, normalized `[0, 1]`.
 */
typedef struct SparrowEngineBBox {
  float x_min;
  float y_min;
  float x_max;
  float y_max;
} SparrowEngineBBox;

/**
 * One image detection.
 */
typedef struct SparrowEngineDetection {
  struct SparrowEngineBBox bbox;
  const char *label;
  uint32_t label_id;
  float confidence;
} SparrowEngineDetection;

/**
 * Image detection output (image inference is deferred to RP-42; see
 * `sparrow_engine_detect`).
 */
typedef struct SparrowEngineDetections {
  const struct SparrowEngineDetection *data;
  uintptr_t len;
  uint32_t image_width;
  uint32_t image_height;
} SparrowEngineDetections;

/**
 * Image detection options.
 */
typedef struct SparrowEngineDetectOpts {
  float confidence_threshold;
  uint32_t max_detections;
} SparrowEngineDetectOpts;

/**
 * One image classification.
 */
typedef struct SparrowEngineClassification {
  const char *label;
  uint32_t label_id;
  float confidence;
} SparrowEngineClassification;

/**
 * Image classification output (deferred to RP-42; see `sparrow_engine_classify`).
 */
typedef struct SparrowEngineClassifyResult {
  const struct SparrowEngineClassification *data;
  uintptr_t len;
  uint32_t image_width;
  uint32_t image_height;
  float processing_time_ms;
} SparrowEngineClassifyResult;

/**
 * Image classification options.
 */
typedef struct SparrowEngineClassifyOpts {
  uint32_t top_k;
} SparrowEngineClassifyOpts;

/**
 * One detected audio segment (single-model `sparrow_engine_detect_audio`).
 */
typedef struct SparrowEngineAudioSegment {
  float start_time_s;
  float end_time_s;
  float confidence;
} SparrowEngineAudioSegment;

/**
 * Single-model audio detection output.
 */
typedef struct SparrowEngineAudioResult {
  const struct SparrowEngineAudioSegment *data;
  uintptr_t len;
  float duration_s;
  uint32_t sample_rate;
  float processing_time_ms;
} SparrowEngineAudioResult;

/**
 * Single-model audio detection options. A `NaN` field means "use the manifest
 * default" (C has no `Option`).
 */
typedef struct SparrowEngineAudioDetectOpts {
  float confidence_threshold;
  float segment_duration_s;
  float segment_stride_s;
} SparrowEngineAudioDetectOpts;

/**
 * One audio-cascade window result.
 */
typedef struct SparrowEngineCascadeSegment {
  float start_s;
  float end_s;
  float detector_logit;
  float detector_probability;
  /**
   * 1 if stage 1 fired (`detector_probability >= threshold`), else 0.
   */
  uint8_t is_detected;
  /**
   * 1 if stage 2 (classifier) ran, else 0.
   */
  uint8_t stage2_ran;
  /**
   * Stage-2 argmax class index, or -1 when stage 2 did not run.
   */
  int32_t stage2_argmax;
  /**
   * Stage-2 top probability, or 0 when stage 2 did not run.
   */
  float stage2_confidence;
} SparrowEngineCascadeSegment;

/**
 * Audio-cascade output. `stage2_probabilities` is a flat row-major buffer of
 * `len * num_stage2_classes` values (segment `i`, class `c` lives at
 * `stage2_probabilities[i * num_stage2_classes + c]`); rows where stage 2 did
 * not run are all-zero.
 */
typedef struct SparrowEngineCascadeResult {
  const char *pipeline_id;
  const struct SparrowEngineCascadeSegment *data;
  uintptr_t len;
  uintptr_t num_stage2_classes;
  const float *stage2_probabilities;
  float duration_s;
  uint32_t sample_rate;
  float processing_time_ms;
} SparrowEngineCascadeResult;

/**
 * Audio-cascade options. A `NaN` field means "use the pipeline default".
 */
typedef struct SparrowEngineCascadeOpts {
  float window_sec;
  float overlap_sec;
  float detector_threshold;
} SparrowEngineCascadeOpts;

/**
 * Create an engine from a JSON config string: `{"model_dir": "...",
 * "intra_threads": 4}`. `intra_threads` is the LiteRT CPU thread count
 * (defaults to 4 — the Pi Zero 2W validated setting); `0` = LiteRT default.
 * Returns null on error; call `sparrow_engine_last_error` for details.
 *
 * # Safety
 * `config_json` must be a valid, non-null, null-terminated UTF-8 string.
 */
SparrowEngine *sparrow_engine_engine_new(const char *config_json);

/**
 * Free an engine. Null-safe. Each non-null engine must be freed exactly once.
 *
 * # Safety
 * `engine` must be a pointer returned by `sparrow_engine_engine_new`, or null.
 * Like every engine call, this must run on the thread that created the engine
 * (single-threaded contract — see the crate-level threading note); freeing from
 * another thread while the owner thread is mid-call is undefined behaviour.
 */
void sparrow_engine_engine_free(SparrowEngine *engine);

/**
 * Return the last error message for this thread, or null if none. The pointer
 * is valid until the next FFI call on the same thread; do not free it.
 *
 * # Safety
 * Thread-safe with respect to other threads' last-error state.
 */
const char *sparrow_engine_last_error(void);

/**
 * Free a string returned by the engine (e.g. `sparrow_engine_list_models`).
 * Null-safe.
 *
 * # Safety
 * `ptr` must be a string returned by an engine FFI function, or null, and freed
 * exactly once.
 */
void sparrow_engine_free_string(char *ptr);

/**
 * Return the engine version (static; do not free).
 */
const char *sparrow_engine_version(void);

/**
 * Load a model by catalog id. Returns null on error.
 *
 * # Safety
 * `engine` must be a valid engine pointer; `model_id` a valid C string.
 */
SparrowEngineModel *sparrow_engine_load_model_by_id(SparrowEngine *engine, const char *model_id);

/**
 * Unload the model this handle refers to and free the handle. Null-safe.
 *
 * # Safety
 * `model` must be a pointer returned by `sparrow_engine_load_model_by_id`, or
 * null, and freed exactly once. Must run on the engine's owner thread
 * (single-threaded contract — see the crate-level threading note).
 */
void sparrow_engine_unload_model(SparrowEngineModel *model);

/**
 * Return a JSON array of available models in the model directory. Caller frees
 * with `sparrow_engine_free_string`. Returns null on error.
 *
 * # Safety
 * `engine` must be a valid engine pointer.
 */
char *sparrow_engine_list_models(const SparrowEngine *engine);

/**
 * Run single-shot image detection over an encoded image buffer (JPEG/PNG).
 * Returns null on error; call `sparrow_engine_last_error` for details. Free the
 * result with `sparrow_engine_detections_free`.
 *
 * # Safety
 * `model` must be a valid model pointer; `image` must point to `len` readable
 * bytes; `opts` a valid pointer or null.
 */
struct SparrowEngineDetections *sparrow_engine_detect(const SparrowEngineModel *model,
                                                      const uint8_t *image,
                                                      uintptr_t len,
                                                      const struct SparrowEngineDetectOpts *opts);

/**
 * Image classification — not yet available on the mobile flavor (no `.tflite`
 * classifier onboarded). Image *detection* (`sparrow_engine_detect`) is
 * available as of RP-42. Always returns null with a clear last-error.
 *
 * # Safety
 * `model` must be a valid model pointer.
 */
struct SparrowEngineClassifyResult *sparrow_engine_classify(const SparrowEngineModel *_model,
                                                            const uint8_t *_image,
                                                            uintptr_t _len,
                                                            const struct SparrowEngineClassifyOpts *_opts);

/**
 * Free a detections result. Null-safe.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_detect`, or null.
 */
void sparrow_engine_detections_free(struct SparrowEngineDetections *ptr);

/**
 * Free a classify result. Null-safe.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_classify`, or null.
 */
void sparrow_engine_classify_result_free(struct SparrowEngineClassifyResult *ptr);

/**
 * Run single-model audio detection over a WAV file. Returns null on error.
 *
 * # Safety
 * `model` must be a valid model pointer; `audio_path` a valid C string; `opts`
 * a valid pointer or null.
 */
struct SparrowEngineAudioResult *sparrow_engine_detect_audio(const SparrowEngineModel *model,
                                                             const char *audio_path,
                                                             const struct SparrowEngineAudioDetectOpts *opts);

/**
 * Free an audio result. Null-safe.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_detect_audio`, or null.
 */
void sparrow_engine_audio_result_free(struct SparrowEngineAudioResult *ptr);

/**
 * Load an audio-cascade pipeline by catalog id. Returns 0 on success, -1 on
 * error (call `sparrow_engine_last_error`).
 *
 * # Safety
 * `engine` must be a valid engine pointer; `pipeline_id` a valid C string.
 */
int32_t sparrow_engine_load_pipeline_by_id(SparrowEngine *engine, const char *pipeline_id);

/**
 * Run a loaded audio-cascade pipeline over raw mono `f32` samples. Returns null
 * on error.
 *
 * # Safety
 * `engine` must be valid; `pipeline_id` a valid C string; `samples` must point
 * to `n_samples` finite `f32`; `opts` a valid pointer or null.
 */
struct SparrowEngineCascadeResult *sparrow_engine_run_pipeline(const SparrowEngine *engine,
                                                               const char *pipeline_id,
                                                               const float *samples,
                                                               uintptr_t n_samples,
                                                               uint32_t sample_rate,
                                                               const struct SparrowEngineCascadeOpts *opts);

/**
 * Unload a pipeline by id (its stage models stay loaded). Returns 0 / -1.
 *
 * # Safety
 * `engine` must be valid; `pipeline_id` a valid C string.
 */
int32_t sparrow_engine_unload_pipeline(SparrowEngine *engine, const char *pipeline_id);

/**
 * Free a cascade result. Null-safe.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_run_pipeline`, or null.
 */
void sparrow_engine_pipeline_result_free(struct SparrowEngineCascadeResult *ptr);
