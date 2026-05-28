#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Opaque engine handle. Consumers must not inspect or dereference.
 */
typedef void SparrowEngine;

/**
 * Opaque model handle. Consumers must not inspect or dereference.
 */
typedef void SparrowEngineModel;

/**
 * Axis-aligned bounding box, normalized [0,1], xyxy format.
 */
typedef struct SparrowEngineBBox {
  float x_min;
  float y_min;
  float x_max;
  float y_max;
} SparrowEngineBBox;

/**
 * Single detection result. `label` pointer valid until `sparrow_engine_detections_free()`.
 */
typedef struct SparrowEngineDetection {
  struct SparrowEngineBBox bbox;
  const char *label;
  uint32_t label_id;
  float confidence;
} SparrowEngineDetection;

/**
 * Detection result set from a single `sparrow_engine_detect()` / `sparrow_engine_detect_raw()` call.
 */
typedef struct SparrowEngineDetections {
  const struct SparrowEngineDetection *data;
  uintptr_t len;
  uint32_t image_width;
  uint32_t image_height;
} SparrowEngineDetections;

/**
 * Detection inference options. Zero = use default.
 */
typedef struct SparrowEngineDetectOpts {
  float confidence_threshold;
  uint32_t max_detections;
} SparrowEngineDetectOpts;

/**
 * Pixel format code for raw image buffers.
 *
 * Valid values are 0 = RGB, 1 = RGBA, 2 = BGRA, 3 = BGR.
 */
typedef uint32_t SparrowEnginePixelFormat;

/**
 * A single image buffer for batch detection.
 */
typedef struct SparrowEngineImageBuffer {
  const uint8_t *data;
  uintptr_t len;
} SparrowEngineImageBuffer;

/**
 * Per-image callback contract for `sparrow_engine_detect_batch`.
 *
 * Called once per image after its detections are ready.
 *
 * # Contract (DO NOT BREAK — extension rules below)
 *
 * Signature: `extern "C" fn(image_index, detections, user_data)`.
 *
 * Arguments:
 * - `image_index`: 0-based index into the `images` slice originally
 *   passed to `sparrow_engine_detect_batch`. Monotonically non-decreasing across
 *   calls within a single `sparrow_engine_detect_batch` invocation (images are
 *   processed in input order within each batch chunk).
 * - `detections`: pointer to a `SparrowEngineDetections` struct owned by
 *   sparrow-engine-cpu. Valid **only** for the duration of this callback. The
 *   callee MUST NOT retain the pointer past return; retained copies
 *   become dangling. Copy fields out before returning if persistence
 *   is needed.
 * - `user_data`: the `user_data` argument passed to `sparrow_engine_detect_batch`.
 *   Opaque to sparrow-engine-cpu.
 *
 * # Extension rules (additive-only)
 *
 * Any Phase 3.5+ extension to this callback MUST be additive:
 * - New fields appended to `SparrowEngineDetections` (never reordered, renamed,
 *   or removed).
 * - New callback variants (e.g. `SparrowEngineDetectBatchCallbackV2`) added
 *   alongside the existing one; the original type stays wire-compatible.
 * - New out-of-band signals routed via a sibling callback pointer in a
 *   new opts struct, not by mutating this signature.
 *
 * This rule coordinates S5 (`sparrow-engine-cli` progress bar, which merged
 * first and authored this contract) with S6 (`sparrow-engine-python` progress
 * callback + `tracing` bridge, which extends additively). See
 * `docs/design/phase3.5/final_design.md` §4 S5 / §4 S6.
 */
typedef void (*SparrowEngineDetectBatchCallback)(uintptr_t image_index,
                                                 const struct SparrowEngineDetections *detections,
                                                 void *user_data);

/**
 * Single classification prediction.
 */
typedef struct SparrowEngineClassification {
  const char *label;
  uint32_t label_id;
  float confidence;
} SparrowEngineClassification;

/**
 * Classification result from a single `sparrow_engine_classify()` call.
 */
typedef struct SparrowEngineClassifyResult {
  const char *label;
  uint32_t label_id;
  float confidence;
  const struct SparrowEngineClassification *top_results;
  uintptr_t top_results_len;
  uint32_t image_width;
  uint32_t image_height;
  float processing_time_ms;
} SparrowEngineClassifyResult;

/**
 * Classification inference options. Zero = use default.
 */
typedef struct SparrowEngineClassifyOpts {
  uint32_t top_k;
} SparrowEngineClassifyOpts;

/**
 * A pipeline detection: detection + optional classification.
 */
typedef struct SparrowEnginePipelineDetection {
  struct SparrowEngineDetection detection;
  bool has_classification;
  struct SparrowEngineClassification classification;
} SparrowEnginePipelineDetection;

/**
 * Pipeline result from `sparrow_engine_run_pipeline()`.
 */
typedef struct SparrowEnginePipelineResult {
  const char *pipeline_id;
  const struct SparrowEnginePipelineDetection *data;
  uintptr_t len;
  uint32_t image_width;
  uint32_t image_height;
  float processing_time_ms;
} SparrowEnginePipelineResult;

/**
 * A single detected audio segment.
 */
typedef struct SparrowEngineAudioSegment {
  float start_time_s;
  float end_time_s;
  float confidence;
} SparrowEngineAudioSegment;

/**
 * Audio detection result from `sparrow_engine_detect_audio`.
 */
typedef struct SparrowEngineAudioResult {
  const struct SparrowEngineAudioSegment *data;
  uintptr_t len;
  float duration_s;
  uint32_t sample_rate;
  float processing_time_ms;
} SparrowEngineAudioResult;

/**
 * Audio detection options. Zero = use manifest default.
 */
typedef struct SparrowEngineAudioDetectOpts {
  float confidence_threshold;
  float segment_duration_s;
  float segment_stride_s;
} SparrowEngineAudioDetectOpts;

/**
 * V2 (Perch 2 + future multi-class classifiers): per-class entry for top-K output.
 * `label` is a borrowed pointer into the result's CString arena; valid until the
 * SparrowEngineAudioResult_v2 is freed via sparrow_engine_audio_result_v2_free.
 * `label` may be null when the model has no label for this index.
 */
typedef struct SparrowEngineAudioClass {
  uint32_t class_idx;
  const char *label;
  float probability;
} SparrowEngineAudioClass;

/**
 * V2 audio segment: same V1 fields plus a top-K classes array.
 * `classes` is a borrowed pointer into the result; valid for the lifetime of the result.
 */
typedef struct SparrowEngineAudioSegment_v2 {
  float start_time_s;
  float end_time_s;
  float confidence;
  const struct SparrowEngineAudioClass *classes;
  uintptr_t classes_len;
} SparrowEngineAudioSegment_v2;

/**
 * V2 audio detection result. Free with sparrow_engine_audio_result_v2_free.
 */
typedef struct SparrowEngineAudioResult_v2 {
  const struct SparrowEngineAudioSegment_v2 *data;
  uintptr_t len;
  float duration_s;
  uint32_t sample_rate;
  float processing_time_ms;
} SparrowEngineAudioResult_v2;

/**
 * Callback type for streaming audio detection.
 * Called once per segment that exceeds the confidence threshold.
 * `user_data` is passed through from the caller (opaque context pointer).
 */
typedef void (*SparrowEngineAudioSegmentCallback)(const struct SparrowEngineAudioSegment *segment,
                                                  void *user_data);

/**
 * Day/night result. Returned by value (small struct).
 */
typedef struct SparrowEngineDayNightResultC {
  /**
   * 0 = success, -1 = error (check `sparrow_engine_last_error`).
   */
  int32_t status;
  /**
   * 1 = day, 0 = night. Undefined if status != 0.
   */
  int32_t is_day;
  /**
   * Mean brightness [0,255]. -1.0 if status != 0.
   */
  float brightness;
} SparrowEngineDayNightResultC;

/**
 * Model verification result. Freed by `sparrow_engine_verify_result_free`.
 */
typedef struct SparrowEngineVerifyResultC {
  /**
   * 0=Ok, 1=NoChecksum, 2=SizeMismatch, 3=ChecksumMismatch.
   * On error, `sparrow_engine_verify_model` returns null (call `sparrow_engine_last_error` for detail).
   */
  int32_t status;
  /**
   * Detail message. Null if status=0 or status=1.
   */
  char *detail;
} SparrowEngineVerifyResultC;

#define SPARROW_ENGINE_PIXEL_FORMAT_RGB 0

#define SPARROW_ENGINE_PIXEL_FORMAT_RGBA 1

#define SPARROW_ENGINE_PIXEL_FORMAT_BGRA 2

#define SPARROW_ENGINE_PIXEL_FORMAT_BGR 3

/**
 * Create a new engine from a JSON config string. Returns null on error.
 *
 * # Safety
 * `config_json` must be a valid, non-null, null-terminated UTF-8 string.
 */
SparrowEngine *sparrow_engine_engine_new(const char *config_json);

/**
 * Free an engine. All models loaded through this engine become invalid.
 *
 * # Safety
 * `engine` must be a pointer returned by `sparrow_engine_engine_new`, or null (no-op).
 */
void sparrow_engine_engine_free(SparrowEngine *engine);

/**
 * Load a model from a TOML manifest file path. Returns null on error.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `manifest_path` must be a valid, non-null, null-terminated UTF-8 string.
 */
SparrowEngineModel *sparrow_engine_load_model(SparrowEngine *engine, const char *manifest_path);

/**
 * Load a model by its ID from the model directory. Returns null on error.
 *
 * Idempotent / lazy: if the model is already loaded, returns a fresh handle
 * to the existing ORT session (no re-creation). Mirrors the lazy-load
 * contract exposed by the HTTP `/v1/models/load` and `/v1/detect`+`/v1/classify`+
 * `/v1/audio` endpoints and the `sparrow-engine-python` package — calling twice with
 * the same id does not invalidate previously-issued handles.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
 */
SparrowEngineModel *sparrow_engine_load_model_by_id(SparrowEngine *engine, const char *model_id);

/**
 * Unload a model and free its resources.
 *
 * # Safety
 * `model` must be a pointer returned by `sparrow_engine_load_model` / `sparrow_engine_load_model_by_id`, or null.
 */
void sparrow_engine_unload_model(SparrowEngineModel *model);

/**
 * Load a pipeline from a TOML manifest. Returns 0 on success, -1 on error.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `manifest_path` must be a valid, non-null, null-terminated UTF-8 string.
 */
int32_t sparrow_engine_load_pipeline(SparrowEngine *engine, const char *manifest_path);

/**
 * Load a pipeline by its ID from the model directory. Returns 0 on success, -1 on error.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
 */
int32_t sparrow_engine_load_pipeline_by_id(SparrowEngine *engine, const char *pipeline_id);

/**
 * Unload a pipeline by ID. Returns 0 on success, -1 on error.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
 */
int32_t sparrow_engine_unload_pipeline(SparrowEngine *engine, const char *pipeline_id);

/**
 * Run detection on an encoded image buffer (JPEG/PNG). Returns null on error.
 *
 * # Safety
 * - `model` must be a valid model pointer.
 * - `image` must point to `len` bytes of encoded image data.
 * - `opts` may be null (use defaults).
 */
struct SparrowEngineDetections *sparrow_engine_detect(const SparrowEngineModel *model,
                                                      const uint8_t *image,
                                                      uintptr_t len,
                                                      const struct SparrowEngineDetectOpts *opts);

/**
 * Run detection on a raw pixel buffer. Returns null on error.
 *
 * # Safety
 * - `model` must be a valid model pointer.
 * - `pixels` must point to `h * stride` bytes of pixel data.
 * - `opts` may be null (use defaults).
 */
struct SparrowEngineDetections *sparrow_engine_detect_raw(const SparrowEngineModel *model,
                                                          const uint8_t *pixels,
                                                          uint32_t w,
                                                          uint32_t h,
                                                          uint32_t stride,
                                                          SparrowEnginePixelFormat format,
                                                          const struct SparrowEngineDetectOpts *opts);

/**
 * Run batch detection on multiple encoded images. Returns 0 on success, -1 on error.
 * The callback is invoked per-image with detection results.
 * Images are processed in batches (default 4) for higher GPU throughput.
 *
 * See `SparrowEngineDetectBatchCallback` for the per-image callback contract and
 * additive-extension rules (both enforced by the Phase 3.5 S5/S6 handoff).
 *
 * # Safety
 * - `model` must be a valid model pointer.
 * - `images` must point to `image_count` `SparrowEngineImageBuffer` structs.
 * - `opts` may be null (use defaults).
 * - `callback` must be valid for the duration of this call.
 * - `batch_size` of 0 uses the default (4).
 */
int32_t sparrow_engine_detect_batch(const SparrowEngineModel *model,
                                    const struct SparrowEngineImageBuffer *images,
                                    uintptr_t image_count,
                                    const struct SparrowEngineDetectOpts *opts,
                                    uintptr_t batch_size,
                                    SparrowEngineDetectBatchCallback callback,
                                    void *user_data);

/**
 * Run classification on an encoded image buffer (JPEG/PNG). Returns null on error.
 *
 * # Safety
 * - `model` must be a valid model pointer.
 * - `image` must point to `len` bytes of encoded image data.
 * - `opts` may be null (use defaults).
 */
struct SparrowEngineClassifyResult *sparrow_engine_classify(const SparrowEngineModel *model,
                                                            const uint8_t *image,
                                                            uintptr_t len,
                                                            const struct SparrowEngineClassifyOpts *opts);

/**
 * Run a pipeline (detect → classify) on an encoded image. Returns null on error.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
 * - `image` must point to `len` bytes of encoded image data.
 * - `detect_opts` and `classify_opts` may be null (use defaults).
 */
struct SparrowEnginePipelineResult *sparrow_engine_run_pipeline(const SparrowEngine *engine,
                                                                const char *pipeline_id,
                                                                const uint8_t *image,
                                                                uintptr_t len,
                                                                const struct SparrowEngineDetectOpts *detect_opts,
                                                                const struct SparrowEngineClassifyOpts *classify_opts);

/**
 * Run audio detection on a WAV file. Returns null on error.
 *
 * # Safety
 * - `model` must be a valid model pointer (audio model with mel spectrogram preprocessing).
 * - `audio_path` must be a valid, non-null, null-terminated UTF-8 path to a WAV file.
 * - `opts` may be null (use defaults).
 */
struct SparrowEngineAudioResult *sparrow_engine_detect_audio(const SparrowEngineModel *model,
                                                             const char *audio_path,
                                                             const struct SparrowEngineAudioDetectOpts *opts);

/**
 * Run audio detection on a WAV file with V2 top-K classes. Returns null on error.
 *
 * # Safety
 * - `model` must be a valid model pointer (audio model with mel spectrogram preprocessing).
 * - `audio_path` must be a valid, non-null, null-terminated UTF-8 path to a WAV file.
 * - `opts` may be null (use defaults).
 */
struct SparrowEngineAudioResult_v2 *sparrow_engine_detect_audio_v2(const SparrowEngineModel *model,
                                                                   const char *audio_path,
                                                                   const struct SparrowEngineAudioDetectOpts *opts);

/**
 * Run audio detection with per-segment streaming callback.
 *
 * CPU callback cadence is per-segment: the callback is invoked immediately
 * after each detected segment is produced, allowing the caller to update
 * progress incrementally.
 *
 * Note: the GPU flavor of this symbol fires callbacks post-detect (after the
 * full chunk loop completes) rather than per-segment. Callers writing
 * flavor-agnostic UI code should not assume per-segment cadence; see the
 * matching doc-comment in `sparrow-engine-gpu/src/ffi.rs`.
 *
 * Returns the complete result (same as `sparrow_engine_detect_audio`).
 *
 * # Safety
 * - `model` must be a valid model pointer.
 * - `audio_path` must be a valid, non-null, null-terminated UTF-8 path.
 * - `opts` may be null (use defaults).
 * - `callback` must be a valid function pointer for the duration of this call.
 * - `user_data` is passed through to the callback unchanged.
 */
struct SparrowEngineAudioResult *sparrow_engine_detect_audio_streaming(const SparrowEngineModel *model,
                                                                       const char *audio_path,
                                                                       const struct SparrowEngineAudioDetectOpts *opts,
                                                                       SparrowEngineAudioSegmentCallback callback,
                                                                       void *user_data);

/**
 * Free a `SparrowEngineAudioResult` returned by `sparrow_engine_detect_audio`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_detect_audio`, or null.
 */
void sparrow_engine_audio_result_free(struct SparrowEngineAudioResult *ptr);

/**
 * Free a `SparrowEngineAudioResult_v2` returned by `sparrow_engine_detect_audio_v2`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_detect_audio_v2`, or null.
 */
void sparrow_engine_audio_result_v2_free(struct SparrowEngineAudioResult_v2 *ptr);

/**
 * Free a `SparrowEngineDetections` returned by `sparrow_engine_detect` or `sparrow_engine_detect_raw`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_detect`/`sparrow_engine_detect_raw`, or null.
 */
void sparrow_engine_detections_free(struct SparrowEngineDetections *ptr);

/**
 * Free a `SparrowEngineClassifyResult` returned by `sparrow_engine_classify`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_classify`, or null.
 */
void sparrow_engine_classify_result_free(struct SparrowEngineClassifyResult *ptr);

/**
 * Free a `SparrowEnginePipelineResult` returned by `sparrow_engine_run_pipeline`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_run_pipeline`, or null.
 */
void sparrow_engine_pipeline_result_free(struct SparrowEnginePipelineResult *ptr);

/**
 * Free a string returned by `sparrow_engine_list_models` or `sparrow_engine_health`.
 *
 * # Safety
 * `ptr` must be a pointer returned by a sparrow-engine function that allocates strings, or null.
 */
void sparrow_engine_free_string(char *ptr);

/**
 * List loaded models as a JSON string. Returns null on error.
 * Caller must free with `sparrow_engine_free_string`.
 *
 * # Safety
 * `engine` must be a valid engine pointer.
 */
char *sparrow_engine_list_models(const SparrowEngine *engine);

/**
 * Return engine health as a JSON string. Returns null on error.
 * Caller must free with `sparrow_engine_free_string`.
 *
 * # Safety
 * `engine` must be a valid engine pointer.
 */
char *sparrow_engine_health(const SparrowEngine *engine);

/**
 * Return the last error message for this thread, or null if no error.
 * The returned pointer is valid until the next FFI call on the same thread.
 *
 * # Safety
 * Thread-safe. Returned pointer must not be freed by the caller.
 */
const char *sparrow_engine_last_error(void);

/**
 * Compute SHA-256 hash of a file. Returns hex string or null on error.
 * Caller must free with `sparrow_engine_hash_result_free`.
 *
 * # Safety
 * `path` must be a valid, non-null, null-terminated UTF-8 string.
 */
char *sparrow_engine_hash_file(const char *path);

/**
 * Free a hash string returned by `sparrow_engine_hash_file`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_hash_file`, or null.
 */
void sparrow_engine_hash_result_free(char *ptr);

/**
 * Classify image as day or night. Returns result by value.
 * On error: status=-1, check `sparrow_engine_last_error`.
 *
 * # Safety
 * `image` must point to `len` bytes of encoded image data (JPEG/PNG).
 */
struct SparrowEngineDayNightResultC sparrow_engine_day_night(const uint8_t *image, uintptr_t len);

/**
 * Compute mean image brightness [0,255]. Returns -1.0 on error.
 *
 * # Safety
 * `image` must point to `len` bytes of encoded image data (JPEG/PNG).
 */
float sparrow_engine_image_brightness(const uint8_t *image, uintptr_t len);

/**
 * Verify a model's ONNX file against manifest checksums.
 * Returns null on error. Caller must free with `sparrow_engine_verify_result_free`.
 *
 * # Safety
 * `model_dir` and `model_id` must be valid, non-null, null-terminated UTF-8 strings.
 */
struct SparrowEngineVerifyResultC *sparrow_engine_verify_model(const char *model_dir,
                                                               const char *model_id);

/**
 * Free a `SparrowEngineVerifyResultC` returned by `sparrow_engine_verify_model` or `sparrow_engine_engine_verify_model`.
 *
 * # Safety
 * `ptr` must be a pointer returned by `sparrow_engine_verify_model`/`sparrow_engine_engine_verify_model`, or null.
 */
void sparrow_engine_verify_result_free(struct SparrowEngineVerifyResultC *ptr);

/**
 * Verify a model using the engine's model directory. Returns null on error.
 * Caller must free with `sparrow_engine_verify_result_free`.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
 */
struct SparrowEngineVerifyResultC *sparrow_engine_engine_verify_model(const SparrowEngine *engine,
                                                                      const char *model_id);

/**
 * Get model info as JSON string. Searches loaded models first, then disk.
 * Returns null on error. Caller must free with `sparrow_engine_free_string`.
 *
 * # Safety
 * - `engine` must be a valid engine pointer.
 * - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
 */
char *sparrow_engine_engine_model_info(const SparrowEngine *engine, const char *model_id);

/**
 * List all available models (on disk) as JSON array with extended info.
 * Includes version, description, and checksums. Returns null on error.
 * Caller must free with `sparrow_engine_free_string`.
 *
 * # Safety
 * `engine` must be a valid engine pointer.
 */
char *sparrow_engine_engine_list_models_extended(const SparrowEngine *engine);

/**
 * Returns a pointer to a static, null-terminated UTF-8 string with the
 * sparrow-engine-cpu crate version (matches `[package].version` in
 * `sparrow-engine-cpu/Cargo.toml`). Caller MUST NOT free.
 *
 * Phase D B-12: useful for installer / Studio Local / brew `test do` smoke
 * tests — a zero-arg, zero-allocation entry point that proves DLL load +
 * symbol resolution without spinning up an engine. Byte-for-byte mirror of
 * the GPU FFI surface (G5 acceptance gate enforces CPU/GPU symbol parity).
 *
 * # Safety
 * Thread-safe. Returned pointer is valid for the lifetime of the process.
 */
const char *sparrow_engine_version(void);
