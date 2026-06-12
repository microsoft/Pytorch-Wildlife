/* bindgen entry point for `spe-bench-tflite`.
 *
 * Pulls in only the headers we directly call. Transitive includes pull in
 * the rest of the API surface (model_types, layout, tensor_buffer_types, etc.)
 * automatically via the litert/c/*.h cross-references.
 */
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_layout.h"
/* RP-41: CPU thread-control surface. litert_cpu_options.h declares the Lrt*
 * CpuOptions API; litert_opaque_options.h declares the attach functions. These
 * symbols are exported by the custom libLiteRt.so source build (the stock
 * ai-edge-litert wheel hides them, pinning inference to 1 thread). */
#include "litert/c/litert_opaque_options.h"
#include "litert/c/options/litert_cpu_options.h"
