# C library (`libsparrow_engine`)

Sparrow Engine has a C interface, so any language that can call C (C, C++, C#, Go, Rust, Swift, ...) can run the model zoo.
[`detect.c`](detect.c) loads a detector by id and prints the detections for one image.

## Build the library

Sparrow Engine releases ship the `spe` binary and ONNX Runtime, but not yet the C library or header, so build them from source (needs a Rust toolchain):

```bash
git clone https://github.com/microsoft/SPARROW-Engine
cd SPARROW-Engine/sparrow-engine
cargo build -p sparrow-engine-cpu --release --features ffi
# library: sparrow-engine/target/release/libsparrow_engine.so   (.dylib on macOS, sparrow_engine.dll on Windows)
# header:  sparrow-engine/include/sparrow_engine.h
```

The library loads ONNX Runtime at run time. Take `lib/libonnxruntime.so.*` from a
[Sparrow Engine release](https://github.com/microsoft/SPARROW-Engine/releases) tarball
(`sparrow-engine-cpu-<version>-linux-x86_64.tar.gz`) and point `ORT_DYLIB_PATH` at it.
Set it before running: with 0.1.32, a missing ONNX Runtime makes the first engine call hang instead of returning an error.

## Build and run the example

```bash
SPE=/path/to/SPARROW-Engine
gcc -std=c11 -O2 interfaces/c/detect.c \
    -I "$SPE/sparrow-engine/include" -L "$SPE/sparrow-engine/target/release" -lsparrow_engine -o detect

export LD_LIBRARY_PATH="$SPE/sparrow-engine/target/release"
export ORT_DYLIB_PATH=/path/to/sparrow-engine-cpu-0.1.32-linux-x86_64/lib/libonnxruntime.so.1.25.1

bash interfaces/cli/get_demo_data.sh      # demo images + MDV6-yolov10-c
./detect ~/.sparrow-engine/models MDV6-yolov10-c \
    data/data/data_example/images/train/alka_squirrels.seqdd0bf15a-2335-4bd0-a534-edadae3ffa99.frame102542.jpg 0.2
```

```
...frame102542.jpg: 640 x 640, 1 detections
  animal   0.44  box(px)  442  186  477  226
```

## The calls

| Call | Purpose |
|---|---|
| `sparrow_engine_engine_new(config_json)` | one engine per process; config `{"device":"auto","model_dir":"..."}` (`device` is `auto`, `cpu` or `cuda:N`) |
| `sparrow_engine_load_model_by_id(engine, id)` | load a model folder from `model_dir` |
| `sparrow_engine_detect(model, bytes, len, opts)` | encoded image bytes (JPEG, PNG, ...) in, detections out; boxes normalized to [0, 1] |
| `sparrow_engine_detections_free`, `sparrow_engine_unload_model`, `sparrow_engine_engine_free` | release each result and handle |
| `sparrow_engine_last_error()` | message for the last failed call (any call that returned `NULL`) |

The header also has classification, embeddings, detect-then-classify pipelines and audio.
C# P/Invoke notes are in the engine's [integration guide](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/integration/local.md).
