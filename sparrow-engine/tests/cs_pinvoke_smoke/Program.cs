// Phase 3.8 Phase C Wave 5 — C# P/Invoke smoke test (acceptance gate G1).
//
// Loads libsparrow_engine.so via [DllImport("sparrow_engine")] and runs a single MDv6
// detection. The directory containing libsparrow_engine.so must be reachable via
// LD_LIBRARY_PATH (set by the operator before invoking `dotnet`); see
// `docs/review/phase3.8-phase-c/round_01/acceptance_gates.md` §7 for the
// reproduce block.
//
// Configured by env vars:
//   SPARROW_ENGINE_MODEL_DIR root of sparrow_engine_models; default
//                   /home/miao/repos/PW_refactor/test_files/sparrow_engine_models
//   SPARROW_ENGINE_IMAGE     fixture image path; default the first jpg under
//                   /home/miao/repos/PW_refactor/test_files/test_cameratrap.
//                   Mutually exclusive with SPARROW_ENGINE_IMAGE_DIR.
//   SPARROW_ENGINE_IMAGE_DIR if set, runs detection over every JPEG in this directory
//                   (sorted by filename) and emits a per-image capture in
//                   the same schema. Used for the G1 corpus parity sweep.
//   SPARROW_ENGINE_LIMIT     optional cap on the number of images consumed when
//                   SPARROW_ENGINE_IMAGE_DIR is set. Default: no cap.
//   SPARROW_ENGINE_OUTPUT    JSON path to write the captured detection list (one
//                   line per detection, sorted canonically). Required.
//
// Exit codes:
//   0 success — engine + model loaded, detection ran, JSON dumped.
//   1 setup failure (missing env / lib not loadable / model not found).
//   2 zero detections across the entire corpus — treated as smoke failure
//     under the intentional smoke-failure semantics. The smoke harness
//     ASSUMES the configured corpus
//     (SPARROW_ENGINE_IMAGE / SPARROW_ENGINE_IMAGE_DIR) contains at least one image
//     with detectable content above the 0.2 confidence threshold; an
//     all-empty result means either the engine wired up wrong or the
//     operator pointed at a corpus with no animals. Either way, the
//     operator must investigate before consuming the JSON. Pick a
//     different corpus if "zero detections" is a legitimate outcome
//     for your input.
//   3 unhandled exception.
//
// Symbols used (subset of sparrow-engine/sparrow-engine-cpu/sparrow_engine.h, the Phase A artifact):
//   sparrow_engine_engine_new           (engine ctor)
//   sparrow_engine_load_model_by_id     (manifest-driven model loader)
//   sparrow_engine_detect               (encoded-bytes detection path)
//   sparrow_engine_detections_free      (result deallocator)
//   sparrow_engine_unload_model         (model cleanup)
//   sparrow_engine_engine_free          (engine cleanup)
//   sparrow_engine_last_error           (error string accessor)

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace SparrowEngineSmoke;

internal static class Native
{
    private const string Lib = "sparrow_engine";

    // Marshalling contract for the [DllImport] block below:
    //   * All `*const c_char` IN params (config_json, model_id) are passed as
    //     UTF-8 NUL-terminated bytes via `IntPtr` (caller pins via `fixed`
    //     and casts a `byte*` to `IntPtr`). The legacy `CharSet = CharSet.Ansi`
    //     attribute on sparrow_engine_engine_new + sparrow_engine_load_model_by_id is INACTIVE
    //     because no `string` params are auto-marshalled — it's preserved
    //     to keep csbindgen-generated NativeMethods.g.cs and this hand-rolled
    //     mirror byte-identical at the attribute level.
    //   * All `*const c_char` OUT pointers (sparrow_engine_last_error, SparrowEngineDetection.Label)
    //     are UTF-8 NUL-terminated and Rust-owned. Read with
    //     `Marshal.PtrToStringUTF8`, NEVER `PtrToStringAnsi`. Do NOT free —
    //     ownership remains with the engine.
    //   * `usize` (`UIntPtr`) is the FFI length type; convert via `ToUInt64()`
    //     and bounds-check before narrowing to `int`.
    //   * Pointer-returning ctors (sparrow_engine_engine_new, sparrow_engine_load_model_by_id,
    //     sparrow_engine_detect) return IntPtr.Zero on failure; caller MUST consult
    //     `sparrow_engine_last_error()` and skip the matching `*_free` deallocator.

    [StructLayout(LayoutKind.Sequential)]
    public struct SparrowEngineBBox
    {
        public float XMin;
        public float YMin;
        public float XMax;
        public float YMax;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SparrowEngineDetection
    {
        public SparrowEngineBBox Bbox;
        public IntPtr Label;       // const char *
        public uint LabelId;
        public float Confidence;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SparrowEngineDetections
    {
        public IntPtr Data;        // *SparrowEngineDetection
        public UIntPtr Len;        // uintptr_t
        public uint ImageWidth;
        public uint ImageHeight;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SparrowEngineDetectOpts
    {
        public float ConfidenceThreshold;
        public uint MaxDetections;
    }

    [DllImport(Lib, EntryPoint = "sparrow_engine_engine_new", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern IntPtr sparrow_engine_engine_new(IntPtr configJson);

    [DllImport(Lib, EntryPoint = "sparrow_engine_engine_free", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern void sparrow_engine_engine_free(IntPtr engine);

    [DllImport(Lib, EntryPoint = "sparrow_engine_load_model_by_id", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern IntPtr sparrow_engine_load_model_by_id(IntPtr engine, IntPtr modelId);

    [DllImport(Lib, EntryPoint = "sparrow_engine_unload_model", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern void sparrow_engine_unload_model(IntPtr model);

    [DllImport(Lib, EntryPoint = "sparrow_engine_detect", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern IntPtr sparrow_engine_detect(IntPtr model, IntPtr imageBytes, UIntPtr len, IntPtr opts);

    [DllImport(Lib, EntryPoint = "sparrow_engine_detections_free", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern void sparrow_engine_detections_free(IntPtr detections);

    [DllImport(Lib, EntryPoint = "sparrow_engine_last_error", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern IntPtr sparrow_engine_last_error();
}

internal static class Program
{
    private static int Main()
    {
        try
        {
            return Run();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"FATAL: {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return 3;
        }
    }

    private static int SetupFailure(string message)
    {
        Console.Error.WriteLine($"SETUP: {message}");
        return 1;
    }

    private static int Run()
    {
        var modelDir = Environment.GetEnvironmentVariable("SPARROW_ENGINE_MODEL_DIR")
            ?? "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models";
        var outputPathEnv = Environment.GetEnvironmentVariable("SPARROW_ENGINE_OUTPUT");
        if (string.IsNullOrWhiteSpace(outputPathEnv))
        {
            return SetupFailure("SPARROW_ENGINE_OUTPUT env var required.");
        }
        var outputPath = outputPathEnv;
        var outputParent = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrEmpty(outputParent))
        {
            Directory.CreateDirectory(outputParent);
        }
        var imageDirEnv = Environment.GetEnvironmentVariable("SPARROW_ENGINE_IMAGE_DIR");
        var limitEnv = Environment.GetEnvironmentVariable("SPARROW_ENGINE_LIMIT");
        const string modelId = "megadetector-v6-yolov10e";

        int? limit = null;
        if (!string.IsNullOrEmpty(limitEnv))
        {
            if (!int.TryParse(limitEnv, out var parsedLimit) || parsedLimit <= 0)
            {
                return SetupFailure("SPARROW_ENGINE_LIMIT must be a positive integer when set.");
            }
            limit = parsedLimit;
        }

        string[] imagePaths;
        if (!string.IsNullOrEmpty(imageDirEnv))
        {
            if (!Directory.Exists(imageDirEnv))
            {
                return SetupFailure($"SPARROW_ENGINE_IMAGE_DIR does not exist: {imageDirEnv}");
            }
            var all = Directory.EnumerateFiles(imageDirEnv, "*.jpg")
                .Concat(Directory.EnumerateFiles(imageDirEnv, "*.jpeg"))
                .OrderBy(p => p, StringComparer.Ordinal)
                .ToArray();
            if (limit is int positiveLimit && positiveLimit < all.Length)
            {
                all = all.Take(positiveLimit).ToArray();
            }
            if (all.Length == 0)
            {
                return SetupFailure($"SPARROW_ENGINE_IMAGE_DIR contains no .jpg/.jpeg files: {imageDirEnv}");
            }
            imagePaths = all;
        }
        else
        {
            var imageEnv = Environment.GetEnvironmentVariable("SPARROW_ENGINE_IMAGE");
            string? single;
            if (!string.IsNullOrEmpty(imageEnv))
            {
                single = imageEnv;
                if (!File.Exists(single))
                {
                    return SetupFailure($"SPARROW_ENGINE_IMAGE does not exist: {single}");
                }
            }
            else
            {
                var defaultImageDir = "/home/miao/repos/PW_refactor/test_files/test_cameratrap";
                if (!Directory.Exists(defaultImageDir))
                {
                    return SetupFailure($"default image directory does not exist: {defaultImageDir}");
                }
                single = Directory.EnumerateFiles(defaultImageDir, "*.jpg")
                    .OrderBy(p => p, StringComparer.Ordinal)
                    .FirstOrDefault();
                if (single is null)
                {
                    return SetupFailure($"default image directory contains no .jpg files: {defaultImageDir}");
                }
            }
            imagePaths = new[] { single };
        }

        Console.Error.WriteLine($"[cs_pinvoke_smoke] model_dir={modelDir}");
        Console.Error.WriteLine($"[cs_pinvoke_smoke] n_images={imagePaths.Length}");
        Console.Error.WriteLine($"[cs_pinvoke_smoke] output={outputPath}");
        Console.Error.WriteLine($"[cs_pinvoke_smoke] model_id={modelId}");

        // Engine config — manifest-driven, points at $SPARROW_ENGINE_MODEL_DIR.
        var configJson = $"{{\"device\":\"auto\",\"model_dir\":\"{modelDir}\"}}";
        var configBytes = Encoding.UTF8.GetBytes(configJson + "\0");
        var modelIdBytes = Encoding.UTF8.GetBytes(modelId + "\0");

        IntPtr engine = IntPtr.Zero, model = IntPtr.Zero;
        var perImage = new List<object>(imagePaths.Length);
        var totalDetections = 0;
        try
        {
            unsafe
            {
                fixed (byte* configPtr = configBytes)
                fixed (byte* modelIdPtr = modelIdBytes)
                {
                    engine = Native.sparrow_engine_engine_new((IntPtr)configPtr);
                    if (engine == IntPtr.Zero)
                    {
                        var err = Native.sparrow_engine_last_error();
                        var errStr = err == IntPtr.Zero ? "(none)" : Marshal.PtrToStringUTF8(err) ?? "(null)";
                        Console.Error.WriteLine($"sparrow_engine_engine_new failed: {errStr}");
                        return 1;
                    }
                    Console.Error.WriteLine("[cs_pinvoke_smoke] engine OK");

                    model = Native.sparrow_engine_load_model_by_id(engine, (IntPtr)modelIdPtr);
                    if (model == IntPtr.Zero)
                    {
                        var err = Native.sparrow_engine_last_error();
                        var errStr = err == IntPtr.Zero ? "(none)" : Marshal.PtrToStringUTF8(err) ?? "(null)";
                        Console.Error.WriteLine($"sparrow_engine_load_model_by_id failed: {errStr}");
                        return 1;
                    }
                    Console.Error.WriteLine("[cs_pinvoke_smoke] model OK");
                }
            }

            for (var imgIdx = 0; imgIdx < imagePaths.Length; imgIdx++)
            {
                var imagePath = imagePaths[imgIdx];
                var sorted = RunOne(model, imagePath, out var imgW, out var imgH);
                if (sorted is null)
                {
                    return 1;
                }
                if (imgIdx % 10 == 0 || imgIdx == imagePaths.Length - 1)
                {
                    Console.Error.WriteLine($"[cs_pinvoke_smoke] {imgIdx + 1}/{imagePaths.Length}: {Path.GetFileName(imagePath)} dets={sorted.Length} {imgW}x{imgH}");
                }
                totalDetections += sorted.Length;
                perImage.Add(new
                {
                    path = imagePath,
                    n_detections = sorted.Length,
                    detections = sorted,
                });
            }

            // Capture predictions in scripts/capture_predictions.py schema so
            // scripts/compare_predictions.py can ingest it directly with any
            // tolerance preset (gate_0 for intra-flavor; cross_flavor for the
            // sparrow-engine-cpu↔sparrow-engine-gpu MDv6 dual-flavor parity gate at §6.2 of
            // docs/review/phase3.8-phase-c/round_01/acceptance_gates.md).
            // Top-level "sparrow-engine" key wrapping engine + per_image list.
            var capture = new
            {
                sparrow_engine = new
                {
                    engine = "libsparrow_engine",
                    model = "MDV6-yolov10-e",
                    n_images = imagePaths.Length,
                    total_detections = totalDetections,
                    per_image = perImage,
                },
            };
            var jsonOptions = new JsonSerializerOptions { WriteIndented = false };
            File.WriteAllText(outputPath, JsonSerializer.Serialize(capture, jsonOptions));
            Console.Error.WriteLine($"[cs_pinvoke_smoke] wrote {outputPath}: {imagePaths.Length} images, {totalDetections} detections");

            if (totalDetections == 0)
            {
                Console.Error.WriteLine("[cs_pinvoke_smoke] zero detections — smoke failed");
                return 2;
            }
            return 0;
        }
        finally
        {
            if (model != IntPtr.Zero) Native.sparrow_engine_unload_model(model);
            if (engine != IntPtr.Zero) Native.sparrow_engine_engine_free(engine);
        }
    }

    private static object[]? RunOne(IntPtr model, string imagePath, out uint imageWidth, out uint imageHeight)
    {
        imageWidth = 0;
        imageHeight = 0;
        var imageBytes = File.ReadAllBytes(imagePath);
        var opts = new Native.SparrowEngineDetectOpts { ConfidenceThreshold = 0.2f, MaxDetections = 0 };

        IntPtr dets = IntPtr.Zero;
        try
        {
            unsafe
            {
                fixed (byte* imgPtr = imageBytes)
                {
                    var optsPtr = Marshal.AllocHGlobal(Marshal.SizeOf<Native.SparrowEngineDetectOpts>());
                    try
                    {
                        Marshal.StructureToPtr(opts, optsPtr, fDeleteOld: false);
                        dets = Native.sparrow_engine_detect(model, (IntPtr)imgPtr, (UIntPtr)imageBytes.Length, optsPtr);
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(optsPtr);
                    }
                }
            }

            if (dets == IntPtr.Zero)
            {
                var err = Native.sparrow_engine_last_error();
                var errStr = err == IntPtr.Zero ? "(none)" : Marshal.PtrToStringUTF8(err) ?? "(null)";
                Console.Error.WriteLine($"sparrow_engine_detect failed on {imagePath}: {errStr}");
                return null;
            }

            var detsStruct = Marshal.PtrToStructure<Native.SparrowEngineDetections>(dets);
            // S-2: usize→int with explicit overflow guard. UIntPtr.ToUInt64
            // never throws; bound to int.MaxValue keeps the row array alloc
            // sane on a hypothetical >2^31 detection result (detection counts
            // never approach this in practice; the guard exists so the
            // contract is correct, not narrowing-by-cast).
            var lenU64 = detsStruct.Len.ToUInt64();
            if (lenU64 > int.MaxValue)
            {
                Console.Error.WriteLine($"sparrow_engine_detect returned {lenU64} detections (>{int.MaxValue}); truncating to int.MaxValue.");
                lenU64 = int.MaxValue;
            }
            var len = (int)lenU64;
            imageWidth = detsStruct.ImageWidth;
            imageHeight = detsStruct.ImageHeight;

            var sz = Marshal.SizeOf<Native.SparrowEngineDetection>();
            var rows = new (string Label, float Score, float X0, float Y0, float X1, float Y1)[len];
            for (var i = 0; i < len; i++)
            {
                var det = Marshal.PtrToStructure<Native.SparrowEngineDetection>(detsStruct.Data + i * sz);
                var label = Marshal.PtrToStringUTF8(det.Label) ?? "(null)";
                rows[i] = (
                    label,
                    (float)Math.Round(det.Confidence, 6),
                    (float)Math.Round(det.Bbox.XMin, 6),
                    (float)Math.Round(det.Bbox.YMin, 6),
                    (float)Math.Round(det.Bbox.XMax, 6),
                    (float)Math.Round(det.Bbox.YMax, 6)
                );
            }

            // Sort canonically: (-score, x_min, y_min) — matches scripts/capture_predictions.py.
            return rows
                .OrderByDescending(r => r.Score)
                .ThenBy(r => r.X0)
                .ThenBy(r => r.Y0)
                .Select(r => (object)new
                {
                    label = r.Label,
                    score = r.Score,
                    x_min = r.X0,
                    y_min = r.Y0,
                    x_max = r.X1,
                    y_max = r.Y1,
                })
                .ToArray();
        }
        finally
        {
            if (dets != IntPtr.Zero) Native.sparrow_engine_detections_free(dets);
        }
    }
}
