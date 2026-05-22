using System.Runtime.InteropServices;
using System.Text;
using SparrowEngine.Native;

// ============================================================================
// SparrowEngine libsparrow_engine P/Invoke FFI Integration Test
// ============================================================================
// Exercises the full FFI boundary: engine lifecycle, model loading,
// detection (vision + audio), classification, error handling, cleanup.
// ============================================================================

unsafe class Program
{
    static int _passed = 0;
    static int _failed = 0;

    static void Assert(bool condition, string testName, string? detail = null)
    {
        if (condition)
        {
            Console.WriteLine($"  [PASS] {testName}");
            _passed++;
        }
        else
        {
            Console.WriteLine($"  [FAIL] {testName}{(detail != null ? $" — {detail}" : "")}");
            _failed++;
        }
    }

    /// <summary>Marshal a C# string to a null-terminated UTF-8 byte* on the unmanaged heap.</summary>
    static byte* ToUtf8(string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s + '\0');
        IntPtr ptr = Marshal.AllocHGlobal(bytes.Length);
        Marshal.Copy(bytes, 0, ptr, bytes.Length);
        return (byte*)ptr;
    }

    /// <summary>Read a null-terminated UTF-8 byte* into a C# string.</summary>
    static string? FromUtf8(byte* ptr)
    {
        if (ptr == null) return null;
        return Marshal.PtrToStringUTF8((IntPtr)ptr);
    }

    /// <summary>Get last error string from sparrow_engine (do NOT free).</summary>
    static string? GetLastError()
    {
        byte* err = NativeMethods.sparrow_engine_last_error();
        return FromUtf8(err);
    }

    // Paths
    const string ModelDir = "/home/miao/repos/PW_refactor/test_files/onnx";
    const string ImageDir = "/home/miao/repos/PW_refactor/test_files/test_cameratrap";
    const string AudioPath = "/home/miao/repos/PW_refactor/test_files/test_audio/G010_timelapse_20250629.wav";
    const string Mdv6Manifest = "/home/miao/repos/PW_refactor/test_files/onnx/mdv6_manifest.toml";
    const string SpeciesNetManifest = "/home/miao/repos/PW_refactor/test_files/onnx/speciesnet_manifest.toml";
    const string AudioBirdsManifest = "/home/miao/repos/PW_refactor/test_files/onnx/audiobirds_manifest.toml";

    static int Main()
    {
        Console.WriteLine("=== SparrowEngine C# P/Invoke FFI Test ===\n");

        // Find first .jpg test image
        string? testImage = null;
        if (Directory.Exists(ImageDir))
        {
            var jpgs = Directory.GetFiles(ImageDir, "*.jpg");
            if (jpgs.Length > 0)
            {
                Array.Sort(jpgs);
                testImage = jpgs[0];
            }
        }
        if (testImage == null)
        {
            Console.WriteLine("[SKIP] No test images found in " + ImageDir);
            return 1;
        }
        Console.WriteLine($"Test image: {testImage}");
        Console.WriteLine($"Audio file: {AudioPath}");
        Console.WriteLine();

        // =====================================================================
        // 1. Engine Lifecycle
        // =====================================================================
        Console.WriteLine("--- 1. Engine Lifecycle ---");

        string configJson = $"{{\"device\": \"cpu\", \"model_dir\": \"{ModelDir}\"}}";
        byte* configPtr = ToUtf8(configJson);

        void* engine = NativeMethods.sparrow_engine_new(configPtr);
        Marshal.FreeHGlobal((IntPtr)configPtr);

        Assert(engine != null, "sparrow_engine_new returns non-null");
        if (engine == null)
        {
            string? err = GetLastError();
            Console.WriteLine($"  Engine creation failed: {err}");
            Console.WriteLine($"\nResults: {_passed} passed, {_failed} failed");
            return 1;
        }

        // Health check
        byte* healthPtr = NativeMethods.sparrow_engine_health(engine);
        string? healthJson = FromUtf8(healthPtr);
        Assert(healthJson != null && healthJson.Contains("\"status\":\"ok\""),
            "sparrow_engine_health returns valid JSON", healthJson);
        if (healthPtr != null) NativeMethods.sparrow_engine_free_string(healthPtr);

        // List models (should be empty initially)
        byte* modelsPtr = NativeMethods.sparrow_engine_list_models(engine);
        string? modelsJson = FromUtf8(modelsPtr);
        Assert(modelsJson != null && modelsJson == "[]",
            "sparrow_engine_list_models returns empty array initially", modelsJson);
        if (modelsPtr != null) NativeMethods.sparrow_engine_free_string(modelsPtr);

        // =====================================================================
        // 2. Model Loading — MegaDetector v6
        // =====================================================================
        Console.WriteLine("\n--- 2. Model Loading (MDv6) ---");

        byte* mdv6PathPtr = ToUtf8(Mdv6Manifest);
        void* mdv6Model = NativeMethods.sparrow_engine_load_model(engine, mdv6PathPtr);
        Marshal.FreeHGlobal((IntPtr)mdv6PathPtr);

        Assert(mdv6Model != null, "sparrow_engine_load_model(MDv6) returns non-null");
        if (mdv6Model == null)
        {
            string? err = GetLastError();
            Console.WriteLine($"  MDv6 load failed: {err}");
        }

        // Verify model appears in list
        modelsPtr = NativeMethods.sparrow_engine_list_models(engine);
        modelsJson = FromUtf8(modelsPtr);
        Assert(modelsJson != null && modelsJson.Contains("detector"),
            "sparrow_engine_list_models includes loaded detector", modelsJson);
        if (modelsPtr != null) NativeMethods.sparrow_engine_free_string(modelsPtr);

        // =====================================================================
        // 3. Detection (MDv6)
        // =====================================================================
        Console.WriteLine("\n--- 3. Detection (MDv6) ---");

        if (mdv6Model != null)
        {
            byte[] imageBytes = File.ReadAllBytes(testImage);
            fixed (byte* imagePtr = imageBytes)
            {
                // With default opts (null)
                SparrowEngineDetections* detections = NativeMethods.sparrow_engine_detect(
                    mdv6Model, imagePtr, (nuint)imageBytes.Length, null);

                Assert(detections != null, "sparrow_engine_detect returns non-null");

                if (detections != null)
                {
                    Assert(detections->image_width > 0, $"image_width > 0 (got {detections->image_width})");
                    Assert(detections->image_height > 0, $"image_height > 0 (got {detections->image_height})");
                    Console.WriteLine($"  Detections: {detections->len} objects in {detections->image_width}x{detections->image_height} image");

                    // Print first few detections
                    for (nuint i = 0; i < detections->len && i < 5; i++)
                    {
                        SparrowEngineDetection* det = &detections->data[i];
                        string? label = FromUtf8(det->label);
                        Console.WriteLine($"    [{i}] label=\"{label}\" id={det->label_id} conf={det->confidence:F3} " +
                            $"bbox=({det->bbox.x_min:F3},{det->bbox.y_min:F3},{det->bbox.x_max:F3},{det->bbox.y_max:F3})");
                    }

                    // Validate bbox normalization: all values in [0, 1]
                    bool bboxValid = true;
                    for (nuint i = 0; i < detections->len; i++)
                    {
                        SparrowEngineDetection* det = &detections->data[i];
                        if (det->bbox.x_min < 0 || det->bbox.x_min > 1 ||
                            det->bbox.y_min < 0 || det->bbox.y_min > 1 ||
                            det->bbox.x_max < 0 || det->bbox.x_max > 1 ||
                            det->bbox.y_max < 0 || det->bbox.y_max > 1)
                        {
                            bboxValid = false;
                            break;
                        }
                    }
                    if (detections->len > 0)
                        Assert(bboxValid, "All bboxes normalized to [0,1]");

                    NativeMethods.sparrow_engine_detections_free(detections);
                    Assert(true, "sparrow_engine_detections_free completed without crash");
                }

                // With custom opts
                SparrowEngineDetectOpts opts;
                opts.confidence_threshold = 0.5f;
                opts.max_detections = 3;
                SparrowEngineDetections* detections2 = NativeMethods.sparrow_engine_detect(
                    mdv6Model, imagePtr, (nuint)imageBytes.Length, &opts);

                Assert(detections2 != null, "sparrow_engine_detect with opts returns non-null");
                if (detections2 != null)
                {
                    Assert(detections2->len <= 3, $"max_detections=3 respected (got {detections2->len})");
                    NativeMethods.sparrow_engine_detections_free(detections2);
                }
            }
        }

        // =====================================================================
        // 4. Classification (SpeciesNet)
        // =====================================================================
        Console.WriteLine("\n--- 4. Classification (SpeciesNet) ---");

        byte* snPathPtr = ToUtf8(SpeciesNetManifest);
        void* snModel = NativeMethods.sparrow_engine_load_model(engine, snPathPtr);
        Marshal.FreeHGlobal((IntPtr)snPathPtr);

        Assert(snModel != null, "sparrow_engine_load_model(SpeciesNet) returns non-null");
        if (snModel == null)
        {
            string? err = GetLastError();
            Console.WriteLine($"  SpeciesNet load failed: {err}");
        }

        if (snModel != null)
        {
            byte[] imageBytes = File.ReadAllBytes(testImage);
            fixed (byte* imagePtr = imageBytes)
            {
                SparrowEngineClassifyResult* result = NativeMethods.sparrow_engine_classify(
                    snModel, imagePtr, (nuint)imageBytes.Length, null);

                Assert(result != null, "sparrow_engine_classify returns non-null");

                if (result != null)
                {
                    string? topLabel = FromUtf8(result->label);
                    Assert(topLabel != null && topLabel.Length > 0,
                        $"Top-1 label is non-empty (got \"{topLabel}\")");
                    Assert(result->confidence > 0 && result->confidence <= 1.0f,
                        $"Top-1 confidence in (0,1] (got {result->confidence:F3})");
                    Assert(result->image_width > 0, $"image_width > 0 (got {result->image_width})");
                    Assert(result->processing_time_ms >= 0,
                        $"processing_time_ms >= 0 (got {result->processing_time_ms:F1}ms)");

                    Console.WriteLine($"  Top-1: \"{topLabel}\" (id={result->label_id}, conf={result->confidence:F3})");
                    Console.WriteLine($"  Image: {result->image_width}x{result->image_height}, time={result->processing_time_ms:F1}ms");

                    // Print top-K results
                    Console.WriteLine($"  Top-K results ({result->top_results_len}):");
                    for (nuint i = 0; i < result->top_results_len && i < 5; i++)
                    {
                        SparrowEngineClassification* cls = &result->top_results[i];
                        string? lbl = FromUtf8(cls->label);
                        Console.WriteLine($"    [{i}] \"{lbl}\" id={cls->label_id} conf={cls->confidence:F3}");
                    }

                    // With top_k opts
                    SparrowEngineClassifyOpts opts;
                    opts.top_k = 3;
                    SparrowEngineClassifyResult* result2 = NativeMethods.sparrow_engine_classify(
                        snModel, imagePtr, (nuint)imageBytes.Length, &opts);
                    Assert(result2 != null, "sparrow_engine_classify with top_k=3 returns non-null");
                    if (result2 != null)
                    {
                        Assert(result2->top_results_len <= 3,
                            $"top_k=3 respected (got {result2->top_results_len})");
                        NativeMethods.sparrow_engine_classify_result_free(result2);
                    }

                    NativeMethods.sparrow_engine_classify_result_free(result);
                    Assert(true, "sparrow_engine_classify_result_free completed without crash");
                }
            }
        }

        // =====================================================================
        // 5. Audio Detection
        // =====================================================================
        Console.WriteLine("\n--- 5. Audio Detection ---");

        byte* audioManifestPtr = ToUtf8(AudioBirdsManifest);
        void* audioModel = NativeMethods.sparrow_engine_load_model(engine, audioManifestPtr);
        Marshal.FreeHGlobal((IntPtr)audioManifestPtr);

        Assert(audioModel != null, "sparrow_engine_load_model(AudioBirds) returns non-null");
        if (audioModel == null)
        {
            string? err = GetLastError();
            Console.WriteLine($"  AudioBirds load failed: {err}");
        }

        if (audioModel != null && File.Exists(AudioPath))
        {
            byte* audioPathPtr = ToUtf8(AudioPath);

            // Default opts
            SparrowEngineAudioResult* audioResult = NativeMethods.sparrow_engine_detect_audio(
                audioModel, audioPathPtr, null);

            Assert(audioResult != null, "sparrow_engine_detect_audio returns non-null");

            if (audioResult != null)
            {
                Assert(audioResult->duration_s > 0,
                    $"duration_s > 0 (got {audioResult->duration_s:F2}s)");
                Assert(audioResult->sample_rate > 0,
                    $"sample_rate > 0 (got {audioResult->sample_rate}Hz)");
                Assert(audioResult->processing_time_ms >= 0,
                    $"processing_time_ms >= 0 (got {audioResult->processing_time_ms:F1}ms)");

                Console.WriteLine($"  Audio: {audioResult->duration_s:F2}s @ {audioResult->sample_rate}Hz, " +
                    $"processed in {audioResult->processing_time_ms:F1}ms");
                Console.WriteLine($"  Segments detected: {audioResult->len}");

                for (nuint i = 0; i < audioResult->len && i < 10; i++)
                {
                    SparrowEngineAudioSegment* seg = &audioResult->data[i];
                    Console.WriteLine($"    [{i}] {seg->start_time_s:F2}s — {seg->end_time_s:F2}s (conf={seg->confidence:F3})");

                    // Validate segment times
                    Assert(seg->start_time_s >= 0, $"segment[{i}] start_time >= 0");
                    Assert(seg->end_time_s > seg->start_time_s, $"segment[{i}] end > start");
                    Assert(seg->confidence >= 0 && seg->confidence <= 1.0f, $"segment[{i}] confidence in [0,1]");
                }

                NativeMethods.sparrow_engine_audio_result_free(audioResult);
                Assert(true, "sparrow_engine_audio_result_free completed without crash");
            }

            // With custom opts
            SparrowEngineAudioDetectOpts audioOpts;
            audioOpts.confidence_threshold = 0.8f;
            audioOpts.segment_duration_s = 0;   // use default
            audioOpts.segment_stride_s = 0;     // use default
            SparrowEngineAudioResult* audioResult2 = NativeMethods.sparrow_engine_detect_audio(
                audioModel, audioPathPtr, &audioOpts);
            Assert(audioResult2 != null, "sparrow_engine_detect_audio with high threshold returns non-null");
            if (audioResult2 != null)
            {
                Console.WriteLine($"  High-threshold segments: {audioResult2->len}");
                NativeMethods.sparrow_engine_audio_result_free(audioResult2);
            }

            Marshal.FreeHGlobal((IntPtr)audioPathPtr);
        }
        else if (!File.Exists(AudioPath))
        {
            Console.WriteLine($"  [SKIP] Audio file not found: {AudioPath}");
        }

        // =====================================================================
        // 6. Error Handling
        // =====================================================================
        Console.WriteLine("\n--- 6. Error Handling ---");

        // Null engine → should fail gracefully
        byte* bogusPtr = ToUtf8("bogus.toml");
        void* badModel = NativeMethods.sparrow_engine_load_model(null, bogusPtr);
        Marshal.FreeHGlobal((IntPtr)bogusPtr);
        Assert(badModel == null, "sparrow_engine_load_model(null engine) returns null");
        {
            string? err = GetLastError();
            Assert(err != null && err.Length > 0, $"sparrow_engine_last_error set after null engine: \"{err}\"");
        }

        // Invalid manifest path
        byte* badPathPtr = ToUtf8("/nonexistent/path/model.toml");
        void* badModel2 = NativeMethods.sparrow_engine_load_model(engine, badPathPtr);
        Marshal.FreeHGlobal((IntPtr)badPathPtr);
        Assert(badModel2 == null, "sparrow_engine_load_model(bad path) returns null");
        {
            string? err = GetLastError();
            Assert(err != null && err.Length > 0, $"sparrow_engine_last_error set after bad path: \"{err}\"");
        }

        // Null model for detect → should return null
        SparrowEngineDetections* nullDetect = NativeMethods.sparrow_engine_detect(null, null, 0, null);
        Assert(nullDetect == null, "sparrow_engine_detect(null model) returns null");

        // Invalid config JSON
        byte* badConfigPtr = ToUtf8("{invalid json}");
        void* badEngine = NativeMethods.sparrow_engine_new(badConfigPtr);
        Marshal.FreeHGlobal((IntPtr)badConfigPtr);
        Assert(badEngine == null, "sparrow_engine_new(bad JSON) returns null");
        {
            string? err = GetLastError();
            Assert(err != null && err.Contains("invalid config JSON"),
                $"Error message for bad JSON: \"{err}\"");
        }

        // Free null pointers — should be no-ops (no crash)
        NativeMethods.sparrow_engine_detections_free(null);
        NativeMethods.sparrow_engine_classify_result_free(null);
        NativeMethods.sparrow_engine_audio_result_free(null);
        NativeMethods.sparrow_engine_pipeline_result_free(null);
        NativeMethods.sparrow_engine_free_string(null);
        NativeMethods.sparrow_engine_free(null);
        Assert(true, "All _free(null) calls are no-ops (no crash)");

        // =====================================================================
        // 6b. Raw Pixel Detection (sparrow_engine_detect_raw — Bitmap.LockBits path)
        // =====================================================================
        Console.WriteLine("\n--- 6b. Raw Pixel Detection (sparrow_engine_detect_raw) ---");

        if (mdv6Model != null)
        {
            Assert(Enum.GetUnderlyingType(typeof(SparrowEnginePixelFormat)) == typeof(uint),
                "SparrowEnginePixelFormat underlying type is uint");
            Assert((uint)SparrowEnginePixelFormat.Rgb == 0 &&
                   (uint)SparrowEnginePixelFormat.Rgba == 1 &&
                   (uint)SparrowEnginePixelFormat.Bgra == 2 &&
                   (uint)SparrowEnginePixelFormat.Bgr == 3,
                "SparrowEnginePixelFormat values match C ABI constants");

            // Simulate a 100x100 BGRA bitmap (what Bitmap.LockBits returns).
            const uint rawW = 100;
            const uint rawH = 100;
            const uint bpp = 4; // BGRA = 4 bytes per pixel
            const uint rawStride = rawW * bpp;
            byte[] rawPixels = new byte[rawH * rawStride];

            // Channel-distinct gradient so BGRA channel ordering is exercised.
            for (int i = 0; i < rawPixels.Length; i += 4)
            {
                int pixel = i / 4;
                int x = pixel % (int)rawW;
                int y = pixel / (int)rawW;
                rawPixels[i]     = (byte)(x * 255 / ((int)rawW - 1));     // B
                rawPixels[i + 1] = (byte)(y * 255 / ((int)rawH - 1));     // G
                rawPixels[i + 2] = (byte)(255 - rawPixels[i]);            // R
                rawPixels[i + 3] = 255;                                   // A
            }

            fixed (byte* rawPtr = rawPixels)
            {
                SparrowEngineDetections* rawDetections = NativeMethods.sparrow_engine_detect_raw(
                    mdv6Model, rawPtr, rawW, rawH, rawStride,
                    SparrowEnginePixelFormat.Bgra, null);

                Assert(rawDetections != null, "sparrow_engine_detect_raw returns non-null");

                if (rawDetections != null)
                {
                    Assert(rawDetections->image_width == rawW,
                        $"raw image_width == {rawW} (got {rawDetections->image_width})");
                    Assert(rawDetections->image_height == rawH,
                        $"raw image_height == {rawH} (got {rawDetections->image_height})");
                    Console.WriteLine($"  Raw detections: {rawDetections->len} objects in {rawDetections->image_width}x{rawDetections->image_height}");

                    NativeMethods.sparrow_engine_detections_free(rawDetections);
                    Assert(true, "sparrow_engine_detections_free(raw) completed without crash");
                }
            }

            // Also test with custom opts
            fixed (byte* rawPtr = rawPixels)
            {
                SparrowEngineDetectOpts rawOpts;
                rawOpts.confidence_threshold = 0.9f;
                rawOpts.max_detections = 5;
                SparrowEngineDetections* rawDetections2 = NativeMethods.sparrow_engine_detect_raw(
                    mdv6Model, rawPtr, rawW, rawH, rawStride,
                    SparrowEnginePixelFormat.Bgra, &rawOpts);

                Assert(rawDetections2 != null, "sparrow_engine_detect_raw with opts returns non-null");
                if (rawDetections2 != null)
                {
                    Assert(rawDetections2->len <= 5,
                        $"raw max_detections=5 respected (got {rawDetections2->len})");
                    NativeMethods.sparrow_engine_detections_free(rawDetections2);
                }
            }

            // Invalid raw pixel format code should fail cleanly and set last_error.
            fixed (byte* rawPtr = rawPixels)
            {
                SparrowEngineDetections* invalidFormatDetections = NativeMethods.sparrow_engine_detect_raw(
                    mdv6Model, rawPtr, rawW, rawH, rawStride,
                    (SparrowEnginePixelFormat)999u, null);

                Assert(invalidFormatDetections == null, "sparrow_engine_detect_raw invalid pixel format returns null");
                string? formatErr = GetLastError();
                Assert(formatErr != null && formatErr.Length > 0,
                    $"sparrow_engine_last_error set after invalid pixel format: \"{formatErr}\"");
                if (invalidFormatDetections != null)
                {
                    NativeMethods.sparrow_engine_detections_free(invalidFormatDetections);
                }
            }
        }

        // =====================================================================
        // 6c. Pipeline Negative Test (sparrow_engine_run_pipeline)
        // =====================================================================
        Console.WriteLine("\n--- 6c. Pipeline Negative Test (sparrow_engine_run_pipeline) ---");

        {
            // Call with a non-existent pipeline ID — should return null + error
            byte* fakePipelineId = ToUtf8("nonexistent-pipeline-v99");
            byte[] dummyImage = File.ReadAllBytes(testImage);
            fixed (byte* imgPtr = dummyImage)
            {
                SparrowEnginePipelineResult* pipeResult = NativeMethods.sparrow_engine_run_pipeline(
                    engine, fakePipelineId, imgPtr, (nuint)dummyImage.Length, null, null);

                Assert(pipeResult == null, "sparrow_engine_run_pipeline(bad pipeline_id) returns null");
                string? pipeErr = GetLastError();
                Assert(pipeErr != null && pipeErr.Length > 0,
                    $"sparrow_engine_last_error set after bad pipeline_id: \"{pipeErr}\"");
            }
            Marshal.FreeHGlobal((IntPtr)fakePipelineId);
        }

        // =====================================================================
        // 6d. Load Model By ID Negative Test (sparrow_engine_load_model_by_id)
        // =====================================================================
        Console.WriteLine("\n--- 6d. Load Model By ID Negative Test (sparrow_engine_load_model_by_id) ---");

        {
            // Model dir doesn't have md-audiobirds-v1/manifest.toml — should fail
            byte* badModelId = ToUtf8("md-audiobirds-v1");
            void* badIdModel = NativeMethods.sparrow_engine_load_model_by_id(engine, badModelId);
            Marshal.FreeHGlobal((IntPtr)badModelId);

            Assert(badIdModel == null, "sparrow_engine_load_model_by_id(bad id) returns null");
            string? idErr = GetLastError();
            Assert(idErr != null && idErr.Length > 0,
                $"sparrow_engine_last_error set after bad model_id: \"{idErr}\"");
        }

        // =====================================================================
        // 7. Cleanup
        // =====================================================================
        Console.WriteLine("\n--- 7. Cleanup ---");

        if (audioModel != null) NativeMethods.sparrow_engine_unload_model(audioModel);
        Assert(true, "sparrow_engine_unload_model(audio) completed");

        if (snModel != null) NativeMethods.sparrow_engine_unload_model(snModel);
        Assert(true, "sparrow_engine_unload_model(speciesnet) completed");

        if (mdv6Model != null) NativeMethods.sparrow_engine_unload_model(mdv6Model);
        Assert(true, "sparrow_engine_unload_model(mdv6) completed");

        NativeMethods.sparrow_engine_free(engine);
        Assert(true, "sparrow_engine_free completed");

        // =====================================================================
        // Summary
        // =====================================================================
        Console.WriteLine($"\n=== Results: {_passed} passed, {_failed} failed ===");
        return _failed == 0 ? 0 : 1;
    }
}
