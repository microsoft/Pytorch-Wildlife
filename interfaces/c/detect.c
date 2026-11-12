/*
 * Detect animals, people and vehicles in one image through the Sparrow Engine C library.
 *
 *   detect <model_dir> <model_id> <image> [threshold]
 *   detect ~/.sparrow-engine/models MDV6-yolov10-c photo.jpg 0.2
 *
 * Build and run: see README.md in this folder.
 */
#include <stdio.h>
#include <stdlib.h>

#include "sparrow_engine.h"

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = size > 0 ? malloc((size_t)size) : NULL;
    if (buf && fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *len = buf ? (size_t)size : 0;
    return buf;
}

static int fail(const char *what) {
    const char *err = sparrow_engine_last_error();
    fprintf(stderr, "%s failed: %s\n", what, err ? err : "(no error message)");
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model_dir> <model_id> <image> [threshold]\n", argv[0]);
        return 2;
    }
    const char *model_dir = argv[1], *model_id = argv[2], *image_path = argv[3];
    float threshold = argc > 4 ? strtof(argv[4], NULL) : 0.2f;

    size_t len;
    unsigned char *image = read_file(image_path, &len);
    if (!image) {
        fprintf(stderr, "cannot read %s\n", image_path);
        return 1;
    }

    /* The engine takes a JSON config; model_dir is required. Keep the path free of quotes. */
    char config[4096];
    snprintf(config, sizeof config, "{\"device\":\"auto\",\"model_dir\":\"%s\"}", model_dir);

    int rc = 0;
    SparrowEngine *engine = sparrow_engine_engine_new(config);
    if (!engine) {
        free(image);
        return fail("sparrow_engine_engine_new");
    }

    SparrowEngineModel *model = sparrow_engine_load_model_by_id(engine, model_id);
    if (!model) {
        rc = fail("sparrow_engine_load_model_by_id");
        goto done_engine;
    }

    SparrowEngineDetectOpts opts = {.confidence_threshold = threshold, .max_detections = 0};
    SparrowEngineDetections *dets = sparrow_engine_detect(model, image, len, &opts);
    if (!dets) {
        rc = fail("sparrow_engine_detect");
        goto done_model;
    }

    printf("%s: %u x %u, %zu detections\n", image_path, dets->image_width, dets->image_height,
           (size_t)dets->len);
    for (size_t i = 0; i < dets->len; i++) {
        const SparrowEngineDetection *d = &dets->data[i];
        /* Boxes are normalized to [0, 1]; scale to pixels. */
        printf("  %-8s %.2f  box(px) %4.0f %4.0f %4.0f %4.0f\n", d->label, d->confidence,
               d->bbox.x_min * dets->image_width, d->bbox.y_min * dets->image_height,
               d->bbox.x_max * dets->image_width, d->bbox.y_max * dets->image_height);
    }
    sparrow_engine_detections_free(dets);

done_model:
    sparrow_engine_unload_model(model);
done_engine:
    sparrow_engine_engine_free(engine);
    free(image);
    return rc;
}
