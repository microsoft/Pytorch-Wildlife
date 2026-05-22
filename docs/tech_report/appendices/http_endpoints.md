# HTTP Endpoint Inventory

`sparrow-engine-server` exposes a 15 endpoint/method surface after Phase 4.2. axum router. JSON request / response. `multipart/form-data` for image/audio uploads.

## Inference (5)

| Method | Path | Purpose | Request | Response |
|--------|------|---------|---------|----------|
| POST | `/v1/detect` | Single-image detection | multipart: `file`, `model` query | `{"detections": [Detection, ...], "model_id": "..."}` |
| POST | `/v1/detect/batch` | Batch detection | multipart: `file` (multiple) | Array of per-image results |
| POST | `/v1/classify` | Single-image classification | multipart: `file`, `model` | `{"classifications": [...], "model_id": "..."}` |
| POST | `/v1/audio/detect` | Audio detection | multipart: `file` (WAV), `model` | `{"segments": [AudioSegment, ...]}` |
| POST | `/v1/pipeline` | Named pipeline or ad-hoc detector/classifier pipeline | multipart: `image`; query `pipeline` OR `detector` + `classifier` | pipeline detections with per-detection classifications |

## Management (8)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/catalog` | List catalog-discovered available models with `loaded` flags |
| GET | `/v1/models` | List currently loaded model sessions (extended `ModelInfo` since Phase 3) |
| POST | `/v1/models/load` | Explicitly load a model by ID (`SPARROW_ENGINE_PRELOAD` uses the same locked path) |
| DELETE | `/v1/models/{id}` | Unload a model |
| GET | `/v1/pipelines` | List registered pipelines |
| POST | `/v1/pipelines` | Create or replace a runtime pipeline alias; optional `persist=true` writes `pipeline.toml` |
| POST | `/v1/pipelines/load` | Register a pipeline from an existing TOML manifest |
| DELETE | `/v1/pipelines/{id}` | Unregister a pipeline; persisted aliases also remove `pipeline.toml` |

## Health (2)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/health` | Application-level health (ORT sessions healthy, model cache OK) |
| GET | `/healthz` | Container-orchestrator-style liveness probe |

## Response schema conventions

- All detections carry normalized bbox `[0, 1]` (D-v3-6).
- All `ModelInfo` responses carry Phase 3 extended fields: `version`, `description`, `onnx_sha256`, `onnx_size_bytes`, `default` (always serialized, even if null).
- Errors follow a consistent JSON shape: `{"error": {"code": "...", "message": "..."}}`.

## Example: `GET /v1/catalog`

```json
[
  {
    "model_id": "mdv6",
    "model_type": "detector",
    "framework": "onnx",
    "loaded": false
  }
]
```

## Example: `GET /v1/models`

```json
{
  "models": [
    {
      "id": "mdv6",
      "model_type": "detector",
      "version": "v6.0.0",
      "description": "MegaDetector v6 — general animal/person/vehicle detection",
      "onnx_sha256": "abc123...",
      "onnx_size_bytes": 164521280,
      "default": true
    },
    // ...
  ]
}
```

## Source

`sparrow-engine/sparrow-engine-server/src/handlers/` — axum handlers split by concern (detect, classify, pipeline, catalog, models, pipelines, pipeline alias management, health).

## Wire-compat note

Phase 3 extended `ModelResponse` with 5 new fields. `default: bool` is always serialized. Strict consumers (including pre-Phase-3 `sparrow-engine-client`) must accept the extended fields. sparrow-engine-client was extended in `commit 4f272a8` as part of Phase 3 final audit-fix R2.

## References

- `06_gotchas_and_constraints.md` — SRV1 wire-compat event
- `09_sparrow_studio_integration.md` — Sparrow Studio Web workers consuming this API
- `docs/master_plan.md § Phase 2` — original endpoint list source
- `docs/master_plan.md § Phase 4.2` — `/v1/catalog`, lazy boot, and runtime pipeline alias additions
