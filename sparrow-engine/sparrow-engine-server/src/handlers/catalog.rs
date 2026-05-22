//! Catalog discovery handler.

use std::collections::HashSet;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::discover::Catalog;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CatalogEntryResponse {
    pub model_id: String,
    pub model_type: String,
    pub framework: String,
    pub loaded: bool,
}

/// GET /v1/catalog — available models discovered at boot, plus loaded state.
pub async fn list_catalog(State(state): State<AppState>) -> Json<Vec<CatalogEntryResponse>> {
    let loaded: HashSet<String> = state
        .engine
        .loaded_models()
        .into_iter()
        .map(|m| m.id)
        .collect();
    Json(catalog_entries(&state.catalog, &loaded))
}

fn catalog_entries(catalog: &Catalog, loaded: &HashSet<String>) -> Vec<CatalogEntryResponse> {
    catalog
        .models
        .values()
        .map(|model| CatalogEntryResponse {
            model_id: model.id.clone(),
            model_type: model.model_type.to_string(),
            framework: "onnx".to_string(),
            loaded: loaded.contains(&model.id),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::{ModelInfo, ModelType};
    use std::path::PathBuf;

    fn model_info(id: &str, model_type: ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}/manifest.toml")),
            model_type,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        }
    }

    #[test]
    fn catalog_entries_include_loaded_flag() {
        let mut catalog = Catalog::default();
        catalog
            .models
            .insert("mdv6".to_string(), model_info("mdv6", ModelType::Detector));
        catalog.models.insert(
            "speciesnet".to_string(),
            model_info("speciesnet", ModelType::Classifier),
        );
        let loaded = HashSet::from(["speciesnet".to_string()]);

        let entries = catalog_entries(&catalog, &loaded);
        assert_eq!(entries.len(), 2);
        let mdv6 = entries.iter().find(|e| e.model_id == "mdv6").unwrap();
        assert!(!mdv6.loaded);
        let speciesnet = entries.iter().find(|e| e.model_id == "speciesnet").unwrap();
        assert!(speciesnet.loaded);
        assert_eq!(speciesnet.framework, "onnx");
    }
}
