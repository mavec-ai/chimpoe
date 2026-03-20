use crate::config::{EmbeddingConfig, OLLAMA_EMBEDDER_BASE_URL};
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::Embedder;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaEmbedder {
    client: Client,
    base_url: String,
    model: String,
    dimension: usize,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    pub fn new(config: &EmbeddingConfig) -> Self {
        let dimension = if config.dimension > 0 {
            config.dimension
        } else {
            Self::default_dimension(&config.model)
        };
        Self {
            client: Client::new(),
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| OLLAMA_EMBEDDER_BASE_URL.to_string()),
            model: config.model.clone(),
            dimension,
        }
    }

    fn default_dimension(model: &str) -> usize {
        match model {
            "nomic-embed-text" => 768,
            "mxbai-embed-large" => 1024,
            "all-minilm" => 384,
            _ => 768,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let request = EmbedRequest {
            model: self.model.clone(),
            input: texts.iter().map(|s| s.to_string()).collect(),
        };

        let url = format!("{}/api/embed", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let embed_response: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        Ok(embed_response.embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
