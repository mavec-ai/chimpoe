use crate::config::{EmbeddingConfig, OPENAI_EMBEDDER_BASE_URL};
use crate::error::{EmbeddingError, EmbeddingResult};
use crate::traits::Embedder;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenAIEmbedder {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    dimension: usize,
}

#[derive(Serialize)]
struct EmbedRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: Option<OpenAIError>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
}

impl OpenAIEmbedder {
    pub fn new(config: &EmbeddingConfig) -> Self {
        let dimension = if config.dimension > 0 {
            config.dimension
        } else {
            Self::default_dimension(&config.model)
        };
        Self {
            client: Client::new(),
            api_key: config.api_key.clone().unwrap_or_default(),
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| OPENAI_EMBEDDER_BASE_URL.to_string()),
            model: config.model.clone(),
            dimension,
        }
    }

    fn default_dimension(model: &str) -> usize {
        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }

    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = api_key;
        self
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
impl Embedder for OpenAIEmbedder {
    async fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        let request = EmbedRequest {
            input: texts.iter().map(|s| s.to_string()).collect(),
            model: self.model.clone(),
        };

        let url = format!("{}/embeddings", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| EmbeddingError::ApiError(e.to_string()))?;

        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body)
                && let Some(error) = error_response.error
            {
                return Err(EmbeddingError::ApiError(error.message));
            }
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let embed_response: EmbedResponse = serde_json::from_str(&body)
            .map_err(|e| EmbeddingError::ApiError(format!("Failed to parse response: {}", e)))?;

        let embeddings: Vec<Vec<f32>> = embed_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
