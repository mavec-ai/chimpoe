use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub pipeline: PipelineConfig,
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub window_size: usize,
    pub overlap_size: usize,
    pub similarity_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            window_size: 40,
            overlap_size: 2,
            similarity_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub temperature: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "llama3.2".to_string(),
            base_url: Some("http://localhost:11434/v1".to_string()),
            temperature: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub dimension: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "nomic-embed-text".to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            dimension: 768,
        }
    }
}
