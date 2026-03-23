use serde::{Deserialize, Serialize};

pub const OLLAMA_LLM_MODEL: &str = "llama3.2";
pub const OLLAMA_LLM_BASE_URL: &str = "http://localhost:11434/v1";

pub const OLLAMA_EMBEDDER_MODEL: &str = "nomic-embed-text";
pub const OLLAMA_EMBEDDER_BASE_URL: &str = "http://localhost:11434";

pub const OPENAI_LLM_BASE_URL: &str = "https://api.openai.com/v1";
pub const OPENAI_EMBEDDER_BASE_URL: &str = "https://api.openai.com/v1";

pub const DEFAULT_WINDOW_SIZE: usize = 10;
pub const DEFAULT_SEMANTIC_TOP_K: usize = 5;
pub const DEFAULT_KEYWORD_TOP_K: usize = 3;
pub const DEFAULT_STRUCTURED_TOP_K: usize = 5;

pub const DEFAULT_KEYWORD_WEIGHT: f32 = 0.2;
pub const DEFAULT_SEMANTIC_WEIGHT: f32 = 0.8;
pub const DEFAULT_DEDUP_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Ollama,
    OpenAI,
}

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
    pub retrieval: RetrievalConfig,
    pub synthesizer: SynthesizerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub semantic_top_k: usize,
    pub keyword_top_k: usize,
    pub structured_top_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            semantic_top_k: DEFAULT_SEMANTIC_TOP_K,
            keyword_top_k: DEFAULT_KEYWORD_TOP_K,
            structured_top_k: DEFAULT_STRUCTURED_TOP_K,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizerConfig {
    pub keyword_weight: f32,
    pub semantic_weight: f32,
    pub dedup_threshold: f32,
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        Self {
            keyword_weight: DEFAULT_KEYWORD_WEIGHT,
            semantic_weight: DEFAULT_SEMANTIC_WEIGHT,
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_WINDOW_SIZE,
            overlap_size: 2,
            retrieval: RetrievalConfig::default(),
            synthesizer: SynthesizerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: Provider,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: Provider::Ollama,
            model: OLLAMA_LLM_MODEL.to_string(),
            base_url: Some(OLLAMA_LLM_BASE_URL.to_string()),
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: Provider,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub dimension: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: Provider::Ollama,
            model: OLLAMA_EMBEDDER_MODEL.to_string(),
            base_url: Some(OLLAMA_EMBEDDER_BASE_URL.to_string()),
            api_key: None,
            dimension: 768,
        }
    }
}
