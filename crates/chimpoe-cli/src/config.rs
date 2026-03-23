use anyhow::{Context, Result};
use chimpoe::config::{
    DEFAULT_DEDUP_THRESHOLD, DEFAULT_KEYWORD_TOP_K, DEFAULT_SEMANTIC_TOP_K,
    DEFAULT_STRUCTURED_TOP_K, DEFAULT_WINDOW_SIZE, OLLAMA_EMBEDDER_BASE_URL, OLLAMA_EMBEDDER_MODEL,
    OLLAMA_LLM_BASE_URL, OLLAMA_LLM_MODEL,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

pub fn chimpoe_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".chimpoe")
}

pub fn config_path() -> PathBuf {
    chimpoe_dir().join("config.toml")
}

pub fn ensure_directories() -> Result<()> {
    let dir = chimpoe_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create directory {}", dir.display()))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CliConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub embedder: EmbedderConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,
    #[serde(default = "default_llm_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            base_url: default_llm_base_url(),
            model: default_llm_model(),
            api_key: None,
        }
    }
}

fn default_llm_provider() -> String {
    "ollama".to_string()
}

fn default_llm_base_url() -> String {
    OLLAMA_LLM_BASE_URL.to_string()
}

fn default_llm_model() -> String {
    OLLAMA_LLM_MODEL.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    #[serde(default = "default_embedder_provider")]
    pub provider: String,
    #[serde(default = "default_embedder_base_url")]
    pub base_url: String,
    #[serde(default = "default_embedder_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_embedder_dimension")]
    pub dimension: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: default_embedder_provider(),
            base_url: default_embedder_base_url(),
            model: default_embedder_model(),
            api_key: None,
            dimension: default_embedder_dimension(),
        }
    }
}

const fn default_embedder_dimension() -> usize {
    768
}

fn default_embedder_provider() -> String {
    "ollama".to_string()
}

fn default_embedder_base_url() -> String {
    OLLAMA_EMBEDDER_BASE_URL.to_string()
}

fn default_embedder_model() -> String {
    OLLAMA_EMBEDDER_MODEL.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_storage_provider")]
    pub provider: String,
    #[serde(default = "default_storage_path")]
    pub path: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            provider: default_storage_provider(),
            path: default_storage_path(),
        }
    }
}

fn default_storage_provider() -> String {
    "sqlite".to_string()
}

fn default_storage_path() -> PathBuf {
    chimpoe_dir().join("chimpoe.db")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_semantic_top_k")]
    pub semantic_top_k: usize,
    #[serde(default = "default_keyword_top_k")]
    pub keyword_top_k: usize,
    #[serde(default = "default_structured_top_k")]
    pub structured_top_k: usize,
    #[serde(default = "default_dedup_threshold")]
    pub dedup_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            window_size: default_window_size(),
            semantic_top_k: default_semantic_top_k(),
            keyword_top_k: default_keyword_top_k(),
            structured_top_k: default_structured_top_k(),
            dedup_threshold: default_dedup_threshold(),
        }
    }
}

const fn default_window_size() -> usize {
    DEFAULT_WINDOW_SIZE
}

const fn default_semantic_top_k() -> usize {
    DEFAULT_SEMANTIC_TOP_K
}

const fn default_keyword_top_k() -> usize {
    DEFAULT_KEYWORD_TOP_K
}

const fn default_structured_top_k() -> usize {
    DEFAULT_STRUCTURED_TOP_K
}

const fn default_dedup_threshold() -> f32 {
    DEFAULT_DEDUP_THRESHOLD
}

impl CliConfig {
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))
    }

    pub fn save(&self) -> Result<()> {
        let dir = chimpoe_dir();
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory {}", dir.display()))?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;

        fs::write(config_path(), content).context("Failed to write config file")?;

        Ok(())
    }
}
