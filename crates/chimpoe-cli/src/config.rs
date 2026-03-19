use anyhow::{Context, Result};
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
    "http://localhost:11434/v1".to_string()
}

fn default_llm_model() -> String {
    "llama3.2".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    #[serde(default = "default_embedder_provider")]
    pub provider: String,
    #[serde(default = "default_embedder_base_url")]
    pub base_url: String,
    #[serde(default = "default_embedder_model")]
    pub model: String,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: default_embedder_provider(),
            base_url: default_embedder_base_url(),
            model: default_embedder_model(),
        }
    }
}

fn default_embedder_provider() -> String {
    "ollama".to_string()
}

fn default_embedder_base_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_embedder_model() -> String {
    "nomic-embed-text".to_string()
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
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            window_size: default_window_size(),
            semantic_top_k: default_semantic_top_k(),
            keyword_top_k: default_keyword_top_k(),
            structured_top_k: default_structured_top_k(),
        }
    }
}

fn default_window_size() -> usize {
    10
}

fn default_semantic_top_k() -> usize {
    5
}

fn default_keyword_top_k() -> usize {
    3
}

fn default_structured_top_k() -> usize {
    5
}

impl CliConfig {
    pub fn load() -> Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config from {:?}", path))?;

        toml::from_str(&content).with_context(|| format!("Failed to parse config from {:?}", path))
    }

    pub fn save(&self) -> Result<()> {
        let dir = chimpoe_dir();
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory {:?}", dir))?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;

        fs::write(config_path(), content).context("Failed to write config file")?;

        Ok(())
    }

    pub fn ensure_directories(&self) -> Result<()> {
        let dir = chimpoe_dir();
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory {:?}", dir))?;
        }
        Ok(())
    }
}
