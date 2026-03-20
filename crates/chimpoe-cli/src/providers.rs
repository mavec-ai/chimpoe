use crate::config::CliConfig;
use anyhow::{Result, anyhow};
use chimpoe_core::{
    config::{EmbeddingConfig, LlmConfig, Provider},
    embed::{OllamaEmbedder, OpenAIEmbedder},
    llm::{OllamaLlm, OpenAILlm},
    traits::{Embedder, LlmClient},
};
use std::sync::Arc;

pub fn create_embedder(config: &CliConfig) -> Result<Arc<dyn Embedder>> {
    let provider = match config.embedder.provider.as_str() {
        "ollama" => Provider::Ollama,
        "openai" => Provider::OpenAI,
        p => return Err(anyhow!("Unknown embedder provider: {}", p)),
    };

    let embedder_config = EmbeddingConfig {
        provider,
        model: config.embedder.model.clone(),
        base_url: Some(config.embedder.base_url.clone()),
        api_key: config.embedder.api_key.clone(),
        dimension: config.embedder.dimension,
    };

    match provider {
        Provider::Ollama => Ok(Arc::new(OllamaEmbedder::new(&embedder_config))),
        Provider::OpenAI => Ok(Arc::new(OpenAIEmbedder::new(&embedder_config))),
    }
}

pub fn create_llm(config: &CliConfig) -> Result<Arc<dyn LlmClient>> {
    let provider = match config.llm.provider.as_str() {
        "ollama" => Provider::Ollama,
        "openai" => Provider::OpenAI,
        p => return Err(anyhow!("Unknown LLM provider: {}", p)),
    };

    let llm_config = LlmConfig {
        provider,
        model: config.llm.model.clone(),
        base_url: Some(config.llm.base_url.clone()),
        api_key: config.llm.api_key.clone(),
        temperature: config.llm.temperature,
    };

    match provider {
        Provider::Ollama => Ok(Arc::new(OllamaLlm::new(&llm_config))),
        Provider::OpenAI => Ok(Arc::new(OpenAILlm::new(&llm_config))),
    }
}
