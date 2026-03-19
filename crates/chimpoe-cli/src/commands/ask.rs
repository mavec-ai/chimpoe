use crate::config::CliConfig;
use anyhow::Result;
use chimpoe_core::{
    Chimpoe,
    embed::OllamaEmbedder,
    llm::OllamaLlm,
    traits::{Embedder, LlmClient},
    vector::SqliteVector,
};
use clap::Args;
use colored::Colorize;
use std::sync::Arc;

#[derive(Args)]
pub struct AskArgs {
    #[arg(help = "Question to ask")]
    pub question: String,
}

pub async fn run(args: AskArgs, config: &CliConfig) -> Result<()> {
    config.ensure_directories()?;

    let vector_store = Arc::new(SqliteVector::new(
        &config.storage.path.to_string_lossy(),
        768,
    )?);

    let embedder_config = chimpoe_core::config::EmbeddingConfig {
        model: config.embedder.model.clone(),
        base_url: Some(config.embedder.base_url.clone()),
        dimension: 768,
    };
    let embedder: Arc<dyn Embedder> = Arc::new(OllamaEmbedder::new(&embedder_config));

    let llm_config = chimpoe_core::config::LlmConfig {
        model: config.llm.model.clone(),
        base_url: Some(config.llm.base_url.clone()),
        temperature: 0.7,
    };
    let llm: Arc<dyn LlmClient> = Arc::new(OllamaLlm::new(&llm_config));

    let core_config = chimpoe_core::Config {
        pipeline: chimpoe_core::config::PipelineConfig {
            retrieval: chimpoe_core::config::RetrievalConfig {
                semantic_top_k: config.memory.semantic_top_k,
                keyword_top_k: config.memory.keyword_top_k,
                structured_top_k: config.memory.structured_top_k,
            },
            ..Default::default()
        },
        embedding: embedder_config,
        llm: llm_config,
    };

    let chimpoe = Chimpoe::builder()
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .config(core_config)
        .build()
        .await?;

    println!("{}", format!("Q: {}", args.question).cyan().bold());
    println!();

    let answer = chimpoe.ask(&args.question).await?;

    println!("{}\n", answer);

    Ok(())
}
