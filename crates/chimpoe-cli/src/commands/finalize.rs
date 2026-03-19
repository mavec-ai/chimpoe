use crate::buffer;
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
pub struct FinalizeArgs;

pub async fn run(_args: FinalizeArgs, config: &CliConfig) -> Result<()> {
    config.ensure_directories()?;

    let dialogues = buffer::load()?;

    if dialogues.is_empty() {
        println!("  {} No dialogues in buffer to process", "!".yellow());
        println!(
            "    Add dialogues with: {}",
            "chimpoe add <content>".yellow()
        );
        return Ok(());
    }

    println!("{}", "Processing dialogues...".cyan());
    println!("  {} dialogues in buffer", dialogues.len());

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

    let pipeline_config = chimpoe_core::config::PipelineConfig {
        window_size: config.memory.window_size,
        ..Default::default()
    };

    let core_config = chimpoe_core::Config {
        pipeline: pipeline_config,
        embedding: embedder_config,
        llm: llm_config,
    };

    let mut chimpoe = Chimpoe::builder()
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .config(core_config)
        .build()
        .await?;

    chimpoe.add_dialogues(dialogues.clone()).await?;

    let memories_created = chimpoe.finalize().await?;

    if memories_created > 0 {
        println!(
            "  {} Created {} new memories",
            "✓".green(),
            memories_created
        );
        buffer::clear()?;
        println!("  {} Buffer cleared", "✓".green());
    } else {
        println!("  {} No new memories extracted", "!".yellow());
    }

    let total_memories = chimpoe.memory_count().await;
    println!("\n  Total memories: {}", total_memories);

    Ok(())
}
