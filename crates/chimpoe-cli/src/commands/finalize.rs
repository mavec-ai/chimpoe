use crate::buffer;
use crate::config::CliConfig;
use crate::providers;
use anyhow::Result;
use chimpoe::{Chimpoe, traits::Embedder, vector::SqliteVector};
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
        config.embedder.dimension,
    )?);

    let embedder: Arc<dyn Embedder> = providers::create_embedder(config)?;
    let llm = providers::create_llm(config)?;

    let pipeline_config = chimpoe::config::PipelineConfig {
        window_size: config.memory.window_size,
        ..Default::default()
    };

    let core_config = chimpoe::Config {
        pipeline: pipeline_config,
        ..Default::default()
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
