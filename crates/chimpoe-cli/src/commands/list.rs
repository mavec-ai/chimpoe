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
use tabled::{Table, Tabled, settings::Style};

#[derive(Args)]
pub struct ListArgs {
    #[arg(
        short = 'l',
        long,
        default_value = "20",
        help = "Limit number of results"
    )]
    pub limit: usize,
}

#[derive(Tabled)]
struct MemoryRow {
    #[tabled(rename = "#")]
    id: usize,
    #[tabled(rename = "Memory")]
    memory: String,
    #[tabled(rename = "Topic")]
    topic: String,
    #[tabled(rename = "Timestamp")]
    timestamp: String,
}

pub async fn run(args: ListArgs, config: &CliConfig) -> Result<()> {
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
        embedding: embedder_config,
        llm: llm_config,
        ..Default::default()
    };

    let chimpoe = Chimpoe::builder()
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .config(core_config)
        .build()
        .await?;

    let memories = chimpoe.list_memories().await;
    let total = memories.len();

    if total == 0 {
        println!("  {} No memories stored yet", "!".yellow());
        println!("  Add dialogues with: {}", "chimpoe add <content>".yellow());
        println!("  Then process with: {}", "chimpoe finalize".yellow());
        return Ok(());
    }

    let display_memories: Vec<_> = memories.into_iter().take(args.limit).collect();

    let rows: Vec<MemoryRow> = display_memories
        .iter()
        .enumerate()
        .map(|(i, m)| MemoryRow {
            id: i + 1,
            memory: if m.lossless_restatement.len() > 50 {
                format!("{}...", &m.lossless_restatement[..47])
            } else {
                m.lossless_restatement.clone()
            },
            topic: m.topic.as_deref().unwrap_or("-").to_string(),
            timestamp: m
                .timestamp
                .map(|t| t.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "-".to_string()),
        })
        .collect();

    println!("{}", "Stored Memories".cyan().bold());
    println!();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);

    if total > args.limit {
        println!("\n  Showing {} of {} memories", args.limit, total);
    } else {
        println!("\n  Total: {} memories", total);
    }

    Ok(())
}
