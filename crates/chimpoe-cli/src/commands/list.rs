use crate::config::CliConfig;
use crate::providers;
use anyhow::Result;
use chimpoe::{Chimpoe, traits::Embedder, vector::SqliteVector};
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
    crate::config::ensure_directories()?;

    let vector_store = Arc::new(SqliteVector::new(
        &config.storage.path.to_string_lossy(),
        config.embedder.dimension,
    )?);

    let embedder: Arc<dyn Embedder> = providers::create_embedder(config)?;
    let llm = providers::create_llm(config)?;

    let core_config = chimpoe::Config::default();

    let chimpoe = Chimpoe::builder()
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .config(core_config)
        .build()
        .await?;

    let memories = chimpoe.list_memories().await?;
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
                .map_or_else(|| "-".to_string(), |t| t.format("%Y-%m-%d").to_string()),
        })
        .collect();

    println!("{}", "Stored Memories".cyan().bold());
    println!();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{table}");

    if total > args.limit {
        println!("\n  Showing {} of {} memories", args.limit, total);
    } else {
        println!("\n  Total: {total} memories");
    }

    Ok(())
}
