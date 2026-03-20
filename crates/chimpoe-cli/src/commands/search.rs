use crate::config::CliConfig;
use crate::providers;
use anyhow::Result;
use chimpoe_core::{Chimpoe, traits::Embedder, vector::SqliteVector};
use clap::Args;
use colored::Colorize;
use std::sync::Arc;
use tabled::{Table, Tabled, settings::Style};

#[derive(Args)]
pub struct SearchArgs {
    #[arg(help = "Search query")]
    pub query: String,

    #[arg(short = 'k', long, default_value = "5", help = "Number of results")]
    pub top_k: usize,
}

#[derive(Tabled)]
struct MemoryRow {
    #[tabled(rename = "#")]
    id: usize,
    #[tabled(rename = "Memory")]
    memory: String,
    #[tabled(rename = "Source")]
    source: String,
}

pub async fn run(args: SearchArgs, config: &CliConfig) -> Result<()> {
    config.ensure_directories()?;

    let vector_store = Arc::new(SqliteVector::new(
        &config.storage.path.to_string_lossy(),
        config.embedder.dimension,
    )?);

    let embedder: Arc<dyn Embedder> = providers::create_embedder(config)?;
    let llm = providers::create_llm(config)?;

    let pipeline_config = chimpoe_core::config::PipelineConfig {
        retrieval: chimpoe_core::config::RetrievalConfig {
            semantic_top_k: args.top_k,
            keyword_top_k: config.memory.keyword_top_k,
            structured_top_k: config.memory.structured_top_k,
        },
        ..Default::default()
    };

    let core_config = chimpoe_core::Config {
        pipeline: pipeline_config,
        ..Default::default()
    };

    let chimpoe = Chimpoe::builder()
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .config(core_config)
        .build()
        .await?;

    println!("{}\n", format!("Searching: \"{}\"", args.query).cyan());

    let result = chimpoe.search(&args.query, Some(args.top_k)).await?;

    if result.results.is_empty() {
        println!("  {} No memories found", "!".yellow());
        return Ok(());
    }

    let rows: Vec<MemoryRow> = result
        .results
        .iter()
        .enumerate()
        .map(|(i, hit)| MemoryRow {
            id: i + 1,
            memory: if hit.memory.len() > 60 {
                format!("{}...", &hit.memory[..57])
            } else {
                hit.memory.clone()
            },
            source: hit.source.clone(),
        })
        .collect();

    let table = Table::new(rows).with(Style::rounded()).to_string();
    println!("{}", table);

    println!("\n  Found {} memories", result.results.len());

    Ok(())
}
