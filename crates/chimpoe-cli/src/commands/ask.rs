use crate::config::CliConfig;
use crate::providers;
use anyhow::Result;
use chimpoe::{Chimpoe, traits::Embedder, vector::SqliteVector};
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
        config.embedder.dimension,
    )?);

    let embedder: Arc<dyn Embedder> = providers::create_embedder(config)?;
    let llm = providers::create_llm(config)?;

    let core_config = chimpoe::Config {
        pipeline: chimpoe::config::PipelineConfig {
            retrieval: chimpoe::config::RetrievalConfig {
                semantic_top_k: config.memory.semantic_top_k,
                keyword_top_k: config.memory.keyword_top_k,
                structured_top_k: config.memory.structured_top_k,
            },
            ..Default::default()
        },
        ..Default::default()
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
