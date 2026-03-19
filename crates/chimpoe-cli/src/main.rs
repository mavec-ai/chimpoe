mod buffer;
mod commands;
mod config;

use clap::{Parser, Subcommand};
use colored::Colorize;

#[derive(Parser)]
#[command(name = "chimpoe")]
#[command(about = "Long-term memory for AI agents", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init(commands::InitArgs),
    Add(commands::AddArgs),
    Finalize(commands::FinalizeArgs),
    Search(commands::SearchArgs),
    Ask(commands::AskArgs),
    List(commands::ListArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = config::CliConfig::load().unwrap_or_default();

    let result = match cli.command {
        Commands::Init(args) => commands::init::run(args, &config).await,
        Commands::Add(args) => commands::add::run(args, &config).await,
        Commands::Finalize(args) => commands::finalize::run(args, &config).await,
        Commands::Search(args) => commands::search::run(args, &config).await,
        Commands::Ask(args) => commands::ask::run(args, &config).await,
        Commands::List(args) => commands::list::run(args, &config).await,
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    Ok(())
}
