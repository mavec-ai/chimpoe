use crate::buffer;
use crate::config::CliConfig;
use anyhow::Result;
use chimpoe_core::types::Dialogue;
use clap::Args;
use colored::Colorize;

#[derive(Args)]
pub struct AddArgs {
    #[arg(help = "Dialogue content")]
    pub content: String,

    #[arg(
        short,
        long,
        default_value = "user",
        help = "Speaker name (default: 'user')"
    )]
    pub speaker: String,

    #[arg(
        short,
        long,
        help = "Optional timestamp (e.g., 'yesterday', '2024-01-15')"
    )]
    pub timestamp: Option<String>,
}

pub async fn run(args: AddArgs, config: &CliConfig) -> Result<()> {
    config.ensure_directories()?;

    let mut dialogues = buffer::load()?;

    let mut dialogue = Dialogue::new(&args.speaker, &args.content);
    if let Some(ts) = args.timestamp.as_deref() {
        dialogue = dialogue.with_timestamp(ts);
    }
    dialogues.push(dialogue);

    buffer::save(&dialogues)?;

    let buffer_count = dialogues.len();
    let window_size = config.memory.window_size;

    println!("{} Dialogue added", "✓".green());
    println!("  Speaker: {}", args.speaker.cyan());
    println!("  Content: {}", args.content);
    println!("\n  Buffer: {}/{} dialogues", buffer_count, window_size);

    if buffer_count >= window_size {
        println!(
            "  {} Buffer full, run {} to process",
            "→".yellow(),
            "chimpoe finalize".yellow()
        );
    }

    Ok(())
}
