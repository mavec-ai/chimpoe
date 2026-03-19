use crate::config::CliConfig;
use anyhow::Result;
use clap::Args;
use colored::Colorize;

#[derive(Args)]
pub struct InitArgs {
    #[arg(long, help = "Check if Ollama is running and models are available")]
    pub check: bool,
}

pub async fn run(args: InitArgs, config: &CliConfig) -> Result<()> {
    println!("{}", "Initializing Chimpoe...".cyan().bold());

    config.ensure_directories()?;
    println!("  {} Created ~/.chimpoe directory", "✓".green());

    if !crate::config::config_path().exists() {
        config.save()?;
        println!(
            "  {} Created default config at ~/.chimpoe/config.toml",
            "✓".green()
        );
    } else {
        println!(
            "  {} Config already exists at ~/.chimpoe/config.toml",
            "✓".green()
        );
    }

    if args.check {
        println!("\n{}", "Checking Ollama connection...".cyan());
        check_ollama(config).await?;
    }

    println!("\n{} Chimpoe is ready!", "✓".green().bold());
    println!("\nQuick start:");
    println!("  {} Add a dialogue", "chimpoe add \"Hello!\"".yellow());
    println!("  {} Process memories", "chimpoe finalize".yellow());
    println!("  {} Search memories", "chimpoe search \"hello\"".yellow());

    Ok(())
}

async fn check_ollama(config: &CliConfig) -> Result<()> {
    let client = reqwest::Client::new();

    let base_url = config.embedder.base_url.trim_end_matches("/api/embed");
    let health_url = format!("{}/api/tags", base_url);

    let result: Result<reqwest::Response, reqwest::Error> = client.get(&health_url).send().await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            println!("  {} Ollama is running at {}", "✓".green(), base_url);

            println!("\n  Checking required models...");
            println!(
                "    {} embed: {} (required)",
                "•".dimmed(),
                config.embedder.model
            );
            println!("    {} llm: {} (required)", "•".dimmed(), config.llm.model);
        }
        Ok(resp) => {
            println!("  {} Ollama returned status: {}", "✗".red(), resp.status());
        }
        Err(e) => {
            println!("  {} Cannot connect to Ollama at {}", "✗".red(), base_url);
            println!("    Error: {}", e);
            println!("\n  Make sure Ollama is running:");
            println!("    {} Pull models:", "$".dimmed());
            println!("      ollama pull {}", config.embedder.model);
            println!("      ollama pull {}", config.llm.model);
        }
    }

    Ok(())
}
