use crate::config::{
    CliConfig, EmbedderConfig, LlmConfig, MemoryConfig, StorageConfig, chimpoe_dir,
};
use anyhow::Result;
use chimpoe::config::{
    OLLAMA_EMBEDDER_BASE_URL, OLLAMA_LLM_BASE_URL, OPENAI_EMBEDDER_BASE_URL, OPENAI_LLM_BASE_URL,
};
use clap::Args;
use colored::Colorize;
use dialoguer::{Input, Password, Select};
use std::fs;

const OLLAMA_LLM_MODELS: &[&str] = &["llama3.2", "llama3.1", "llama3", "mistral", "qwen2.5"];
const OPENAI_LLM_MODELS: &[&str] = &["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"];
const OLLAMA_EMBED_MODELS: &[&str] = &["nomic-embed-text", "mxbai-embed-large", "all-minilm"];
const OPENAI_EMBED_MODELS: &[&str] = &[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
];

const OLLAMA_EMBED_DIMENSIONS: &[usize] = &[768, 1024, 384];
const OPENAI_EMBED_DIMENSIONS: &[usize] = &[1536, 3072, 1536];

#[derive(Args)]
pub struct InitArgs {
    #[arg(long, help = "Skip interactive mode, use defaults")]
    pub yes: bool,
    #[arg(long, help = "Force re-initialization, overwrite existing config")]
    pub force: bool,
}

pub async fn run(args: InitArgs) -> Result<()> {
    println!("\n{}", " Welcome to Chimpoe! ".black().on_cyan().bold());
    println!("{}\n", "Let's set up your memory system.".dimmed());

    crate::config::ensure_directories()?;

    let config_exists = crate::config::config_path().exists();
    if args.yes && !args.force && config_exists {
        let config = CliConfig::load()?;
        print_summary(&config);
        return Ok(());
    }

    if config_exists && !args.force {
        println!(
            "{} Config already exists. Use --force to reinitialize.",
            "⚠".yellow()
        );
        let config = CliConfig::load()?;
        print_summary(&config);
        return Ok(());
    }

    if args.force {
        let db_path = chimpoe_dir().join("chimpoe.db");
        if db_path.exists() {
            fs::remove_file(&db_path)?;
            println!(
                "{} Database reset (dimension change requires fresh DB)",
                "✓".green()
            );
        }
    }

    let llm_provider = select_provider("LLM", &["Ollama", "OpenAI"])?;
    let llm_provider_name = if llm_provider == 0 {
        "ollama"
    } else {
        "openai"
    };

    let llm_models = if llm_provider == 0 {
        OLLAMA_LLM_MODELS
    } else {
        OPENAI_LLM_MODELS
    };
    let (_, llm_model) = select_model("LLM", llm_models)?;
    let llm_api_key = if llm_provider == 1 {
        prompt_api_key("OpenAI")?
    } else {
        None
    };

    let embedder_provider = select_provider("Embedder", &["Ollama", "OpenAI"])?;
    let embedder_provider_name = if embedder_provider == 0 {
        "ollama"
    } else {
        "openai"
    };

    let embed_models = if embedder_provider == 0 {
        OLLAMA_EMBED_MODELS
    } else {
        OPENAI_EMBED_MODELS
    };
    let (embedder_model_idx, embedder_model) = select_model("Embedder", embed_models)?;
    let embedder_api_key = if embedder_provider == 1 {
        if llm_provider == 1 && llm_api_key.is_some() {
            llm_api_key.clone()
        } else {
            prompt_api_key("OpenAI")?
        }
    } else {
        None
    };

    let embedder_dimension = if embedder_provider == 0 {
        OLLAMA_EMBED_DIMENSIONS[embedder_model_idx]
    } else {
        OPENAI_EMBED_DIMENSIONS[embedder_model_idx]
    };

    let window_size = prompt_window_size()?;

    let config = CliConfig {
        llm: LlmConfig {
            provider: llm_provider_name.to_string(),
            base_url: if llm_provider == 0 {
                OLLAMA_LLM_BASE_URL.to_string()
            } else {
                OPENAI_LLM_BASE_URL.to_string()
            },
            model: llm_model.to_string(),
            api_key: llm_api_key,
        },
        embedder: EmbedderConfig {
            provider: embedder_provider_name.to_string(),
            base_url: if embedder_provider == 0 {
                OLLAMA_EMBEDDER_BASE_URL.to_string()
            } else {
                OPENAI_EMBEDDER_BASE_URL.to_string()
            },
            model: embedder_model.to_string(),
            api_key: embedder_api_key,
            dimension: embedder_dimension,
        },
        storage: StorageConfig::default(),
        memory: MemoryConfig {
            window_size,
            ..MemoryConfig::default()
        },
    };

    println!("\n{}", "Testing connection...".cyan());
    if llm_provider_name == "ollama" || embedder_provider_name == "ollama" {
        check_ollama(&config).await?;
    }

    config.save()?;
    println!("{} Config saved to ~/.chimpoe/config.toml", "✓".green());

    print_summary(&config);
    Ok(())
}

fn select_provider(name: &str, items: &[&str]) -> Result<usize> {
    let selection = Select::new()
        .with_prompt(format!("Choose your {name} provider"))
        .items(items)
        .default(0)
        .interact()?;
    Ok(selection)
}

fn select_model<'a>(name: &str, models: &'a [&str]) -> Result<(usize, &'a str)> {
    let selection = Select::new()
        .with_prompt(format!("Select {name} model"))
        .items(models)
        .default(0)
        .interact()?;
    Ok((selection, models[selection]))
}

fn prompt_api_key(provider: &str) -> Result<Option<String>> {
    let api_key: String = Password::new()
        .with_prompt(format!("Enter {provider} API Key"))
        .interact()?;
    if api_key.is_empty() {
        Ok(None)
    } else {
        Ok(Some(api_key))
    }
}

fn prompt_window_size() -> Result<usize> {
    let input: String = Input::new()
        .with_prompt("Memory window size (dialogues before processing)")
        .default("10".to_string())
        .interact()?;
    Ok(input.parse().unwrap_or(10))
}

async fn check_ollama(config: &CliConfig) -> Result<()> {
    let client = reqwest::Client::new();
    let base_url = config.embedder.base_url.trim_end_matches("/api/embed");
    let tags_url = format!("{base_url}/api/tags");

    match client.get(&tags_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("{} Ollama is running at {}", "✓".green(), base_url);

            let body = resp.text().await?;
            let models: Vec<String> = parse_ollama_models(&body);

            let llm_missing = config.llm.provider == "ollama"
                && !models.iter().any(|m| m.starts_with(&config.llm.model));
            let embedder_missing = config.embedder.provider == "ollama"
                && !models.iter().any(|m| m.starts_with(&config.embedder.model));

            if llm_missing || embedder_missing {
                println!();
            }

            if llm_missing {
                println!(
                    "{} Model '{}' not found locally",
                    "⚠".yellow(),
                    config.llm.model
                );
                println!("  Run: ollama pull {}", config.llm.model);
            }

            if embedder_missing {
                println!(
                    "{} Model '{}' not found locally",
                    "⚠".yellow(),
                    config.embedder.model
                );
                println!("  Run: ollama pull {}", config.embedder.model);
            }
        }
        _ => {
            println!("{} Cannot connect to Ollama at {}", "⚠".yellow(), base_url);
            println!("  Make sure Ollama is running: ollama serve");
        }
    }
    Ok(())
}

fn parse_ollama_models(json: &str) -> Vec<String> {
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        _ => return vec![],
    };

    value
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.get("name")
                        .and_then(|n| n.as_str())
                        .map(std::string::ToString::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn pad_to_width(s: &str, width: usize) -> String {
    let len = s.len();
    if len >= width - 2 {
        format!(" {}", &s[..width.saturating_sub(2)])
    } else {
        format!(" {:width$} ", s, width = width - 2)
    }
}

fn print_summary(config: &CliConfig) {
    let width = 50;
    println!("\n{}", " Configuration ".black().on_green().bold());

    let border = "─".repeat(width);
    println!("┌{border}┐");

    let llm = format!("LLM: {} ({})", config.llm.model, config.llm.provider);
    let embedder = format!(
        "Embedder: {} ({})",
        config.embedder.model, config.embedder.provider
    );
    let window = format!("Window Size: {}", config.memory.window_size);

    println!("│{}│", pad_to_width(&llm, width));
    println!("│{}│", pad_to_width(&embedder, width));
    println!("│{}│", pad_to_width(&window, width));
    println!("└{border}┘");

    println!("\n{}", "Chimpoe is ready!".green().bold());
    println!("\nQuick start:");
    println!("  {} Add a dialogue", "chimpoe add \"Hello!\"".yellow());
    println!("  {} Process memories", "chimpoe finalize".yellow());
    println!("  {} Search memories", "chimpoe search \"hello\"".yellow());
    println!(
        "  {} Ask about memories",
        "chimpoe ask \"What did I say about X?\"".yellow()
    );
}
