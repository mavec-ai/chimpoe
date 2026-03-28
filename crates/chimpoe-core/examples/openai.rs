use chimpoe::{
    Chimpoe, Config, EmbeddingConfig, LlmConfig, OpenAIEmbedder, OpenAILlm, PipelineConfig,
    Provider, RetrievalConfig, SqliteVector, SynthesizerConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chimpoe Demo (OpenAI + Custom Config) ===\n");

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("Note: Set OPENAI_API_KEY environment variable");
        "your-api-key-here".to_string()
    });

    println!("1. Building custom config...");
    let config = Config {
        pipeline: PipelineConfig {
            window_size: 5,
            overlap_size: 1,
            retrieval: RetrievalConfig {
                semantic_top_k: 10,
                keyword_top_k: 5,
                structured_top_k: 8,
            },
            synthesizer: SynthesizerConfig::default(),
        },
        llm: LlmConfig {
            provider: Provider::OpenAI,
            model: "gpt-4o-mini".to_string(),
            base_url: None,
            api_key: Some(api_key.clone()),
        },
        embedding: EmbeddingConfig {
            provider: Provider::OpenAI,
            model: "text-embedding-3-small".to_string(),
            base_url: None,
            api_key: Some(api_key),
            dimension: 1536,
        },
    };
    println!("   window_size: {}", config.pipeline.window_size);
    println!("   llm_model: {}", config.llm.model);
    println!("   embed_model: {}", config.embedding.model);
    println!("   dimension: {}", config.embedding.dimension);
    println!();

    println!("2. Initializing with custom providers...");
    let vector_store = Arc::new(SqliteVector::in_memory(config.embedding.dimension)?);
    let embedder = Arc::new(OpenAIEmbedder::new(&config.embedding));
    let llm = Arc::new(OpenAILlm::new(&config.llm));

    let mut chimpoe = Chimpoe::builder()
        .config(config)
        .vector_store(vector_store)
        .embedder(embedder)
        .llm(llm)
        .build()
        .await?;
    println!("   Ready!\n");

    println!("3. Adding dialogues...");
    chimpoe
        .add_dialogue(
            "User",
            "I prefer dark mode for all my applications, it's easier on my eyes.",
            None,
        )
        .await?;
    chimpoe
        .add_dialogue(
            "Assistant",
            "I'll remember that! Dark mode preference noted.",
            None,
        )
        .await?;
    chimpoe
        .add_dialogue(
            "User",
            "Also, I work as a software engineer at TechCorp in Jakarta.",
            None,
        )
        .await?;
    chimpoe
        .add_dialogue("User", "My favorite programming language is Rust.", None)
        .await?;
    chimpoe
        .add_dialogue(
            "User",
            "I have a meeting with the team every Monday at 9 AM.",
            None,
        )
        .await?;
    println!("   Added 5 dialogues\n");

    println!("4. Auto-processing (window_size=5 reached)...\n");

    println!("5. Searching memories...");
    let results = chimpoe.search("programming preferences", Some(3)).await?;
    println!("{results}\n");

    println!("6. Asking a question...");
    match chimpoe.ask("What do you know about my work?").await {
        Ok(answer) => println!("   Answer: {answer}\n"),
        Err(e) => println!("   Failed: {e}\n"),
    }

    println!("7. Stats:");
    println!("   Memories stored: {}", chimpoe.memory_count().await?);

    println!("\n=== Demo Complete ===");
    Ok(())
}
