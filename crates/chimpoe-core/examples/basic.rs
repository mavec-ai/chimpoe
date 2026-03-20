use chimpoe::Chimpoe;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chimpoe Demo (InMemoryVector) ===\n");

    println!("1. Initializing Chimpoe (using defaults)...");
    println!("   Note: For persistent storage, use:");
    println!(
        "   Chimpoe::builder().vector_store(Arc::new(SqliteVector::new(\"memories.db\", 768)?)).build()"
    );
    let mut chimpoe = Chimpoe::new().await?;
    println!("   Ready!\n");

    println!("2. Adding dialogues");
    chimpoe
        .add_dialogue(
            "Alice",
            "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product.",
            None,
        )
        .await?;
    chimpoe
        .add_dialogue("Bob", "Okay, I'll prepare the materials.", None)
        .await?;
    chimpoe
        .add_dialogue(
            "Alice",
            "Remember to bring the market research report.",
            Some("2025-01-15T10:30:00"),
        )
        .await?;
    println!("   Added 3 dialogues\n");

    println!("3. Finalizing (process remaining buffer)...");
    match chimpoe.finalize().await {
        Ok(count) => println!("   Extracted {} memories\n", count),
        Err(e) => {
            println!("   Failed: {}", e);
            return Ok(());
        }
    }

    println!("4. Searching memories...");
    let results = chimpoe.search("what does Alice like?", Some(3)).await?;
    println!("{}\n", results);

    println!("5. Asking a question (uses LLM + memories)...");
    match chimpoe.ask("Tell me about Alice's preferences").await {
        Ok(answer) => println!("   Answer: {}\n", answer),
        Err(e) => println!("   Failed: {}\n", e),
    }

    println!("6. Stats:");
    println!("   Memories stored: {}", chimpoe.memory_count().await);

    println!("\n7. All memories in DB:");
    for (i, m) in chimpoe.list_memories().await.iter().enumerate() {
        println!("\n   [{}]", i + 1);
        println!("   ID: {}", m.entry_id);
        println!("   Restatement: {}", m.lossless_restatement);
        println!("   Keywords: {:?}", m.keywords);
        println!("   Persons: {:?}", m.persons);
        println!("   Entities: {:?}", m.entities);
        println!("   Location: {:?}", m.location);
        println!("   Topic: {:?}", m.topic);
        println!("   Timestamp: {:?}", m.timestamp);
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}
