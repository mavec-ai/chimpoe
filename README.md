# Chimpoe

Long-term memory for AI agents.

## Installation

```bash
# Build from source
git clone https://github.com/mavec-ai/chimpoe
cd chimpoe
cargo install --path crates/chimpoe-cli
```

## CLI Usage

```bash
# Initialize config
chimpoe init

# Add dialogues (auto-processes when window_size=10)
chimpoe add "I'm working on a new API endpoint for user authentication"
chimpoe add -s "Mila" "Should we use JWT or OAuth2 for the implementation?"
chimpoe add -s "Deni" -t "2025-01-15T14:31:00" "Let's go with JWT, it's simpler for our use case"

# Force process remaining buffer
chimpoe finalize

# Search memories (hybrid: semantic + keyword + structured)
chimpoe search "authentication"
chimpoe search "API endpoint" -k 10

# Ask a question (retrieval + LLM answer)
chimpoe ask "What did Deni decide about the authentication approach?"

# List all memories
chimpoe list
```

## API Usage

```rust
use std::sync::Arc;
use chimpoe::{Chimpoe, Config, traits::{Embedder, LlmClient, VectorStore}};
use chimpoe::{OllamaEmbedder, OllamaLlm, SqliteVector};

// Setup dependencies
let embedder = Arc::new(OllamaEmbedder::new(&Default::default())) as Arc<dyn Embedder>;
let llm = Arc::new(OllamaLlm::new(&Default::default())) as Arc<dyn LlmClient>;
let vector_store = Arc::new(SqliteVector::new("memories.db", 768)?) as Arc<dyn VectorStore>;

// Build memory system
let mut memory = Chimpoe::builder()
    .embedder(embedder)
    .llm(llm)
    .vector_store(vector_store)
    .config(Config::default())
    .build()
    .await?;

// Add dialogues (auto-processes when buffer >= window_size)
memory.add_dialogue("Deni", "I'm working on a new API endpoint for authentication", None).await?;
memory.add_dialogue("Mila", "Should we use JWT or OAuth2?", None).await?;
memory.add_dialogue("Deni", "Let's go with JWT", Some("2025-01-15T14:31:00")).await?;

// Process remaining buffer
memory.finalize().await?;

// Search memories
let result = memory.search("authentication approach", Some(5)).await?;

// Ask a question (retrieval + LLM answer)
let answer = memory.ask("What did Deni decide about authentication?").await?;

// List all stored memories
let all_memories = memory.list_memories().await;
```

## How It Works

### Stage 1: Compress

Raw dialogues are transformed into structured memory entries via LLM.

**Input:**
```
[2025-01-15T14:30:00] Deni: I'm working on a new API endpoint for authentication
[2025-01-15T14:31:00] Mila: Should we use JWT or OAuth2?
[2025-01-15T14:32:00] Deni: Let's go with JWT, it's simpler
```

**Output:**
```json
{
  "lossless_restatement": "Deni is developing an API endpoint for authentication. Mila and Deni discussed JWT vs OAuth2 on 2025-01-15. Deni chose JWT for simplicity.",
  "keywords": ["API", "authentication", "JWT", "OAuth2", "Deni", "Mila"],
  "persons": ["Deni", "Mila"],
  "entities": ["API", "JWT", "OAuth2"],
  "topic": "API authentication implementation",
  "timestamp": "2025-01-15T14:32:00"
}
```

### Stage 2: Synthesize

Similar memories are merged and duplicates removed using hybrid similarity (80% semantic vectors + 20% entity overlap).

**Before (3 memories):**
```
1. "Deni is building an authentication API" [persons: Deni]
2. "Deni is developing an API for user auth" [persons: Deni]  ← similar to #1
3. "The team chose PostgreSQL for the database" [entities: PostgreSQL]
```

**After (2 memories):**
```
1. "Deni is building an authentication API" [persons: Deni]  ← kept
2. "The team chose PostgreSQL for the database" [entities: PostgreSQL]
```

### Stage 3: Retrieve

Hybrid search combines semantic similarity, keyword matching (FTS5), and structured filters.

**Query:** `"authentication approach"`

**Results:**
```
1. "Deni is building an authentication API" [score: 0.89]
   persons: [Deni], entities: [API, authentication]

2. "The team chose PostgreSQL for the database" [score: 0.12]
   entities: [PostgreSQL]
```

## Features

### Hybrid Retrieval

- **Semantic**: Vector similarity search with configurable top-k
- **Keyword**: Full-text search with FTS5 (Porter tokenizer)
- **Structured**: Filter by persons, entities, location, time expressions
- **Query Analysis**: LLM extracts search parameters from natural language queries

### Storage

- **SQLite** — Persistent storage with sqlite-vec for embeddings
- **InMemory** — Fast in-memory storage for testing

### Providers

- **Ollama** — Local LLM and embeddings
- **OpenAI** — Cloud LLM and embeddings

## License

[Apache-2.0](LICENSE)
