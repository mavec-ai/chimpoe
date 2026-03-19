# Chimpoe

Long-term memory for AI agents.

## Features

### 3-Stage Memory Pipeline

- **Compress** — Extract structured memory entries from dialogues with forced disambiguation (no pronouns, absolute timestamps)
- **Synthesize** — Deduplicate and merge similar entries using Jaccard similarity
- **Retrieve** — Hybrid search across semantic, keyword, and structured indexes

### Hybrid Retrieval

- **Semantic**: Vector similarity search with configurable top-k
- **Keyword**: Full-text search with FTS5 (Porter stemmer)
- **Structured**: Filter by persons, entities, location, time expressions
- **Query Analysis**: LLM extracts search parameters from natural language queries

## License

[Apache-2.0](LICENSE)
