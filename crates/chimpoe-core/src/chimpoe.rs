use crate::config::Config;
use crate::embed::OllamaEmbedder;
use crate::error::Result;
use crate::llm::OllamaLlm;
use crate::pipeline::{Compressor, Synthesizer};
use crate::store::SqliteStore;
use crate::traits::{Embedder, LlmClient, Message, MessageRole, Store, VectorStore};
use crate::types::{Dialogue, MemoryEntry};
use crate::vector::InMemoryVector;
use std::sync::Arc;

pub struct Chimpoe {
    config: Config,
    dialogues: Vec<Dialogue>,
    memories: Vec<MemoryEntry>,
    compressor: Option<Compressor>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<InMemoryVector>,
    llm: Option<Arc<dyn LlmClient>>,
    store: Arc<SqliteStore>,
}

impl Chimpoe {
    pub async fn new() -> Result<Self> {
        Self::with_config(Config::default()).await
    }

    pub async fn with_config(config: Config) -> Result<Self> {
        let store = Arc::new(SqliteStore::new(":memory:").await?);
        let embedder = Arc::new(OllamaEmbedder::new(&config.embedding));
        let llm = Arc::new(OllamaLlm::new(&config.llm));
        let vector_store = Arc::new(InMemoryVector::new());

        let compressor =
            Compressor::new(llm.clone(), config.pipeline.clone(), config.llm.temperature);

        Ok(Self {
            config,
            dialogues: Vec::new(),
            memories: Vec::new(),
            compressor: Some(compressor),
            embedder,
            vector_store,
            llm: Some(llm),
            store,
        })
    }

    pub async fn add_dialogue(
        &mut self,
        speaker: &str,
        content: &str,
        timestamp: Option<&str>,
    ) -> Result<()> {
        let mut dialogue = Dialogue::new(speaker, content);
        if let Some(ts) = timestamp {
            dialogue = dialogue.with_timestamp(ts);
        }
        self.dialogues.push(dialogue);
        if self.dialogues.len() >= self.config.pipeline.window_size {
            self.process_window().await?;
        }
        Ok(())
    }

    pub async fn add_dialogues(&mut self, dialogues: Vec<Dialogue>) -> Result<()> {
        for dialogue in dialogues {
            self.dialogues.push(dialogue);
            if self.dialogues.len() >= self.config.pipeline.window_size {
                self.process_window().await?;
            }
        }
        Ok(())
    }

    pub async fn finalize(&mut self) -> Result<usize> {
        self.process_window().await
    }

    async fn process_window(&mut self) -> Result<usize> {
        if self.dialogues.is_empty() {
            return Ok(0);
        }

        let compressor = match &self.compressor {
            Some(c) => c,
            None => return Ok(0),
        };

        let mut new_memories = compressor.compress_dialogues(&self.dialogues).await?;

        if !new_memories.is_empty() {
            let synthesizer = Synthesizer::new(self.config.pipeline.clone());
            new_memories = synthesizer.synthesize(new_memories)?;
        }

        let count = new_memories.len();

        if !new_memories.is_empty() {
            let texts: Vec<&str> = new_memories
                .iter()
                .map(|m| m.lossless_restatement.as_str())
                .collect();
            let vectors = self.embedder.encode(&texts).await?;

            for entry in &new_memories {
                self.store.save_entry(entry).await?;
            }

            self.vector_store
                .add_entries(&new_memories, &vectors)
                .await?;
            self.memories.extend(new_memories);
        }

        self.dialogues.clear();

        Ok(count)
    }

    pub async fn search(&self, query: &str, top_k: Option<usize>) -> Result<SearchResult> {
        let k = top_k.unwrap_or(5);

        let query_vector = self.embedder.encode_single(query).await?;
        let semantic_results = self.vector_store.semantic_search(&query_vector, k).await?;

        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(String::from)
            .collect();

        let lexical_results = if !keywords.is_empty() {
            self.vector_store.keyword_search(&keywords, k).await?
        } else {
            Vec::new()
        };

        let mut seen_ids = std::collections::HashSet::new();
        let mut combined = Vec::new();

        for entry in semantic_results {
            if seen_ids.insert(entry.entry_id) {
                combined.push((entry, "semantic".to_string()));
            }
        }

        for entry in lexical_results {
            if seen_ids.insert(entry.entry_id) {
                combined.push((entry, "lexical".to_string()));
            }
        }

        combined.truncate(k);

        Ok(SearchResult {
            query: query.to_string(),
            results: combined
                .into_iter()
                .map(|(entry, source)| MemoryHit {
                    memory: entry.lossless_restatement,
                    persons: entry.persons,
                    entities: entry.entities,
                    location: entry.location,
                    topic: entry.topic,
                    timestamp: entry.timestamp.map(|t| t.to_rfc3339()),
                    source,
                })
                .collect(),
        })
    }

    pub async fn ask(&self, question: &str) -> Result<String> {
        let llm = match &self.llm {
            Some(l) => l,
            None => {
                return Err(crate::error::ChimpoeError::Llm(
                    crate::error::LlmError::ApiError("No LLM configured".to_string()),
                ));
            }
        };

        let search_result = self.search(question, Some(5)).await?;

        let context = if search_result.results.is_empty() {
            "No relevant memories found.".to_string()
        } else {
            search_result
                .results
                .iter()
                .map(|m| format!("- {}", m.memory))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let system_prompt = format!(
            "You are a helpful assistant. Answer the question based on the following memories:\n\n{}",
            context
        );

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: system_prompt,
            },
            Message {
                role: MessageRole::User,
                content: question.to_string(),
            },
        ];

        let response = llm
            .chat_completion(&messages, self.config.llm.temperature)
            .await?;
        Ok(response)
    }

    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    pub fn dialogue_count(&self) -> usize {
        self.dialogues.len()
    }

    pub fn list_memories(&self) -> &[MemoryEntry] {
        &self.memories
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub query: String,
    pub results: Vec<MemoryHit>,
}

#[derive(Debug, Clone)]
pub struct MemoryHit {
    pub memory: String,
    pub persons: Vec<String>,
    pub entities: Vec<String>,
    pub location: Option<String>,
    pub topic: Option<String>,
    pub timestamp: Option<String>,
    pub source: String,
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Search: {}", self.query)?;
        writeln!(f, "Found {} results:", self.results.len())?;
        for (i, hit) in self.results.iter().enumerate() {
            writeln!(f, "\n{}. {} [{}]", i + 1, hit.memory, hit.source)?;
            if !hit.persons.is_empty() {
                writeln!(f, "   Persons: {:?}", hit.persons)?;
            }
            if !hit.entities.is_empty() {
                writeln!(f, "   Entities: {:?}", hit.entities)?;
            }
            if let Some(ref loc) = hit.location {
                writeln!(f, "   Location: {}", loc)?;
            }
        }
        Ok(())
    }
}
