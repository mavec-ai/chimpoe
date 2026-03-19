use crate::config::Config;
use crate::embed::OllamaEmbedder;
use crate::error::Result;
use crate::llm::OllamaLlm;
use crate::pipeline::{Compressor, HybridRetriever, Synthesizer};
use crate::traits::{Embedder, LlmClient, Message, MessageRole, VectorStore};
use crate::types::{Dialogue, MemoryEntry};
use crate::vector::InMemoryVector;
use std::sync::Arc;

pub struct Chimpoe {
    config: Config,
    dialogues: Vec<Dialogue>,
    compressor: Option<Compressor>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    llm: Option<Arc<dyn LlmClient>>,
    retriever: Option<HybridRetriever>,
}

#[derive(Default)]
pub struct ChimpoeBuilder {
    config: Config,
    embedder: Option<Arc<dyn Embedder>>,
    vector_store: Option<Arc<dyn VectorStore>>,
    llm: Option<Arc<dyn LlmClient>>,
}

impl ChimpoeBuilder {
    pub fn vector_store(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.vector_store = Some(store);
        self
    }

    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub async fn build(self) -> Result<Chimpoe> {
        let embedder = self
            .embedder
            .unwrap_or_else(|| Arc::new(OllamaEmbedder::new(&self.config.embedding)));

        let vector_store = self
            .vector_store
            .unwrap_or_else(|| Arc::new(InMemoryVector::new()));

        let llm = self
            .llm
            .or_else(|| Some(Arc::new(OllamaLlm::new(&self.config.llm)) as Arc<dyn LlmClient>));

        let compressor = llm.as_ref().map(|l| {
            Compressor::new(
                l.clone(),
                self.config.pipeline.clone(),
                self.config.llm.temperature,
            )
        });

        let retriever = llm.as_ref().map(|l| {
            HybridRetriever::new(
                l.clone(),
                vector_store.clone(),
                embedder.clone(),
                &self.config.pipeline.retrieval,
            )
        });

        Ok(Chimpoe {
            config: self.config,
            dialogues: Vec::new(),
            compressor,
            embedder,
            vector_store,
            llm,
            retriever,
        })
    }
}

impl Chimpoe {
    pub fn builder() -> ChimpoeBuilder {
        ChimpoeBuilder::default()
    }

    pub async fn new() -> Result<Self> {
        Self::builder().build().await
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

            self.vector_store
                .add_entries(&new_memories, &vectors)
                .await?;
        }

        self.dialogues.clear();

        Ok(count)
    }

    pub async fn search(&self, query: &str, top_k: Option<usize>) -> Result<SearchResult> {
        let retriever = match &self.retriever {
            Some(r) => r,
            None => {
                return Ok(SearchResult {
                    query: query.to_string(),
                    results: Vec::new(),
                });
            }
        };

        let hits = retriever.retrieve(query, top_k).await?;

        Ok(SearchResult {
            query: query.to_string(),
            results: hits
                .into_iter()
                .map(|hit| MemoryHit {
                    memory: hit.memory,
                    persons: hit.persons,
                    entities: hit.entities,
                    location: hit.location,
                    topic: hit.topic,
                    timestamp: hit.timestamp,
                    source: hit.source,
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

    pub async fn memory_count(&self) -> usize {
        self.vector_store.count().await.unwrap_or(0)
    }

    pub fn dialogue_count(&self) -> usize {
        self.dialogues.len()
    }

    pub async fn list_memories(&self) -> Vec<MemoryEntry> {
        self.vector_store
            .get_all_entries()
            .await
            .unwrap_or_default()
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
