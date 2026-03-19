use crate::error::{EmbeddingResult, LlmResult, VectorResult};
use crate::traits::{Embedder, LlmClient, Message, VectorStore};
use crate::types::{MemoryEntry, StructuredSearchParams};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.1; self.dimension]).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub struct MockLlmClient {
    responses: Arc<Mutex<Vec<serde_json::Value>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_responses(responses: Vec<serde_json::Value>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            call_count: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn chat_completion(&self, _messages: &[Message], _temperature: f32) -> LlmResult<String> {
        Ok("mock response".to_string())
    }

    async fn chat_completion_with_json(
        &self,
        _messages: &[Message],
        _temperature: f32,
    ) -> LlmResult<serde_json::Value> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;
        let idx = (*count - 1) as usize;

        let responses = self.responses.lock().unwrap();
        if idx < responses.len() {
            Ok(responses[idx].clone())
        } else {
            Err(crate::error::LlmError::ApiError(
                "No mock response configured".to_string(),
            ))
        }
    }
}

pub struct MockVectorStore {
    entries: Arc<Mutex<Vec<(MemoryEntry, Vec<f32>)>>>,
}

impl MockVectorStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_entries(entries: Vec<(MemoryEntry, Vec<f32>)>) -> Self {
        Self {
            entries: Arc::new(Mutex::new(entries)),
        }
    }
}

#[async_trait]
impl VectorStore for MockVectorStore {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> VectorResult<()> {
        let mut store = self.entries.lock().unwrap();
        for (entry, vector) in entries.iter().zip(vectors.iter()) {
            store.push((entry.clone(), vector.clone()));
        }
        Ok(())
    }

    async fn semantic_search(
        &self,
        _query_vector: &[f32],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let store = self.entries.lock().unwrap();
        Ok(store.iter().take(top_k).map(|(e, _)| e.clone()).collect())
    }

    async fn keyword_search(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let store = self.entries.lock().unwrap();
        let results: Vec<MemoryEntry> = store
            .iter()
            .filter(|(entry, _)| {
                keywords.iter().any(|k| {
                    entry
                        .lossless_restatement
                        .to_lowercase()
                        .contains(&k.to_lowercase())
                })
            })
            .take(top_k)
            .map(|(e, _)| e.clone())
            .collect();
        Ok(results)
    }

    async fn structured_search(
        &self,
        params: &StructuredSearchParams,
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let store = self.entries.lock().unwrap();
        let results: Vec<MemoryEntry> = store
            .iter()
            .filter(|(entry, _)| {
                let mut matches = true;
                if let Some(ref persons) = params.persons {
                    matches &= persons.iter().any(|p| entry.persons.contains(p));
                }
                if let Some(ref loc) = params.location {
                    matches &= entry.location.as_ref() == Some(loc);
                }
                if let Some(ref entities) = params.entities {
                    matches &= entities.iter().any(|e| entry.entities.contains(e));
                }
                matches
            })
            .take(top_k)
            .map(|(e, _)| e.clone())
            .collect();
        Ok(results)
    }

    async fn delete_entry(&self, entry_id: &uuid::Uuid) -> VectorResult<bool> {
        let mut store = self.entries.lock().unwrap();
        let len_before = store.len();
        store.retain(|(e, _)| e.entry_id != *entry_id);
        Ok(store.len() < len_before)
    }

    async fn count(&self) -> VectorResult<usize> {
        Ok(self.entries.lock().unwrap().len())
    }

    async fn get_all_entries(&self) -> VectorResult<Vec<MemoryEntry>> {
        Ok(self
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|(e, _)| e.clone())
            .collect())
    }
}
