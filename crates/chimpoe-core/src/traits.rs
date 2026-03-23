use crate::error::{EmbeddingResult, LlmResult, VectorResult};
use crate::types::{MemoryEntry, StructuredSearchParams};
use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>>;
    async fn encode_single(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        let results = self.encode(&[text]).await?;
        Ok(results.into_iter().next().unwrap_or_default())
    }
    fn dimension(&self) -> usize;
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_completion(&self, messages: &[Message], temperature: f32) -> LlmResult<String>;

    async fn chat_completion_with_json(
        &self,
        messages: &[Message],
        temperature: f32,
    ) -> LlmResult<serde_json::Value>;
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> VectorResult<()>;
    async fn semantic_search(
        &self,
        query_vector: &[f32],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>>;
    async fn keyword_search(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>>;
    async fn structured_search(
        &self,
        params: &StructuredSearchParams,
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>>;
    async fn delete_entry(&self, entry_id: &uuid::Uuid) -> VectorResult<bool>;
    async fn count(&self) -> VectorResult<usize>;
    async fn get_all_entries(&self) -> VectorResult<Vec<MemoryEntry>>;
    async fn get_all_entries_with_vectors(&self) -> VectorResult<Vec<(MemoryEntry, Vec<f32>)>>;
}
