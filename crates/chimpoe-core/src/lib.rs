pub mod config;
pub mod error;
pub mod traits;
pub mod types;

pub mod embed;
pub mod llm;
pub mod pipeline;
pub mod vector;

mod chimpoe;

#[cfg(test)]
mod mocks;

pub use config::{
    Config, EmbeddingConfig, LlmConfig, PipelineConfig, Provider, RetrievalConfig,
    SynthesizerConfig,
};
pub use error::{ChimpoeError, Result};
pub use traits::{Embedder, LlmClient, Message, MessageRole, VectorStore};
pub use types::{Dialogue, MemoryEntry};

pub use chimpoe::{Chimpoe, ChimpoeBuilder, MemoryHit, SearchResult};
pub use embed::{OllamaEmbedder, OpenAIEmbedder};
pub use llm::{OllamaLlm, OpenAILlm};
pub use pipeline::{AnswerGenerator, Compressor, HybridRetriever, RetrievalHit, Synthesizer};
pub use vector::{InMemoryVector, SqliteVector};
