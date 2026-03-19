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

pub use config::{Config, RetrievalConfig};
pub use error::{ChimpoeError, Result};
pub use traits::*;
pub use types::*;

pub use chimpoe::{Chimpoe, ChimpoeBuilder, MemoryHit, SearchResult};
pub use embed::OllamaEmbedder;
pub use llm::OllamaLlm;
pub use pipeline::{Compressor, HybridRetriever, RetrievalHit, Synthesizer};
pub use vector::{InMemoryVector, SqliteVector};
