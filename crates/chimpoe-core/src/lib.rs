pub mod config;
pub mod error;
pub mod traits;
pub mod types;

pub mod embed;
pub mod llm;
pub mod pipeline;
pub mod store;
pub mod vector;

mod chimpoe;

pub use config::Config;
pub use error::{ChimpoeError, Result};
pub use traits::*;
pub use types::*;

pub use chimpoe::{Chimpoe, MemoryHit, SearchResult};
pub use embed::OllamaEmbedder;
pub use llm::OllamaLlm;
pub use pipeline::{Compressor, Synthesizer};
pub use store::SqliteStore;
pub use vector::InMemoryVector;
