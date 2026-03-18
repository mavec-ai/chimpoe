use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChimpoeError {
    #[error("Store error: {0}")]
    Store(#[from] StoreError),

    #[error("Vector store error: {0}")]
    Vector(#[from] VectorError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("Database connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Entry not found: {0}")]
    NotFound(String),

    #[error("Migration failed: {0}")]
    MigrationFailed(String),

    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
}

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("Index creation failed: {0}")]
    IndexCreationFailed(String),

    #[error("Search failed: {0}")]
    SearchFailed(String),

    #[error("Insertion failed: {0}")]
    InsertionFailed(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Table not found: {0}")]
    TableNotFound(String),
}

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Encoding failed: {0}")]
    EncodingFailed(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Timeout")]
    Timeout,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Timeout")]
    Timeout,

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),

    #[error("JSON extraction failed: {0}")]
    JsonExtractionFailed(String),
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Compression failed: {0}")]
    CompressionFailed(String),

    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),

    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Window processing failed: {0}")]
    WindowProcessingFailed(String),

    #[error("LLM error: {0}")]
    LlmError(#[from] LlmError),

    #[error("Store error: {0}")]
    StoreError(#[from] StoreError),

    #[error("Vector error: {0}")]
    VectorError(#[from] VectorError),

    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),
}

pub type Result<T> = std::result::Result<T, ChimpoeError>;
pub type StoreResult<T> = std::result::Result<T, StoreError>;
pub type VectorResult<T> = std::result::Result<T, VectorError>;
pub type EmbeddingResult<T> = std::result::Result<T, EmbeddingError>;
pub type LlmResult<T> = std::result::Result<T, LlmError>;
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;
