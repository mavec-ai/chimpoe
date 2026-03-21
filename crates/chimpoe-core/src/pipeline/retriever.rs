use crate::config::RetrievalConfig;
use crate::error::Result;
use crate::traits::{Embedder, LlmClient, Message, MessageRole, VectorStore};
use crate::types::{MemoryEntry, QueryAnalysis, StructuredSearchParams};
use std::sync::Arc;

const DEFAULT_TEMPERATURE: f32 = 0.1;
const QUERY_ANALYSIS_PROMPT: &str = r#"Analyze the following query and extract key information:

Query: {query}

Please extract:
1. keywords: List of keywords (names, places, topic words, etc.)
2. persons: Person names mentioned
3. entities: Entities (companies, products, organizations, etc.)
4. location: Location mentioned (if any)
5. time_expression: Time expression (if any, e.g., "last week", "yesterday", "January 15", "2 days ago")

Return in JSON format:
{{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1", "name2"],
  "entities": ["entity1"],
  "location": "location or null",
  "time_expression": "time expression or null"
}}

Return ONLY JSON, no other content."#;

pub struct HybridRetriever {
    llm: Arc<dyn LlmClient>,
    vector_store: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    semantic_top_k: usize,
    keyword_top_k: usize,
    structured_top_k: usize,
}

impl HybridRetriever {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        vector_store: Arc<dyn VectorStore>,
        embedder: Arc<dyn Embedder>,
        config: &RetrievalConfig,
    ) -> Self {
        Self {
            llm,
            vector_store,
            embedder,
            semantic_top_k: config.semantic_top_k,
            keyword_top_k: config.keyword_top_k,
            structured_top_k: config.structured_top_k,
        }
    }

    pub async fn analyze_query(&self, query: &str) -> Option<QueryAnalysis> {
        let prompt = QUERY_ANALYSIS_PROMPT.replace("{query}", query);

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a query analysis assistant. You must output valid JSON format."
                    .to_string(),
            },
            Message {
                role: MessageRole::User,
                content: prompt,
            },
        ];

        match self.llm.chat_completion_with_json(&messages, DEFAULT_TEMPERATURE).await {
            Ok(json) => {
                if let Ok(analysis) = serde_json::from_value::<QueryAnalysis>(json) {
                    tracing::debug!("Query analysis: {:?}", analysis);
                    return Some(analysis);
                }
            }
            Err(e) => {
                tracing::warn!("Query analysis failed: {}", e);
            }
        }

        None
    }

    pub async fn retrieve(&self, query: &str, top_k: Option<usize>) -> Result<Vec<RetrievalHit>> {
        let k = top_k.unwrap_or(self.semantic_top_k);

        let query_analysis = self.analyze_query(query).await;

        let keywords = query_analysis
            .as_ref()
            .map(|a| a.keywords.clone())
            .unwrap_or_else(|| {
                query
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(String::from)
                    .collect()
            });

        let structured_params = query_analysis.as_ref().and_then(|analysis| {
            let has_structured_conditions = !analysis.persons.is_empty()
                || !analysis.entities.is_empty()
                || analysis.location.is_some()
                || analysis.time_expression.is_some();

            if has_structured_conditions {
                Some(StructuredSearchParams::from(analysis.clone()))
            } else {
                None
            }
        });

        let semantic_fut = self.semantic_search(query, k);
        let lexical_fut = async {
            if keywords.is_empty() {
                Ok(Vec::new())
            } else {
                self.keyword_search(&keywords, self.keyword_top_k).await
            }
        };
        let structured_fut = async {
            if let Some(ref params) = structured_params {
                self.structured_search(params, self.structured_top_k).await
            } else {
                Ok(Vec::new())
            }
        };

        let (semantic_results, lexical_results, structured_results) =
            tokio::try_join!(semantic_fut, lexical_fut, structured_fut)?;

        let merged =
            self.merge_and_deduplicate(structured_results, semantic_results, lexical_results, k);

        Ok(merged)
    }

    async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryEntry>> {
        let query_vector = self.embedder.encode_single(query).await?;
        if query_vector.is_empty() {
            return Ok(Vec::new());
        }

        let results = self
            .vector_store
            .semantic_search(&query_vector, top_k)
            .await?;

        Ok(results)
    }

    async fn keyword_search(&self, keywords: &[String], top_k: usize) -> Result<Vec<MemoryEntry>> {
        let results = self.vector_store.keyword_search(keywords, top_k).await?;
        Ok(results)
    }

    async fn structured_search(
        &self,
        params: &StructuredSearchParams,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let results = self.vector_store.structured_search(params, top_k).await?;
        Ok(results)
    }

    fn merge_and_deduplicate(
        &self,
        structured: Vec<MemoryEntry>,
        semantic: Vec<MemoryEntry>,
        lexical: Vec<MemoryEntry>,
        top_k: usize,
    ) -> Vec<RetrievalHit> {
        let mut seen_ids = std::collections::HashSet::new();
        let mut merged = Vec::new();

        for entry in structured {
            if seen_ids.insert(entry.entry_id) {
                merged.push(RetrievalHit::from_entry(entry, "structured"));
            }
        }

        for entry in semantic {
            if seen_ids.insert(entry.entry_id) {
                merged.push(RetrievalHit::from_entry(entry, "semantic"));
            }
        }

        for entry in lexical {
            if seen_ids.insert(entry.entry_id) {
                merged.push(RetrievalHit::from_entry(entry, "lexical"));
            }
        }

        merged.truncate(top_k);
        merged
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub memory: String,
    pub persons: Vec<String>,
    pub entities: Vec<String>,
    pub location: Option<String>,
    pub topic: Option<String>,
    pub timestamp: Option<String>,
    pub source: String,
}

impl RetrievalHit {
    pub fn from_entry(entry: MemoryEntry, source: &str) -> Self {
        Self {
            memory: entry.lossless_restatement,
            persons: entry.persons,
            entities: entry.entities,
            location: entry.location,
            topic: entry.topic,
            timestamp: entry.timestamp.map(|t| t.to_rfc3339()),
            source: source.to_string(),
        }
    }
}

impl std::fmt::Display for RetrievalHit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}]", self.memory, self.source)?;
        if !self.persons.is_empty() {
            write!(f, " | Persons: {:?}", self.persons)?;
        }
        if !self.entities.is_empty() {
            write!(f, " | Entities: {:?}", self.entities)?;
        }
        if let Some(ref loc) = self.location {
            write!(f, " | Location: {}", loc)?;
        }
        Ok(())
    }
}

#[cfg(test)]
use crate::types::TimeRange;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::{MockEmbedder, MockLlmClient, MockVectorStore};

    fn make_entry(text: &str) -> MemoryEntry {
        MemoryEntry::new(text.to_string())
    }

    fn make_entry_with_persons(text: &str, persons: Vec<&str>) -> MemoryEntry {
        let mut entry = make_entry(text);
        entry.persons = persons.into_iter().map(String::from).collect();
        entry
    }

    fn make_entry_with_location(text: &str, location: &str) -> MemoryEntry {
        let mut entry = make_entry(text);
        entry.location = Some(location.to_string());
        entry
    }

    fn create_retriever(
        llm: MockLlmClient,
        vector_store: MockVectorStore,
        embedder: MockEmbedder,
    ) -> HybridRetriever {
        let config = RetrievalConfig {
            semantic_top_k: 5,
            keyword_top_k: 3,
            structured_top_k: 5,
        };
        HybridRetriever::new(
            Arc::new(llm),
            Arc::new(vector_store),
            Arc::new(embedder),
            &config,
        )
    }

    #[tokio::test]
    async fn test_analyze_query_valid_response() {
        let response = serde_json::json!({
            "keywords": ["pizza", "food"],
            "persons": ["Alice"],
            "entities": ["Restaurant"],
            "location": "Jakarta",
            "time_expression": "last week"
        });

        let llm = MockLlmClient::with_responses(vec![response]);
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let analysis = retriever
            .analyze_query("What did Alice eat last week?")
            .await;

        assert!(analysis.is_some());
        let a = analysis.unwrap();
        assert!(a.keywords.contains(&"pizza".to_string()));
        assert!(a.persons.contains(&"Alice".to_string()));
        assert!(a.entities.contains(&"Restaurant".to_string()));
        assert_eq!(a.location, Some("Jakarta".to_string()));
        assert_eq!(a.time_expression, Some("last week".to_string()));
    }

    #[tokio::test]
    async fn test_analyze_query_returns_none_on_error() {
        let llm = MockLlmClient::new();
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let analysis = retriever.analyze_query("test query").await;
        assert!(analysis.is_none());
    }

    #[tokio::test]
    async fn test_retrieve_empty_store() {
        let llm = MockLlmClient::new();
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever.retrieve("test query", Some(5)).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_semantic_results() {
        let entry = make_entry("User likes pizza");
        let vector_store = MockVectorStore::with_entries(vec![(entry, vec![0.1; 128])]);
        let llm = MockLlmClient::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever
            .retrieve("pizza preferences", Some(5))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory, "User likes pizza");
        assert_eq!(results[0].source, "semantic");
    }

    #[tokio::test]
    async fn test_retrieve_keyword_results() {
        let entry = make_entry("User likes pizza");
        let vector_store = MockVectorStore::with_entries(vec![(entry, vec![0.1; 128])]);
        let llm = MockLlmClient::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever.retrieve("pizza", Some(5)).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_structured_results_with_persons() {
        let entry = make_entry_with_persons("Met Alice at conference", vec!["Alice"]);
        let vector_store = MockVectorStore::with_entries(vec![(entry, vec![0.1; 128])]);

        let response = serde_json::json!({
            "keywords": ["conference"],
            "persons": ["Alice"],
            "entities": [],
            "location": null,
            "time_expression": null
        });
        let llm = MockLlmClient::with_responses(vec![response]);
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever
            .retrieve("Did I meet Alice?", Some(5))
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieve_structured_results_with_location() {
        let entry = make_entry_with_location("Visited Jakarta", "Jakarta");
        let vector_store = MockVectorStore::with_entries(vec![(entry, vec![0.1; 128])]);

        let response = serde_json::json!({
            "keywords": ["visited"],
            "persons": [],
            "entities": [],
            "location": "Jakarta",
            "time_expression": null
        });
        let llm = MockLlmClient::with_responses(vec![response]);
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever
            .retrieve("Where did I go in Jakarta?", Some(5))
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_merge_and_deduplicate_removes_duplicates() {
        let llm = MockLlmClient::new();
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let entry = make_entry("Test memory");
        let structured = vec![entry.clone()];
        let semantic = vec![entry.clone()];
        let lexical = vec![entry];

        let merged = retriever.merge_and_deduplicate(structured, semantic, lexical, 10);

        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_and_deduplicate_respects_top_k() {
        let llm = MockLlmClient::new();
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| make_entry(&format!("Memory {}", i)))
            .collect();

        let merged = retriever.merge_and_deduplicate(entries.clone(), vec![], vec![], 3);

        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_and_deduplicate_combines_sources() {
        let llm = MockLlmClient::new();
        let vector_store = MockVectorStore::new();
        let embedder = MockEmbedder::new(128);
        let retriever = create_retriever(llm, vector_store, embedder);

        let structured = vec![make_entry("Structured memory")];
        let semantic = vec![make_entry("Semantic memory")];
        let lexical = vec![make_entry("Lexical memory")];

        let merged = retriever.merge_and_deduplicate(structured, semantic, lexical, 10);

        assert_eq!(merged.len(), 3);
        let sources: Vec<&str> = merged.iter().map(|h| h.source.as_str()).collect();
        assert!(sources.contains(&"structured"));
        assert!(sources.contains(&"semantic"));
        assert!(sources.contains(&"lexical"));
    }

    #[test]
    fn test_retrieval_hit_from_entry() {
        let mut entry = MemoryEntry::new("Test memory".to_string());
        entry.persons = vec!["Alice".to_string()];
        entry.entities = vec!["Google".to_string()];
        entry.location = Some("Jakarta".to_string());
        entry.topic = Some("work".to_string());

        let hit = RetrievalHit::from_entry(entry, "semantic");

        assert_eq!(hit.memory, "Test memory");
        assert_eq!(hit.persons, vec!["Alice"]);
        assert_eq!(hit.entities, vec!["Google"]);
        assert_eq!(hit.location, Some("Jakarta".to_string()));
        assert_eq!(hit.topic, Some("work".to_string()));
        assert_eq!(hit.source, "semantic");
    }

    #[test]
    fn test_retrieval_hit_display() {
        let hit = RetrievalHit {
            memory: "Test memory".to_string(),
            persons: vec!["Alice".to_string()],
            entities: vec!["Google".to_string()],
            location: Some("Jakarta".to_string()),
            topic: None,
            timestamp: None,
            source: "semantic".to_string(),
        };

        let display = format!("{}", hit);
        assert!(display.contains("Test memory"));
        assert!(display.contains("semantic"));
        assert!(display.contains("Alice"));
        assert!(display.contains("Google"));
        assert!(display.contains("Jakarta"));
    }

    #[test]
    fn test_retrieval_hit_display_minimal() {
        let hit = RetrievalHit {
            memory: "Simple memory".to_string(),
            persons: vec![],
            entities: vec![],
            location: None,
            topic: None,
            timestamp: None,
            source: "lexical".to_string(),
        };

        let display = format!("{}", hit);
        assert_eq!(display, "Simple memory [lexical]");
    }

    #[test]
    fn test_structured_search_params_from_persons() {
        let analysis = QueryAnalysis {
            keywords: vec!["test".to_string()],
            persons: vec!["Alice".to_string()],
            entities: vec![],
            location: None,
            time_expression: None,
        };

        let params = StructuredSearchParams::from(analysis);
        assert_eq!(params.persons, Some(vec!["Alice".to_string()]));
        assert!(params.location.is_none());
        assert!(params.entities.is_none());
    }

    #[test]
    fn test_structured_search_params_from_location() {
        let analysis = QueryAnalysis {
            keywords: vec!["test".to_string()],
            persons: vec![],
            entities: vec![],
            location: Some("Jakarta".to_string()),
            time_expression: None,
        };

        let params = StructuredSearchParams::from(analysis);
        assert_eq!(params.location, Some("Jakarta".to_string()));
    }

    #[test]
    fn test_structured_search_params_from_entities() {
        let analysis = QueryAnalysis {
            keywords: vec!["test".to_string()],
            persons: vec![],
            entities: vec!["Google".to_string()],
            location: None,
            time_expression: None,
        };

        let params = StructuredSearchParams::from(analysis);
        assert_eq!(params.entities, Some(vec!["Google".to_string()]));
    }

    #[test]
    fn test_time_range_parse_relative_date() {
        let result = TimeRange::parse("yesterday");
        assert!(result.is_some());
        let range = result.unwrap();
        assert!(range.start < range.end);
    }

    #[test]
    fn test_time_range_parse_invalid() {
        let result = TimeRange::parse("not a date");
        assert!(result.is_none());
    }
}
