use crate::config::RetrievalConfig;
use crate::error::Result;
use crate::traits::{Embedder, LlmClient, Message, MessageRole, VectorStore};
use crate::types::{MemoryEntry, QueryAnalysis, StructuredSearchParams};
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_TEMPERATURE: f32 = 0.1;
const RRF_K: f32 = 60.0;
const QUERY_ANALYSIS_PROMPT: &str = r#"Analyze the following query and extract key information.

Current date: {current_date}

Query: {query}

Please extract:
1. keywords: List of keywords (names, places, topic words, etc.)
2. persons: Person names mentioned
3. entities: Entities (companies, products, organizations, etc.)
4. location: Location mentioned (if any)
5. time_expression: Time expression in original language (preserve as-is)
6. resolved_date: Convert the time expression to ISO date format (YYYY-MM-DD) based on Current Date.
   Handle ANY language - use the same logic for all:
   - yesterday / kemarin / hier / 昨日 / أيام → Current Date - 1 day
   - tomorrow / besok / demain / 明日 → Current Date + 1 day
   - last week / minggu lalu / 先週 → Current Date - 7 days
   - "January 15" or similar absolute dates → use as-is (current year if not specified)
   - If no time reference, use null

Return in JSON format:
{{
  "keywords": ["keyword1", "keyword2"],
  "persons": ["name1", "name2"],
  "entities": ["entity1"],
  "location": "location or null",
  "time_expression": "original time expression or null",
  "resolved_date": "YYYY-MM-DD or null"
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
        let now = chrono::Local::now();
        let current_date = now.format("%Y-%m-%d").to_string();
        let prompt = QUERY_ANALYSIS_PROMPT
            .replace("{current_date}", &current_date)
            .replace("{query}", query);

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

        match self
            .llm
            .chat_completion_with_json(&messages, DEFAULT_TEMPERATURE)
            .await
        {
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

        let keywords = query_analysis.as_ref().map_or_else(
            || {
                query
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(String::from)
                    .collect()
            },
            |a| a.keywords.clone(),
        );

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
            Self::merge_and_deduplicate(structured_results, semantic_results, lexical_results, k);

        let expanded = self.expand_clusters(merged).await?;

        Ok(expanded)
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
        structured: Vec<MemoryEntry>,
        semantic: Vec<MemoryEntry>,
        lexical: Vec<MemoryEntry>,
        top_k: usize,
    ) -> Vec<RetrievalHit> {
        let channels: [(&str, Vec<MemoryEntry>); 3] = [
            ("structured", structured),
            ("lexical", lexical),
            ("semantic", semantic),
        ];

        Self::reciprocal_rank_fusion(&channels, top_k)
    }

    fn reciprocal_rank_fusion(
        channels: &[(&str, Vec<MemoryEntry>)],
        top_k: usize,
    ) -> Vec<RetrievalHit> {
        let mut rrf_scores: HashMap<uuid::Uuid, f64> = HashMap::new();
        let mut entries_by_id: HashMap<uuid::Uuid, MemoryEntry> = HashMap::new();
        let mut source_by_id: HashMap<uuid::Uuid, String> = HashMap::new();

        for (channel_name, entries) in channels {
            for (rank, entry) in entries.iter().enumerate() {
                let rank_1_based = rank + 1;
                let rrf_score = 1.0 / (f64::from(RRF_K) + rank_1_based as f64);

                *rrf_scores.entry(entry.entry_id).or_default() += rrf_score;
                entries_by_id
                    .entry(entry.entry_id)
                    .or_insert_with(|| entry.clone());
                source_by_id
                    .entry(entry.entry_id)
                    .or_insert_with(|| channel_name.to_string());
            }
        }

        let mut scored_entries: Vec<_> = rrf_scores.into_iter().collect();
        scored_entries.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored_entries.truncate(top_k);

        scored_entries
            .into_iter()
            .map(|(id, score)| {
                let entry = entries_by_id.get(&id).cloned().unwrap();
                let source = source_by_id
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| "rrf".to_string());
                RetrievalHit::from_entry_with_score(entry, &source, score as f32)
            })
            .collect()
    }

    async fn expand_clusters(&self, hits: Vec<RetrievalHit>) -> Result<Vec<RetrievalHit>> {
        let unique_cluster_ids: Vec<String> = hits
            .iter()
            .filter_map(|h| h.cluster_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if unique_cluster_ids.is_empty() {
            return Ok(hits);
        }

        let cluster_siblings = match self
            .vector_store
            .get_by_cluster_ids(&unique_cluster_ids)
            .await
        {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!("Failed to expand clusters: {}", e);
                return Ok(hits);
            }
        };

        let seen_ids: std::collections::HashSet<uuid::Uuid> =
            hits.iter().map(|h| h.entry_id).collect();

        let mut expanded = hits;
        for sibling in cluster_siblings {
            if seen_ids.contains(&sibling.entry_id) {
                continue;
            }
            expanded.push(RetrievalHit {
                entry_id: sibling.entry_id,
                memory: sibling.lossless_restatement,
                persons: sibling.persons,
                entities: sibling.entities,
                location: sibling.location,
                topic: sibling.topic,
                timestamp: sibling.timestamp.map(|t| t.to_rfc3339()),
                source: "cluster".to_string(),
                score: 0.0,
                cluster_id: sibling.cluster_id,
            });
        }

        Ok(expanded)
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub entry_id: uuid::Uuid,
    pub memory: String,
    pub persons: Vec<String>,
    pub entities: Vec<String>,
    pub location: Option<String>,
    pub topic: Option<String>,
    pub timestamp: Option<String>,
    pub source: String,
    pub score: f32,
    pub cluster_id: Option<String>,
}

impl RetrievalHit {
    #[must_use]
    pub fn from_entry(entry: MemoryEntry, source: &str) -> Self {
        Self {
            entry_id: entry.entry_id,
            memory: entry.lossless_restatement,
            persons: entry.persons,
            entities: entry.entities,
            location: entry.location,
            topic: entry.topic,
            timestamp: entry.timestamp.map(|t| t.to_rfc3339()),
            source: source.to_string(),
            score: 0.0,
            cluster_id: entry.cluster_id,
        }
    }

    #[must_use]
    pub fn from_entry_with_score(entry: MemoryEntry, source: &str, score: f32) -> Self {
        Self {
            entry_id: entry.entry_id,
            memory: entry.lossless_restatement,
            persons: entry.persons,
            entities: entry.entities,
            location: entry.location,
            topic: entry.topic,
            timestamp: entry.timestamp.map(|t| t.to_rfc3339()),
            source: source.to_string(),
            score,
            cluster_id: entry.cluster_id,
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
            write!(f, " | Location: {loc}")?;
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
        let valid_sources = ["semantic", "lexical", "rrf"];
        assert!(
            valid_sources.contains(&results[0].source.as_str()),
            "Expected valid source, got {}",
            results[0].source
        );
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
        let entry = make_entry("Test memory");
        let structured = vec![entry.clone()];
        let semantic = vec![entry.clone()];
        let lexical = vec![entry];

        let merged = HybridRetriever::merge_and_deduplicate(structured, semantic, lexical, 10);

        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_and_deduplicate_respects_top_k() {
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|i| make_entry(&format!("Memory {i}")))
            .collect();

        let merged = HybridRetriever::merge_and_deduplicate(entries, vec![], vec![], 3);

        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merge_and_deduplicate_combines_sources() {
        let structured = vec![make_entry("Structured memory")];
        let semantic = vec![make_entry("Semantic memory")];
        let lexical = vec![make_entry("Lexical memory")];

        let merged = HybridRetriever::merge_and_deduplicate(structured, semantic, lexical, 10);

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
            entry_id: uuid::Uuid::new_v4(),
            memory: "Test memory".to_string(),
            persons: vec!["Alice".to_string()],
            entities: vec!["Google".to_string()],
            location: Some("Jakarta".to_string()),
            topic: None,
            timestamp: None,
            source: "semantic".to_string(),
            score: 0.0167,
            cluster_id: None,
        };

        let display = format!("{hit}");
        assert!(display.contains("Test memory"));
        assert!(display.contains("semantic"));
        assert!(display.contains("Alice"));
        assert!(display.contains("Google"));
        assert!(display.contains("Jakarta"));
    }

    #[test]
    fn test_retrieval_hit_display_minimal() {
        let hit = RetrievalHit {
            entry_id: uuid::Uuid::new_v4(),
            memory: "Simple memory".to_string(),
            persons: vec![],
            entities: vec![],
            location: None,
            topic: None,
            timestamp: None,
            source: "lexical".to_string(),
            score: 0.0,
            cluster_id: None,
        };

        let display = format!("{hit}");
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
            resolved_date: None,
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
            resolved_date: None,
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
            resolved_date: None,
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

    #[test]
    fn test_rrf_single_channel_score_calculation() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();

        let e1 = MemoryEntry {
            entry_id: id1,
            lossless_restatement: "First".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };
        let e2 = MemoryEntry {
            entry_id: id2,
            lossless_restatement: "Second".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };
        let e3 = MemoryEntry {
            entry_id: id3,
            lossless_restatement: "Third".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };

        let channels: [(&str, Vec<MemoryEntry>); 1] = [("semantic", vec![e1, e2, e3])];
        let results = HybridRetriever::reciprocal_rank_fusion(&channels, 10);

        assert_eq!(results.len(), 3);
        let expected_rank1 = 1.0 / (60.0f64 + 1.0);
        let expected_rank2 = 1.0 / (60.0f64 + 2.0);
        let expected_rank3 = 1.0 / (60.0f64 + 3.0);

        assert!((results[0].score as f64 - expected_rank1).abs() < 1e-5);
        assert!((results[1].score as f64 - expected_rank2).abs() < 1e-5);
        assert!((results[2].score as f64 - expected_rank3).abs() < 1e-5);
    }

    #[test]
    fn test_rrf_multi_channel_score_accumulation() {
        let id_shared = uuid::Uuid::new_v4();
        let id_only_a = uuid::Uuid::new_v4();

        let shared = MemoryEntry {
            entry_id: id_shared,
            lossless_restatement: "Shared entry".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };
        let only_a = MemoryEntry {
            entry_id: id_only_a,
            lossless_restatement: "Only in A".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };

        let channels: [(&str, Vec<MemoryEntry>); 2] = [
            ("semantic", vec![shared.clone(), only_a]),
            ("lexical", vec![shared]),
        ];
        let results = HybridRetriever::reciprocal_rank_fusion(&channels, 10);

        assert_eq!(results.len(), 2);

        let shared_score_semantic = 1.0 / (60.0f64 + 1.0);
        let shared_score_lexical = 1.0 / (60.0f64 + 1.0);
        let shared_total = shared_score_semantic + shared_score_lexical;

        let only_a_score = 1.0 / (60.0f64 + 2.0);

        assert!((results[0].score as f64 - shared_total).abs() < 1e-5);
        assert!((results[1].score as f64 - only_a_score).abs() < 1e-5);
        assert_eq!(results[0].memory, "Shared entry");
        assert_eq!(results[1].memory, "Only in A");
    }

    #[test]
    fn test_rrf_top_k_truncation() {
        let entries: Vec<MemoryEntry> = (0..10)
            .map(|_| MemoryEntry::new(format!("Entry {}", uuid::Uuid::new_v4())))
            .collect();

        let channels: [(&str, Vec<MemoryEntry>); 1] = [("semantic", entries)];
        let results = HybridRetriever::reciprocal_rank_fusion(&channels, 3);

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_rrf_empty_channels() {
        let channels: [(&str, Vec<MemoryEntry>); 3] = [
            ("structured", vec![]),
            ("lexical", vec![]),
            ("semantic", vec![]),
        ];
        let results = HybridRetriever::reciprocal_rank_fusion(&channels, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rrf_sorted_by_score_descending() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        let e1 = MemoryEntry {
            entry_id: id1,
            lossless_restatement: "Low rank".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };
        let e2 = MemoryEntry {
            entry_id: id2,
            lossless_restatement: "High rank".to_string(),
            keywords: vec![],
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            cluster_id: None,
        };

        let channels: [(&str, Vec<MemoryEntry>); 2] =
            [("semantic", vec![e1.clone()]), ("lexical", vec![e2, e1])];
        let results = HybridRetriever::reciprocal_rank_fusion(&channels, 10);

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);

        let id1_score = 1.0 / (60.0f64 + 1.0) + 1.0 / (60.0f64 + 2.0);
        assert!((results[0].score as f64 - id1_score).abs() < 1e-5);
        assert_eq!(results[0].memory, "Low rank");
    }

    #[tokio::test]
    async fn test_multi_channel_retrieval_integration() {
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();

        let e1 = MemoryEntry {
            entry_id: id1,
            lossless_restatement: "Alice works at Google in Jakarta".to_string(),
            keywords: vec!["google".to_string()],
            timestamp: None,
            location: Some("Jakarta".to_string()),
            persons: vec!["Alice".to_string()],
            entities: vec!["Google".to_string()],
            topic: None,
            cluster_id: None,
        };
        let e2 = MemoryEntry {
            entry_id: id2,
            lossless_restatement: "Bob works at Apple in Bandung".to_string(),
            keywords: vec!["apple".to_string()],
            timestamp: None,
            location: Some("Bandung".to_string()),
            persons: vec!["Bob".to_string()],
            entities: vec!["Apple".to_string()],
            topic: None,
            cluster_id: None,
        };

        let analysis_response = serde_json::json!({
            "keywords": ["works", "Google"],
            "persons": ["Alice"],
            "entities": ["Google"],
            "location": "Jakarta",
            "time_expression": null
        });

        let llm = MockLlmClient::with_responses(vec![analysis_response]);
        let vector_store = MockVectorStore::with_entries(vec![
            (e1.clone(), vec![0.1; 64]),
            (e2.clone(), vec![0.2; 64]),
        ]);
        let embedder = MockEmbedder::new(64);
        let retriever = create_retriever(llm, vector_store, embedder);

        let results = retriever
            .retrieve("Where does Alice work?", Some(5))
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].memory.contains("Alice"));
    }
}
