use crate::config::RetrievalConfig;
use crate::error::Result;
use crate::traits::{Embedder, LlmClient, Message, MessageRole, VectorStore};
use crate::types::{MemoryEntry, QueryAnalysis, StructuredSearchParams, TimeRange};
use chrono::Utc;
use chrono_english::{Dialect, parse_date_string};
use std::sync::Arc;

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

fn parse_time_expression(expr: &str) -> Option<TimeRange> {
    let now = Utc::now();

    let parsed = parse_date_string(expr, now, Dialect::Uk).ok()?;

    let start = if parsed > now { now } else { parsed };

    Some(TimeRange { start, end: now })
}

fn build_search_params(analysis: &QueryAnalysis) -> StructuredSearchParams {
    let timestamp_range = analysis
        .time_expression
        .as_ref()
        .and_then(|expr| parse_time_expression(expr));

    StructuredSearchParams {
        persons: if analysis.persons.is_empty() {
            None
        } else {
            Some(analysis.persons.clone())
        },
        location: analysis.location.clone(),
        entities: if analysis.entities.is_empty() {
            None
        } else {
            Some(analysis.entities.clone())
        },
        timestamp_range,
    }
}

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

        match self.llm.chat_completion_with_json(&messages, 0.1).await {
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

        let semantic_results = self.semantic_search(query, k).await?;

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

        let lexical_results = if !keywords.is_empty() {
            self.keyword_search(&keywords, self.keyword_top_k).await?
        } else {
            Vec::new()
        };

        let structured_results = if let Some(ref analysis) = query_analysis {
            let has_structured_conditions = !analysis.persons.is_empty()
                || !analysis.entities.is_empty()
                || analysis.location.is_some()
                || analysis.time_expression.is_some();

            if has_structured_conditions {
                let params = build_search_params(analysis);
                self.structured_search(&params, self.structured_top_k)
                    .await?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let merged =
            self.merge_and_deduplicate(structured_results, semantic_results, lexical_results, k);

        Ok(merged)
    }

    async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<MemoryEntry>> {
        let query_vector = self.embedder.encode_single(query).await?;
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
    fn from_entry(entry: MemoryEntry, source: &str) -> Self {
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
