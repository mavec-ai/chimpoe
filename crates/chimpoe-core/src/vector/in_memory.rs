use crate::error::{VectorError, VectorResult};
use crate::traits::VectorStore;
use crate::types::{MemoryEntry, StructuredSearchParams};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

type VectorEntry = (MemoryEntry, Vec<f32>);
type VectorData = Arc<RwLock<HashMap<Uuid, VectorEntry>>>;

pub struct InMemoryVector {
    data: VectorData,
}

impl InMemoryVector {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryVector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVector {
    async fn add_entries(&self, entries: &[MemoryEntry], vectors: &[Vec<f32>]) -> VectorResult<()> {
        if entries.len() != vectors.len() {
            return Err(VectorError::InsertionFailed(
                "entries and vectors length mismatch".to_string(),
            ));
        }

        let mut data = self.data.write().await;
        for (entry, vector) in entries.iter().zip(vectors.iter()) {
            data.insert(entry.entry_id, (entry.clone(), vector.clone()));
        }

        Ok(())
    }

    async fn semantic_search(
        &self,
        query_vector: &[f32],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let data = self.data.read().await;

        let mut scored: Vec<(f32, MemoryEntry)> = data
            .values()
            .map(|(entry, vec)| {
                let score = cosine_similarity(query_vector, vec);
                (score, entry.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, entry)| entry).collect())
    }

    async fn keyword_search(
        &self,
        keywords: &[String],
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let data = self.data.read().await;

        let mut scored: Vec<(usize, MemoryEntry)> = data
            .values()
            .map(|(entry, _)| {
                let score = keywords
                    .iter()
                    .filter(|kw| {
                        entry
                            .keywords
                            .iter()
                            .any(|k| k.to_lowercase().contains(&kw.to_lowercase()))
                            || entry
                                .lossless_restatement
                                .to_lowercase()
                                .contains(&kw.to_lowercase())
                    })
                    .count();
                (score, entry.clone())
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, entry)| entry).collect())
    }

    async fn structured_search(
        &self,
        params: &StructuredSearchParams,
        top_k: usize,
    ) -> VectorResult<Vec<MemoryEntry>> {
        let data = self.data.read().await;

        let results: Vec<MemoryEntry> = data
            .values()
            .filter_map(|(entry, _)| {
                if let Some(ref persons) = params.persons
                    && !persons.is_empty()
                {
                    let has_any = persons
                        .iter()
                        .any(|p| entry.persons.iter().any(|ep| ep.eq_ignore_ascii_case(p)));
                    if !has_any {
                        return None;
                    }
                }

                if let Some(ref location) = params.location {
                    let matches = entry
                        .location
                        .as_ref()
                        .map(|el| el.to_lowercase().contains(&location.to_lowercase()))
                        .unwrap_or(false);
                    if !matches {
                        return None;
                    }
                }

                if let Some(ref entities) = params.entities
                    && !entities.is_empty()
                {
                    let has_any = entities
                        .iter()
                        .any(|e| entry.entities.iter().any(|ee| ee.eq_ignore_ascii_case(e)));
                    if !has_any {
                        return None;
                    }
                }

                if let Some(ref time_range) = params.timestamp_range {
                    if let Some(ref ts) = entry.timestamp {
                        if ts < &time_range.start || ts > &time_range.end {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }

                Some(entry.clone())
            })
            .take(top_k)
            .collect();

        Ok(results)
    }

    async fn delete_entry(&self, entry_id: &Uuid) -> VectorResult<bool> {
        let mut data = self.data.write().await;
        Ok(data.remove(entry_id).is_some())
    }

    async fn count(&self) -> VectorResult<usize> {
        let data = self.data.read().await;
        Ok(data.len())
    }

    async fn get_all_entries(&self) -> VectorResult<Vec<MemoryEntry>> {
        let data = self.data.read().await;
        Ok(data.values().map(|(entry, _)| entry.clone()).collect())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}
