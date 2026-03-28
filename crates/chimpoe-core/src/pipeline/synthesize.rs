use crate::config::SynthesizerConfig;
use crate::error::PipelineResult;
use crate::traits::Embedder;
use crate::types::MemoryEntry;
use std::collections::HashSet;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

pub struct Synthesizer {
    embedder: Option<Arc<dyn Embedder>>,
    config: SynthesizerConfig,
}

impl Default for Synthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Synthesizer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            embedder: None,
            config: SynthesizerConfig::default(),
        }
    }

    #[must_use]
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    #[must_use]
    pub fn with_config(mut self, config: SynthesizerConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn synthesize(&self, entries: Vec<MemoryEntry>) -> PipelineResult<Vec<MemoryEntry>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let deduped = Self::deduplicate_entries(entries);
        let merged = self.merge_similar_entries(deduped).await?;

        Ok(merged)
    }

    pub async fn filter_against_existing(
        &self,
        new_entries: Vec<MemoryEntry>,
        existing_with_vectors: &[(MemoryEntry, Vec<f32>)],
    ) -> PipelineResult<Vec<MemoryEntry>> {
        if new_entries.is_empty() || existing_with_vectors.is_empty() {
            return Ok(new_entries);
        }

        let new_embeddings = self.compute_embeddings(&new_entries).await?;

        let mut filtered: Vec<MemoryEntry> = Vec::new();

        for (i, new_entry) in new_entries.into_iter().enumerate() {
            let mut is_duplicate = false;

            for (existing_entry, existing_vec) in existing_with_vectors {
                let sim = self.compute_similarity(
                    &new_entry,
                    existing_entry,
                    new_embeddings.get(i),
                    Some(existing_vec),
                );

                if sim >= self.config.dedup_threshold {
                    is_duplicate = true;
                    break;
                }
            }

            if !is_duplicate {
                filtered.push(new_entry);
            }
        }

        Ok(filtered)
    }

    fn deduplicate_entries(entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut unique: Vec<MemoryEntry> = Vec::new();

        for entry in entries {
            let key: String = entry
                .lossless_restatement
                .nfkc()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .to_lowercase();
            if seen.insert(key) {
                unique.push(entry);
            }
        }

        unique
    }

    async fn merge_similar_entries(
        &self,
        entries: Vec<MemoryEntry>,
    ) -> PipelineResult<Vec<MemoryEntry>> {
        if entries.len() <= 1 {
            return Ok(entries);
        }

        let embeddings = self.compute_embeddings(&entries).await?;

        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; entries.len()];

        for (i, entry) in entries.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for (j, other) in entries.iter().enumerate() {
                if i != j && !assigned[j] {
                    let sim =
                        self.compute_similarity(entry, other, embeddings.get(i), embeddings.get(j));
                    if sim >= self.config.dedup_threshold {
                        cluster.push(j);
                        assigned[j] = true;
                    }
                }
            }

            clusters.push(cluster);
        }

        Ok(clusters
            .into_iter()
            .flat_map(|cluster| {
                let cluster_entries: Vec<MemoryEntry> =
                    cluster.into_iter().map(|i| entries[i].clone()).collect();
                Self::merge_cluster(cluster_entries)
            })
            .collect())
    }

    async fn compute_embeddings(&self, entries: &[MemoryEntry]) -> PipelineResult<Vec<Vec<f32>>> {
        match &self.embedder {
            Some(embedder) => {
                let texts: Vec<&str> = entries
                    .iter()
                    .map(|e| e.lossless_restatement.as_str())
                    .collect();
                embedder.encode(&texts).await.map_err(|e| e.into())
            }
            None => Ok(Vec::new()),
        }
    }

    fn compute_similarity(
        &self,
        a: &MemoryEntry,
        b: &MemoryEntry,
        embedding_a: Option<&Vec<f32>>,
        embedding_b: Option<&Vec<f32>>,
    ) -> f32 {
        let entity_jaccard = Self::entity_jaccard_similarity(a, b);

        match (embedding_a, embedding_b) {
            (Some(emb_a), Some(emb_b)) => {
                let semantic_sim = Self::cosine_similarity(emb_a, emb_b);
                self.config.keyword_weight * entity_jaccard
                    + self.config.semantic_weight * semantic_sim
            }
            _ => entity_jaccard,
        }
    }

    fn entity_jaccard_similarity(a: &MemoryEntry, b: &MemoryEntry) -> f32 {
        let mut all_a: HashSet<&str> = HashSet::new();
        let mut all_b: HashSet<&str> = HashSet::new();

        all_a.extend(a.keywords.iter().map(String::as_str));
        all_a.extend(a.persons.iter().map(String::as_str));
        all_a.extend(a.entities.iter().map(String::as_str));

        all_b.extend(b.keywords.iter().map(String::as_str));
        all_b.extend(b.persons.iter().map(String::as_str));
        all_b.extend(b.entities.iter().map(String::as_str));

        if all_a.is_empty() || all_b.is_empty() {
            return 0.0;
        }

        let intersection = all_a.intersection(&all_b).count();
        let union = all_a.union(&all_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
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

    fn merge_cluster(cluster: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        if cluster.len() == 1 {
            return cluster;
        }

        let mut merged_keywords: HashSet<String> = HashSet::new();
        let mut merged_persons: HashSet<String> = HashSet::new();
        let mut merged_entities: HashSet<String> = HashSet::new();
        let mut merged_location: Option<String> = None;
        let mut merged_topic: Option<String> = None;
        let mut earliest_timestamp: Option<chrono::DateTime<chrono::Utc>> = None;

        for entry in &cluster {
            merged_keywords.extend(entry.keywords.clone());
            merged_persons.extend(entry.persons.clone());
            merged_entities.extend(entry.entities.clone());

            if merged_location.is_none() && entry.location.is_some() {
                merged_location = entry.location.clone();
            }

            if merged_topic.is_none() && entry.topic.is_some() {
                merged_topic = entry.topic.clone();
            }

            if let Some(ts) = entry.timestamp {
                match earliest_timestamp {
                    None => earliest_timestamp = Some(ts),
                    Some(earliest) if ts < earliest => earliest_timestamp = Some(ts),
                    _ => {}
                }
            }
        }

        let cluster_id = uuid::Uuid::new_v4().to_string();

        cluster
            .into_iter()
            .map(|mut entry| {
                entry.keywords = merged_keywords.clone().into_iter().collect();
                entry.persons = merged_persons.clone().into_iter().collect();
                entry.entities = merged_entities.clone().into_iter().collect();
                if entry.location.is_none() {
                    entry.location = merged_location.clone();
                }
                if entry.topic.is_none() {
                    entry.topic = merged_topic.clone();
                }
                if entry.timestamp.is_none() {
                    entry.timestamp = earliest_timestamp;
                }
                entry.cluster_id = Some(cluster_id.clone());
                entry
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(text: &str, keywords: Vec<&str>) -> MemoryEntry {
        MemoryEntry::new(text.to_string())
            .with_keywords(keywords.into_iter().map(String::from).collect())
    }

    #[tokio::test]
    async fn test_synthesize_empty_input() {
        let synth = Synthesizer::new();
        let result = synth.synthesize(Vec::new()).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_synthesize_single_entry() {
        let synth = Synthesizer::new();
        let entry = make_entry("User likes pizza", vec!["pizza", "food"]);
        let result = synth.synthesize(vec![entry]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lossless_restatement, "User likes pizza");
    }

    #[tokio::test]
    async fn test_deduplicate_exact_duplicates() {
        let synth = Synthesizer::new();
        let entry1 = make_entry("User likes pizza", vec!["pizza"]);
        let entry2 = make_entry("User likes pizza", vec!["food"]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_deduplicate_case_insensitive() {
        let synth = Synthesizer::new();
        let entry1 = make_entry("User likes pizza", vec!["pizza"]);
        let entry2 = make_entry("USER LIKES PIZZA", vec!["food"]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_merge_similar_entries() {
        let synth = Synthesizer::new();

        let shared = vec![
            "food".to_string(),
            "italian".to_string(),
            "dinner".to_string(),
            "restaurant".to_string(),
            "cuisine".to_string(),
        ];
        let mut entry1_keywords = shared.clone();
        entry1_keywords.push("pizza".to_string());
        let mut entry2_keywords = shared.clone();
        entry2_keywords.push("pasta".to_string());

        let entry1 =
            MemoryEntry::new("User likes pizza".to_string()).with_keywords(entry1_keywords);
        let entry2 =
            MemoryEntry::new("User enjoys pasta".to_string()).with_keywords(entry2_keywords);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);
        assert!(result[0].keywords.contains(&"food".to_string()));
        assert!(result[1].keywords.contains(&"food".to_string()));
    }

    #[tokio::test]
    async fn test_no_merge_dissimilar_entries() {
        let synth = Synthesizer::new();

        let entry1 = make_entry("User likes pizza", vec!["pizza", "food"]);
        let entry2 = make_entry("User works at Google", vec!["google", "work"]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_merge_cluster_combines_persons_and_entities() {
        let synth = Synthesizer::new();

        let shared = vec![
            "cafe".to_string(),
            "meeting".to_string(),
            "morning".to_string(),
            "weekday".to_string(),
            "coffeeshop".to_string(),
            "drink".to_string(),
            "social".to_string(),
            "friend".to_string(),
        ];

        let entry1 = MemoryEntry::new("Met John at cafe".to_string())
            .with_keywords(shared.clone())
            .with_persons(vec!["John".to_string()]);

        let entry2 = MemoryEntry::new("Met Mary at cafe".to_string())
            .with_keywords(shared)
            .with_persons(vec!["Mary".to_string()]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        for entry in &result {
            assert!(entry.persons.contains(&"John".to_string()));
            assert!(entry.persons.contains(&"Mary".to_string()));
        }
    }

    #[tokio::test]
    async fn test_merge_cluster_preserves_individual_statements() {
        let synth = Synthesizer::new();

        let short = make_entry("Short", vec!["shared", "keywords"]);
        let long = make_entry(
            "This is a much longer statement with more details",
            vec!["shared", "keywords"],
        );

        let result = synth.synthesize(vec![short, long]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        let statements: Vec<&str> = result
            .iter()
            .map(|e| e.lossless_restatement.as_str())
            .collect();
        assert!(
            statements.contains(&"Short"),
            "Short statement should be preserved"
        );
        assert!(
            statements
                .iter()
                .any(|s| s.contains("This is a much longer")),
            "Long statement should be preserved"
        );
    }

    #[test]
    fn test_entity_jaccard_similarity_identical() {
        let a = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["a".to_string(), "b".to_string()])
            .with_persons(vec!["John".to_string()]);
        let b = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["a".to_string(), "b".to_string()])
            .with_persons(vec!["John".to_string()]);

        let sim = Synthesizer::entity_jaccard_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_entity_jaccard_similarity_partial() {
        let a = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["a".to_string(), "b".to_string()])
            .with_persons(vec!["John".to_string()]);
        let b = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["b".to_string(), "c".to_string()])
            .with_persons(vec!["Mary".to_string()]);

        let sim = Synthesizer::entity_jaccard_similarity(&a, &b);
        assert!((sim - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_entity_jaccard_similarity_no_overlap() {
        let a = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["a".to_string(), "b".to_string()]);
        let b = MemoryEntry::new("test".to_string())
            .with_keywords(vec!["c".to_string(), "d".to_string()]);

        let sim = Synthesizer::entity_jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_filter_against_existing_empty_new() {
        let synth = Synthesizer::new();
        let existing = vec![(
            make_entry("old memory", vec!["old"]),
            vec![1.0_f32, 0.0, 0.0],
        )];
        let result = synth
            .filter_against_existing(Vec::new(), &existing)
            .await
            .unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_filter_against_existing_empty_existing() {
        let synth = Synthesizer::new();
        let new_entries = vec![make_entry("new memory", vec!["new"])];
        let result = synth
            .filter_against_existing(new_entries.clone(), &[])
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_filter_against_existing_keeps_novel() {
        let synth = Synthesizer::new();
        let new_entry = make_entry("User visited Tokyo", vec!["tokyo", "travel"]);
        let existing = vec![(
            make_entry("User works at Google", vec!["google", "work"]),
            vec![1.0_f32, 0.0, 0.0],
        )];
        let result = synth
            .filter_against_existing(vec![new_entry], &existing)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_deduplicate_nfkc_unicode_normalization() {
        let synth = Synthesizer::new();
        let entry1 = make_entry("User likes Ｐｉｚｚａ", vec!["pizza"]);
        let entry2 = make_entry("User likes Pizza", vec!["food"]);
        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_merge_cluster_preserves_location_and_topic() {
        let synth = Synthesizer::new();

        let shared = vec![
            "project".to_string(),
            "work".to_string(),
            "team".to_string(),
            "deadline".to_string(),
            "sprint".to_string(),
            "meeting".to_string(),
            "collab".to_string(),
            "agile".to_string(),
        ];

        let entry1 = MemoryEntry::new("Discussed roadmap".to_string())
            .with_keywords(shared.clone())
            .with_location("Office Room A".to_string())
            .with_topic("Project Planning".to_string());

        let entry2 = MemoryEntry::new("Reviewed backlog items".to_string())
            .with_keywords(shared)
            .with_persons(vec!["Alice".to_string()]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        for entry in &result {
            assert_eq!(entry.location.as_deref(), Some("Office Room A"));
            assert_eq!(entry.topic.as_deref(), Some("Project Planning"));
            assert!(entry.persons.contains(&"Alice".to_string()));
        }
    }

    #[tokio::test]
    async fn test_merge_cluster_earliest_timestamp() {
        let synth = Synthesizer::new();

        let shared = vec![
            "event".to_string(),
            "conference".to_string(),
            "talk".to_string(),
            "speaker".to_string(),
            "venue".to_string(),
            "audience".to_string(),
            "keynote".to_string(),
            "schedule".to_string(),
        ];

        let mut entry1 =
            MemoryEntry::new("Morning keynote".to_string()).with_keywords(shared.clone());
        entry1.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2025-06-15T10:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let mut entry2 = MemoryEntry::new("Afternoon workshop".to_string()).with_keywords(shared);
        entry2.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2025-06-14T08:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        let timestamps: Vec<String> = result
            .iter()
            .map(|e| e.timestamp.unwrap().format("%Y-%m-%dT%H:%M:%S").to_string())
            .collect();
        assert!(
            timestamps.contains(&"2025-06-14T08:00:00".to_string()),
            "Earliest timestamp should be in cluster"
        );
        assert!(
            timestamps.contains(&"2025-06-15T10:00:00".to_string()),
            "Later timestamp should be in cluster"
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let sim = Synthesizer::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let sim = Synthesizer::cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 0.0];

        let sim = Synthesizer::cosine_similarity(&a, &b);
        let expected: f32 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sim - expected).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_temporal_overwrite_preserves_all_facts() {
        let synth = Synthesizer::new();

        let shared = vec![
            "Joko".to_string(),
            "money".to_string(),
            "debt".to_string(),
            "loan".to_string(),
            "repay".to_string(),
            "pay".to_string(),
            "salary".to_string(),
        ];

        let mut entry1 = MemoryEntry::new("Joko borrowed $50 on 2026-03-27.".to_string())
            .with_keywords(shared.clone())
            .with_persons(vec!["Joko".to_string()]);
        entry1.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2026-03-27T12:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let mut entry2 = MemoryEntry::new(
            "Joko promised to repay his debt on 2026-03-30 after receiving his salary.".to_string(),
        )
        .with_keywords(shared)
        .with_persons(vec!["Joko".to_string()]);
        entry2.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2026-03-28T08:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        let statements: Vec<&str> = result
            .iter()
            .map(|e| e.lossless_restatement.as_str())
            .collect();
        assert!(
            statements.iter().any(|s| s.contains("$50")),
            "Loan amount should be preserved"
        );
        assert!(
            statements
                .iter()
                .any(|s| s.contains("repay") || s.contains("salary")),
            "Promise to repay should be preserved"
        );
    }

    #[tokio::test]
    async fn test_directionality_preserves_both_directions() {
        let synth = Synthesizer::new();

        let shared = vec![
            "Budi".to_string(),
            "Andi".to_string(),
            "hit".to_string(),
            "conflict".to_string(),
            "debt".to_string(),
            "violence".to_string(),
            "afternoon".to_string(),
        ];

        let entry1 =
            MemoryEntry::new("Budi hit Andi yesterday afternoon over a debt dispute.".to_string())
                .with_keywords(shared.clone())
                .with_persons(vec!["Budi".to_string(), "Andi".to_string()]);

        let entry2 =
            MemoryEntry::new("Andi hit Budi in front of the school as revenge.".to_string())
                .with_keywords(shared)
                .with_persons(vec!["Budi".to_string(), "Andi".to_string()]);

        let result = synth.synthesize(vec![entry1, entry2]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        let statements: Vec<&str> = result
            .iter()
            .map(|e| e.lossless_restatement.as_str())
            .collect();
        assert!(
            statements.iter().any(|s| s.contains("Budi hit Andi")),
            "Attack Budi->Andi should be preserved"
        );
        assert!(
            statements.iter().any(|s| s.contains("Andi hit Budi")),
            "Attack Andi->Budi should be preserved"
        );
    }

    #[tokio::test]
    async fn test_merge_timestamp_ordering() {
        let synth = Synthesizer::new();

        let shared = vec![
            "project".to_string(),
            "update".to_string(),
            "status".to_string(),
            "deadline".to_string(),
            "sprint".to_string(),
            "team".to_string(),
        ];

        let mut later =
            MemoryEntry::new("Alpha phase completed.".to_string()).with_keywords(shared.clone());
        later.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2026-03-28T14:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let mut earlier = MemoryEntry::new("Beta phase started.".to_string()).with_keywords(shared);
        earlier.timestamp = Some(
            chrono::DateTime::parse_from_rfc3339("2026-03-27T09:00:00Z")
                .unwrap()
                .to_utc(),
        );

        let result = synth.synthesize(vec![later, earlier]).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].cluster_id.is_some());
        assert_eq!(result[0].cluster_id, result[1].cluster_id);

        let statements: Vec<&str> = result
            .iter()
            .map(|e| e.lossless_restatement.as_str())
            .collect();
        assert!(
            statements.iter().any(|s| s.contains("Alpha")),
            "Alpha statement should be preserved"
        );
        assert!(
            statements.iter().any(|s| s.contains("Beta")),
            "Beta statement should be preserved"
        );
    }
}
