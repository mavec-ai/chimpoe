use crate::config::PipelineConfig;
use crate::error::PipelineResult;
use crate::types::MemoryEntry;
use std::collections::HashSet;

pub struct Synthesizer {
    config: PipelineConfig,
}

impl Synthesizer {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn synthesize(&self, entries: Vec<MemoryEntry>) -> PipelineResult<Vec<MemoryEntry>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let deduped = self.deduplicate_entries(entries);
        let merged = self.merge_similar_entries(deduped);

        Ok(merged)
    }

    fn deduplicate_entries(&self, entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut unique: Vec<MemoryEntry> = Vec::new();

        for entry in entries {
            let key = entry.lossless_restatement.to_lowercase();
            if !seen.contains(&key) {
                seen.insert(key);
                unique.push(entry);
            }
        }

        unique
    }

    fn merge_similar_entries(&self, entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        if entries.len() <= 1 {
            return entries;
        }

        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut assigned: Vec<bool> = vec![false; entries.len()];

        for (i, entry) in entries.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for (j, other) in entries.iter().enumerate() {
                if i != j && !assigned[j] && self.are_similar(entry, other) {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
            .into_iter()
            .map(|cluster| {
                let cluster_entries: Vec<MemoryEntry> =
                    cluster.into_iter().map(|i| entries[i].clone()).collect();
                self.merge_cluster(cluster_entries)
            })
            .collect()
    }

    fn are_similar(&self, a: &MemoryEntry, b: &MemoryEntry) -> bool {
        let jaccard = self.jaccard_similarity(&a.keywords, &b.keywords);
        jaccard >= self.config.similarity_threshold
    }

    fn jaccard_similarity(&self, a: &[String], b: &[String]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
        let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn merge_cluster(&self, cluster: Vec<MemoryEntry>) -> MemoryEntry {
        if cluster.len() == 1 {
            return cluster.into_iter().next().unwrap();
        }

        let mut merged_keywords: HashSet<String> = HashSet::new();
        let mut merged_persons: HashSet<String> = HashSet::new();
        let mut merged_entities: HashSet<String> = HashSet::new();
        let mut merged_location: Option<String> = None;
        let mut merged_topic: Option<String> = None;
        let mut earliest_timestamp: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut richest_statement = String::new();

        for entry in cluster {
            merged_keywords.extend(entry.keywords);
            merged_persons.extend(entry.persons);
            merged_entities.extend(entry.entities);

            if merged_location.is_none() && entry.location.is_some() {
                merged_location = entry.location;
            }

            if merged_topic.is_none() && entry.topic.is_some() {
                merged_topic = entry.topic;
            }

            if let Some(ts) = entry.timestamp {
                match earliest_timestamp {
                    None => earliest_timestamp = Some(ts),
                    Some(earliest) if ts < earliest => earliest_timestamp = Some(ts),
                    _ => {}
                }
            }

            if entry.lossless_restatement.len() > richest_statement.len() {
                richest_statement = entry.lossless_restatement;
            }
        }

        MemoryEntry {
            entry_id: uuid::Uuid::new_v4(),
            lossless_restatement: richest_statement,
            keywords: merged_keywords.into_iter().collect(),
            timestamp: earliest_timestamp,
            location: merged_location,
            persons: merged_persons.into_iter().collect(),
            entities: merged_entities.into_iter().collect(),
            topic: merged_topic,
        }
    }
}
