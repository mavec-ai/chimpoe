use crate::config::PipelineConfig;
use crate::error::PipelineResult;
use crate::types::MemoryEntry;
use std::collections::HashSet;

pub struct Synthesizer {
    config: PipelineConfig,
}

impl Synthesizer {
    #[must_use]
    pub const fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    pub fn synthesize(&self, entries: Vec<MemoryEntry>) -> PipelineResult<Vec<MemoryEntry>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let deduped = Self::deduplicate_entries(entries);
        let merged = self.merge_similar_entries(deduped);

        Ok(merged)
    }

    fn deduplicate_entries(entries: Vec<MemoryEntry>) -> Vec<MemoryEntry> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut unique: Vec<MemoryEntry> = Vec::new();

        for entry in entries {
            let key = entry.lossless_restatement.to_lowercase();
            if seen.insert(key) {
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
        let jaccard = Self::jaccard_similarity(&a.keywords, &b.keywords);
        jaccard >= self.config.similarity_threshold
    }

    fn jaccard_similarity(a: &[String], b: &[String]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let set_a: HashSet<&str> = a.iter().map(std::string::String::as_str).collect();
        let set_b: HashSet<&str> = b.iter().map(std::string::String::as_str).collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetrievalConfig;

    fn test_config() -> PipelineConfig {
        PipelineConfig {
            window_size: 10,
            overlap_size: 2,
            similarity_threshold: 0.5,
            retrieval: RetrievalConfig::default(),
        }
    }

    fn make_entry(text: &str, keywords: Vec<&str>) -> MemoryEntry {
        MemoryEntry::new(text.to_string())
            .with_keywords(keywords.into_iter().map(String::from).collect())
    }

    #[test]
    fn test_synthesize_empty_input() {
        let synth = Synthesizer::new(test_config());
        let result = synth.synthesize(Vec::new()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_synthesize_single_entry() {
        let synth = Synthesizer::new(test_config());
        let entry = make_entry("User likes pizza", vec!["pizza", "food"]);
        let result = synth.synthesize(vec![entry]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lossless_restatement, "User likes pizza");
    }

    #[test]
    fn test_deduplicate_exact_duplicates() {
        let synth = Synthesizer::new(test_config());
        let entry1 = make_entry("User likes pizza", vec!["pizza"]);
        let entry2 = make_entry("User likes pizza", vec!["food"]);

        let result = synth.synthesize(vec![entry1, entry2]).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_deduplicate_case_insensitive() {
        let synth = Synthesizer::new(test_config());
        let entry1 = make_entry("User likes pizza", vec!["pizza"]);
        let entry2 = make_entry("USER LIKES PIZZA", vec!["food"]);

        let result = synth.synthesize(vec![entry1, entry2]).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_merge_similar_entries() {
        let config = PipelineConfig {
            similarity_threshold: 0.3,
            ..test_config()
        };
        let synth = Synthesizer::new(config);

        let entry1 = make_entry("User likes pizza", vec!["pizza", "food", "italian"]);
        let entry2 = make_entry("User enjoys pasta", vec!["pasta", "food", "italian"]);

        let result = synth.synthesize(vec![entry1, entry2]).unwrap();
        assert_eq!(result.len(), 1);

        let merged = &result[0];
        assert!(merged.keywords.contains(&"pizza".to_string()));
        assert!(merged.keywords.contains(&"pasta".to_string()));
        assert!(merged.keywords.contains(&"food".to_string()));
    }

    #[test]
    fn test_no_merge_dissimilar_entries() {
        let synth = Synthesizer::new(test_config());

        let entry1 = make_entry("User likes pizza", vec!["pizza", "food"]);
        let entry2 = make_entry("User works at Google", vec!["google", "work"]);

        let result = synth.synthesize(vec![entry1, entry2]).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_cluster_combines_persons_and_entities() {
        let config = PipelineConfig {
            similarity_threshold: 0.5,
            ..test_config()
        };
        let synth = Synthesizer::new(config);

        let entry1 = MemoryEntry::new("Met John at cafe".to_string())
            .with_keywords(vec!["cafe".to_string(), "meeting".to_string()])
            .with_persons(vec!["John".to_string()]);

        let entry2 = MemoryEntry::new("Met Mary at cafe".to_string())
            .with_keywords(vec!["cafe".to_string(), "meeting".to_string()])
            .with_persons(vec!["Mary".to_string()]);

        let result = synth.synthesize(vec![entry1, entry2]).unwrap();
        assert_eq!(result.len(), 1);

        let merged = &result[0];
        assert!(merged.persons.contains(&"John".to_string()));
        assert!(merged.persons.contains(&"Mary".to_string()));
    }

    #[test]
    fn test_merge_keeps_longest_statement() {
        let config = PipelineConfig {
            similarity_threshold: 0.5,
            ..test_config()
        };
        let synth = Synthesizer::new(config);

        let short = make_entry("Short", vec!["shared", "keywords"]);
        let long = make_entry(
            "This is a much longer statement with more details",
            vec!["shared", "keywords"],
        );

        let result = synth.synthesize(vec![short, long]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result[0]
                .lossless_restatement
                .starts_with("This is a much longer")
        );
    }

    #[test]
    fn test_jaccard_similarity_identical_sets() {
        let _synth = Synthesizer::new(test_config());
        let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let b = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let sim = Synthesizer::jaccard_similarity(&a, &b);
        assert!((sim - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_no_overlap() {
        let _synth = Synthesizer::new(test_config());
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["c".to_string(), "d".to_string()];

        let sim = Synthesizer::jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_half_overlap() {
        let _synth = Synthesizer::new(test_config());
        let a = vec!["a".to_string(), "b".to_string()];
        let b = vec!["b".to_string(), "c".to_string()];

        let sim = Synthesizer::jaccard_similarity(&a, &b);
        assert!((sim - 0.333_333_3).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_similarity_empty_sets() {
        let a: Vec<String> = Vec::new();
        let b = vec!["a".to_string()];

        let sim = Synthesizer::jaccard_similarity(&a, &b);
        assert!((sim - 0.0).abs() < f32::EPSILON);

        let sim = Synthesizer::jaccard_similarity(&b, &a);
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }
}
