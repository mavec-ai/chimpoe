use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryEntry {
    #[serde(default = "Uuid::new_v4")]
    pub entry_id: Uuid,
    pub lossless_restatement: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub location: Option<String>,
    #[serde(default)]
    pub persons: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    pub topic: Option<String>,
}

impl MemoryEntry {
    pub fn new(lossless_restatement: String) -> Self {
        Self {
            entry_id: Uuid::new_v4(),
            lossless_restatement,
            keywords: Vec::new(),
            timestamp: Some(Utc::now()),
            location: None,
            persons: Vec::new(),
            entities: Vec::new(),
            topic: None,
        }
    }

    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_persons(mut self, persons: Vec<String>) -> Self {
        self.persons = persons;
        self
    }

    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }

    pub fn with_topic(mut self, topic: String) -> Self {
        self.topic = Some(topic);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialogue {
    pub speaker: String,
    pub content: String,
    pub timestamp: Option<String>,
}

impl Dialogue {
    pub fn new(speaker: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            content: content.into(),
            timestamp: None,
        }
    }

    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }
}

impl std::fmt::Display for Dialogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref ts) = self.timestamp {
            write!(f, "[{}] {}: {}", ts, self.speaker, self.content)
        } else {
            write!(f, "{}: {}", self.speaker, self.content)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct StructuredSearchParams {
    pub persons: Option<Vec<String>>,
    pub location: Option<String>,
    pub entities: Option<Vec<String>>,
    pub timestamp_range: Option<TimeRange>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryAnalysis {
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub persons: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub time_expression: Option<String>,
}

impl From<QueryAnalysis> for StructuredSearchParams {
    fn from(analysis: QueryAnalysis) -> Self {
        Self {
            persons: if analysis.persons.is_empty() {
                None
            } else {
                Some(analysis.persons)
            },
            location: analysis.location,
            entities: if analysis.entities.is_empty() {
                None
            } else {
                Some(analysis.entities)
            },
            timestamp_range: None,
        }
    }
}
