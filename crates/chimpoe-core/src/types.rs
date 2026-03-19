use chrono::{DateTime, Duration, NaiveTime, Utc};
use chrono_english::{Dialect, parse_date_string};
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

impl TimeRange {
    pub fn parse(expression: &str) -> Option<Self> {
        let base_time = Utc::now();
        let lower = expression.to_lowercase();

        let parsed = if lower.contains("last week") {
            base_time - Duration::days(7)
        } else if lower.contains("last month") {
            base_time - Duration::days(30)
        } else {
            parse_date_string(expression, base_time, Dialect::Us).ok()?
        };

        let start = parsed
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(0, 0, 0)?)
            .and_utc();
        let end = parsed
            .date_naive()
            .and_time(NaiveTime::from_hms_opt(23, 59, 59)?)
            .and_utc();

        let (final_start, final_end) = if lower.contains("week") {
            (start - Duration::days(7), end + Duration::days(7))
        } else if lower.contains("month") {
            (start - Duration::days(30), end + Duration::days(30))
        } else {
            (start, end)
        };

        Some(Self {
            start: final_start,
            end: final_end,
        })
    }
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
        let timestamp_range = analysis
            .time_expression
            .as_ref()
            .and_then(|expr| TimeRange::parse(expr));

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
            timestamp_range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_range_yesterday() {
        let range = TimeRange::parse("yesterday").unwrap();
        assert!(range.start < range.end);
    }

    #[test]
    fn test_time_range_last_week() {
        let range = TimeRange::parse("last week").unwrap();
        let now = Utc::now();
        assert!(range.start < now);
        assert!(range.end > range.start);
    }

    #[test]
    fn test_time_range_today() {
        let range = TimeRange::parse("today").unwrap();
        let now = Utc::now();
        assert!(range.start <= now);
        assert!(range.end >= now);
    }

    #[test]
    fn test_time_range_invalid() {
        let range = TimeRange::parse("invalid time expression");
        assert!(range.is_none());
    }

    #[test]
    fn test_query_analysis_to_params_with_time() {
        let analysis = QueryAnalysis {
            keywords: vec!["test".to_string()],
            persons: vec!["John".to_string()],
            entities: Vec::new(),
            location: None,
            time_expression: Some("yesterday".to_string()),
        };

        let params: StructuredSearchParams = analysis.into();
        assert!(params.timestamp_range.is_some());
        assert!(params.persons.is_some());
    }

    #[test]
    fn test_query_analysis_to_params_without_time() {
        let analysis = QueryAnalysis {
            keywords: vec!["test".to_string()],
            persons: vec!["John".to_string()],
            entities: Vec::new(),
            location: None,
            time_expression: None,
        };

        let params: StructuredSearchParams = analysis.into();
        assert!(params.timestamp_range.is_none());
    }
}
