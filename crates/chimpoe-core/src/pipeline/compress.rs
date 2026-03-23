use crate::config::PipelineConfig;
use crate::error::{PipelineError, PipelineResult};
use crate::traits::{LlmClient, Message};
use crate::types::{Dialogue, MemoryEntry};
use std::sync::Arc;

const EXTRACTION_PROMPT: &str = r#"
Your task is to extract structured memory entries from the following dialogues.

Current Date: {current_date}

{previous_context}

[Current Window Dialogues]
{conversation}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to capture ALL valuable information from the dialogues
2. **Language Preservation (CRITICAL)**:
   - Output lossless_restatement in THE EXACT SAME LANGUAGE as dialogue
   - NEVER translate, convert, or change language
   - Indonesian input → Indonesian output
   - English input → English output
   - WRONG: "Saya pergi ke pasar" → "I went to the market" (TRANSLATED!)
   - RIGHT: "Saya pergi ke pasar" → "Saya pergi ke pasar pagi ini" (SAME LANGUAGE)
3. **No Hallucination (CRITICAL)**:
   - ONLY information EXPLICITLY stated in dialogue
   - Do NOT add names not mentioned (keep pronouns as-is, do NOT invent names)
   - Do NOT infer, assume, or guess
   - When in doubt, omit
4. **Lossless Information**: Each lossless_restatement must be complete, independent, and understandable without context
5. **Force Disambiguation**: NO pronouns (he, she, it, they, this, that). Use actual names.
6. **Temporal Anchoring (CRITICAL)**:
   - Convert ALL relative time expressions to absolute ISO 8601 format
   - Use Current Date as reference point for calculation
   - Examples (apply same logic to ANY language):
     * "yesterday" / "kemarin" / "昨日" / "أمس" → Current Date - 1 day
     * "tomorrow" / "besok" / "明日" → Current Date + 1 day
     * "last week" / "minggu lalu" / "先週" → Current Date - 7 days
     * "2 days ago" / "2 hari lalu" → Current Date - 2 days
   - Include the resolved absolute timestamp in the "timestamp" field
   - In lossless_restatement, use absolute dates instead of relative expressions
   - Example: "User bought laptop yesterday" → "User bought laptop on 2026-03-22"
7. **Precise Metadata**:
   - keywords: Core keywords (names, places, entities, topic words)
   - persons: ONLY proper names (Alice, Bob). SKIP pronouns.
   - location: ONLY specific places (Starbucks, Jakarta). SKIP vague references (here, there).
   - entities: ONLY clear named entities (Google, iPhone). SKIP generic terms.
   - topic: Brief topic/category of this information (e.g., "Meeting arrangement", "Travel plan")
   - If unclear or vague → leave empty, do NOT guess.
8. **Direct Factual Statement (CRITICAL)**: Write lossless_restatement as DIRECT facts, NOT reported speech.
   - WRONG: "User stated that user likes programming" (reported speech)
   - WRONG: "User mentioned that they work at Google" (reported speech)
   - RIGHT: "User likes programming" (direct statement)
   - RIGHT: "User works at Google" (direct statement)
   - Always use simple Subject-Verb-Object structure.

[Output Format]
Return a JSON object with a "memories" array:

```json
{{
  "memories": [
    {{
      "lossless_restatement": "Complete unambiguous restatement with ABSOLUTE dates in the SAME LANGUAGE as dialogue",
      "keywords": ["keyword1", "keyword2"],
      "persons": ["name1", "name2"],
      "entities": ["entity1", "entity2"],
      "location": "location name or null",
      "topic": "topic phrase",
      "timestamp": "YYYY-MM-DDTHH:MM:SS or null"
    }}
  ]
}}
```

[Example - FOR FORMAT REFERENCE ONLY, DO NOT COPY THE CONTENT]
Current Date: 2025-11-15

Dialogues:
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
[2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials

Output:
```json
{{
  "memories": [
    {{
      "lossless_restatement": "Alice will meet Bob at Starbucks on 2025-11-16 at 2pm to discuss the new product.",
      "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
      "persons": ["Alice", "Bob"],
      "entities": ["new product"],
      "location": "Starbucks",
      "topic": "Product discussion meeting arrangement",
      "timestamp": "2025-11-16T14:00:00"
    }},
    {{
      "lossless_restatement": "Bob agreed on 2025-11-15 to attend the meeting and committed to prepare relevant materials.",
      "keywords": ["Bob", "prepare materials", "agree"],
      "persons": ["Bob"],
      "entities": [],
      "location": null,
      "topic": "Meeting preparation confirmation",
      "timestamp": "2025-11-15T14:31:00"
    }}
  ]
}}
```

IMPORTANT: 
- Process the ACTUAL dialogues in [Current Window Dialogues] above.
- The examples show FORMAT only - do not copy their content.
- Output in THE SAME LANGUAGE as the dialogue (Indonesian → Indonesian, English → English).
- ALWAYS convert relative times to absolute dates using Current Date.

Return ONLY valid JSON, no markdown or explanation outside the JSON structure.
"#;

const DEFAULT_TEMPERATURE: f32 = 0.1;

pub struct Compressor {
    llm: Arc<dyn LlmClient>,
    config: PipelineConfig,
}

impl Compressor {
    pub fn new(llm: Arc<dyn LlmClient>, config: PipelineConfig) -> Self {
        Self { llm, config }
    }

    pub async fn compress_dialogues(
        &self,
        dialogues: &[Dialogue],
    ) -> PipelineResult<Vec<MemoryEntry>> {
        let windows = self.create_windows(dialogues);
        let mut all_entries = Vec::new();
        let mut previous_entries: Vec<MemoryEntry> = Vec::new();

        for window in windows {
            let entries = self.process_window(&window, &previous_entries).await?;
            previous_entries.clone_from(&entries);
            all_entries.extend(entries);
        }

        Ok(all_entries)
    }

    fn create_windows<'a>(&self, dialogues: &'a [Dialogue]) -> Vec<Vec<&'a Dialogue>> {
        if dialogues.is_empty() {
            return Vec::new();
        }

        let window_size = self.config.window_size;
        let overlap = self.config.overlap_size;
        let mut windows = Vec::new();
        let mut start = 0;

        while start < dialogues.len() {
            let end = (start + window_size).min(dialogues.len());
            let window: Vec<&Dialogue> = dialogues[start..end].iter().collect();
            windows.push(window);

            if end >= dialogues.len() {
                break;
            }

            start += window_size - overlap;
        }

        windows
    }

    async fn process_window(
        &self,
        window: &[&Dialogue],
        previous_entries: &[MemoryEntry],
    ) -> PipelineResult<Vec<MemoryEntry>> {
        let conversation = Self::format_conversation(window);
        let previous_context = Self::format_previous_context(previous_entries);
        let current_date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let prompt = EXTRACTION_PROMPT
            .replace("{current_date}", &current_date)
            .replace("{conversation}", &conversation)
            .replace("{previous_context}", &previous_context);

        let messages = vec![
            Message {
                role: crate::traits::MessageRole::System,
                content:
                    "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
                        .to_string(),
            },
            Message {
                role: crate::traits::MessageRole::User,
                content: prompt,
            },
        ];

        let response = self
            .llm
            .chat_completion_with_json(&messages, DEFAULT_TEMPERATURE)
            .await?;
        let entries = Self::parse_extraction_response(&response)?;

        Ok(entries)
    }

    fn format_previous_context(entries: &[MemoryEntry]) -> String {
        if entries.is_empty() {
            String::new()
        } else {
            let context_lines: Vec<String> = entries
                .iter()
                .take(3)
                .map(|e| format!("- {}", e.lossless_restatement))
                .collect();
            format!(
                "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n{}\n",
                context_lines.join("\n")
            )
        }
    }

    fn format_conversation(dialogues: &[&Dialogue]) -> String {
        dialogues
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn parse_extraction_response(json: &serde_json::Value) -> PipelineResult<Vec<MemoryEntry>> {
        let memories = json
            .get("memories")
            .ok_or_else(|| {
                PipelineError::CompressionFailed("No 'memories' field in response".to_string())
            })?
            .as_array()
            .ok_or_else(|| {
                PipelineError::CompressionFailed("'memories' is not an array".to_string())
            })?;

        let mut entries = Vec::new();
        for mem in memories {
            let lossless_restatement = mem
                .get("lossless_restatement")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    PipelineError::CompressionFailed("Missing lossless_restatement".to_string())
                })?
                .to_string();

            let mut entry = MemoryEntry::new(lossless_restatement);

            if let Some(keywords) = mem.get("keywords").and_then(|v| v.as_array()) {
                entry.keywords = keywords
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }

            if let Some(persons) = mem.get("persons").and_then(|v| v.as_array()) {
                entry.persons = persons
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .filter(|s| !s.is_empty())
                    .collect();
            }

            if let Some(entities) = mem.get("entities").and_then(|v| v.as_array()) {
                entry.entities = entities
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .filter(|s| !s.is_empty())
                    .collect();
            }

            if let Some(location) = mem.get("location").and_then(|v| v.as_str())
                && !location.is_empty()
                && location != "null"
            {
                entry.location = Some(location.to_string());
            }

            if let Some(topic) = mem.get("topic").and_then(|v| v.as_str())
                && !topic.is_empty()
                && topic != "null"
            {
                entry.topic = Some(topic.to_string());
            }

            if let Some(ts_str) = mem.get("timestamp").and_then(|v| v.as_str())
                && !ts_str.is_empty()
                && ts_str != "null"
            {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(ts_str) {
                    entry.timestamp = Some(ts.with_timezone(&chrono::Utc));
                } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%dT%H:%M:%S")
                {
                    entry.timestamp = Some(chrono::DateTime::from_naive_utc_and_offset(dt, chrono::Utc));
                } else if let Ok(d) = chrono::NaiveDate::parse_from_str(ts_str, "%Y-%m-%d") {
                    entry.timestamp = Some(chrono::DateTime::from_naive_utc_and_offset(
                        d.and_hms_opt(0, 0, 0).unwrap(),
                        chrono::Utc,
                    ));
                }
            }

            entries.push(entry);
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetrievalConfig;
    use crate::config::SynthesizerConfig;
    use crate::mocks::MockLlmClient;
    use std::sync::Arc;

    fn test_config() -> PipelineConfig {
        PipelineConfig {
            window_size: 4,
            overlap_size: 1,
            retrieval: RetrievalConfig::default(),
            synthesizer: SynthesizerConfig::default(),
        }
    }

    fn make_dialogue(speaker: &str, content: &str) -> Dialogue {
        Dialogue::new(speaker, content)
    }

    #[tokio::test]
    async fn test_compress_empty_dialogues() {
        let llm = Arc::new(MockLlmClient::new());
        let compressor = Compressor::new(llm, test_config());

        let result = compressor.compress_dialogues(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_compress_single_window() {
        let response = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "User likes pizza",
                    "keywords": ["pizza", "food"],
                    "persons": [],
                    "entities": [],
                    "location": null,
                    "topic": "food preferences"
                }
            ]
        });

        let llm = Arc::new(MockLlmClient::with_responses(vec![response]));
        let compressor = Compressor::new(llm, test_config());

        let dialogues = vec![
            make_dialogue("User", "I love pizza"),
            make_dialogue("Assistant", "That's great!"),
        ];

        let result = compressor.compress_dialogues(&dialogues).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lossless_restatement, "User likes pizza");
    }

    #[tokio::test]
    async fn test_compress_multiple_memories() {
        let response = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "User works at Google",
                    "keywords": ["google", "work"],
                    "persons": [],
                    "entities": ["Google"],
                    "location": null,
                    "topic": "work"
                },
                {
                    "lossless_restatement": "User lives in Jakarta",
                    "keywords": ["jakarta", "home"],
                    "persons": [],
                    "entities": [],
                    "location": "Jakarta",
                    "topic": "location"
                }
            ]
        });

        let llm = Arc::new(MockLlmClient::with_responses(vec![response]));
        let compressor = Compressor::new(llm, test_config());

        let dialogues = vec![make_dialogue(
            "User",
            "I work at Google and live in Jakarta",
        )];

        let result = compressor.compress_dialogues(&dialogues).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_compress_with_persons() {
        let response = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "User met Alice at the conference",
                    "keywords": ["conference", "meeting"],
                    "persons": ["Alice"],
                    "entities": [],
                    "location": null,
                    "topic": "networking"
                }
            ]
        });

        let llm = Arc::new(MockLlmClient::with_responses(vec![response]));
        let compressor = Compressor::new(llm, test_config());

        let dialogues = vec![make_dialogue("User", "I met Alice at the conference")];

        let result = compressor.compress_dialogues(&dialogues).await.unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].persons.contains(&"Alice".to_string()));
    }

    #[tokio::test]
    async fn test_create_windows_empty() {
        let llm = Arc::new(MockLlmClient::new());
        let compressor = Compressor::new(llm, test_config());

        let windows = compressor.create_windows(&[]);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_create_windows_single_window() {
        let llm = Arc::new(MockLlmClient::new());
        let config = PipelineConfig {
            window_size: 10,
            overlap_size: 2,
            ..test_config()
        };
        let compressor = Compressor::new(llm, config);

        let dialogues: Vec<Dialogue> = (0..5)
            .map(|i| make_dialogue("User", &format!("Message {i}")))
            .collect();

        let windows = compressor.create_windows(&dialogues);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len(), 5);
    }

    #[test]
    fn test_create_windows_multiple_windows() {
        let llm = Arc::new(MockLlmClient::new());
        let config = PipelineConfig {
            window_size: 4,
            overlap_size: 1,
            ..test_config()
        };
        let compressor = Compressor::new(llm, config);

        let dialogues: Vec<Dialogue> = (0..10)
            .map(|i| make_dialogue("User", &format!("Message {i}")))
            .collect();

        let windows = compressor.create_windows(&dialogues);
        assert!(windows.len() > 1);
    }

    #[test]
    fn test_create_windows_with_overlap() {
        let llm = Arc::new(MockLlmClient::new());
        let config = PipelineConfig {
            window_size: 4,
            overlap_size: 2,
            ..test_config()
        };
        let compressor = Compressor::new(llm, config);

        let dialogues: Vec<Dialogue> = (0..8)
            .map(|i| make_dialogue("User", &format!("Message {i}")))
            .collect();

        let windows = compressor.create_windows(&dialogues);

        if windows.len() > 1 {
            let last_of_first = windows[0].last().unwrap().content.clone();
            let first_of_second = windows[1].first().unwrap().content.clone();

            assert!(
                windows[0].iter().any(|d| d.content == first_of_second)
                    || windows[1].iter().any(|d| d.content == last_of_first)
            );
        }
    }

    #[tokio::test]
    async fn test_parse_extraction_response_valid() {
        let json = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "Test memory",
                    "keywords": ["test"],
                    "persons": ["Alice"],
                    "entities": ["ProjectX"],
                    "location": "Jakarta",
                    "topic": "testing"
                }
            ]
        });

        let entries = Compressor::parse_extraction_response(&json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lossless_restatement, "Test memory");
        assert!(entries[0].keywords.contains(&"test".to_string()));
        assert!(entries[0].persons.contains(&"Alice".to_string()));
        assert!(entries[0].entities.contains(&"ProjectX".to_string()));
        assert_eq!(entries[0].location, Some("Jakarta".to_string()));
        assert_eq!(entries[0].topic, Some("testing".to_string()));
    }

    #[tokio::test]
    async fn test_parse_extraction_response_missing_memories() {
        let json = serde_json::json!({
            "other_field": "value"
        });

        let result = Compressor::parse_extraction_response(&json);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_extraction_response_memories_not_array() {
        let json = serde_json::json!({
            "memories": "not an array"
        });

        let result = Compressor::parse_extraction_response(&json);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_extraction_response_missing_lossless_restatement() {
        let json = serde_json::json!({
            "memories": [
                {
                    "keywords": ["test"]
                }
            ]
        });

        let result = Compressor::parse_extraction_response(&json);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parse_extraction_response_empty_arrays() {
        let json = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "Minimal memory",
                    "keywords": [],
                    "persons": [],
                    "entities": [],
                    "location": null,
                    "topic": null
                }
            ]
        });

        let entries = Compressor::parse_extraction_response(&json).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].keywords.is_empty());
        assert!(entries[0].persons.is_empty());
        assert!(entries[0].entities.is_empty());
        assert!(entries[0].location.is_none());
        assert!(entries[0].topic.is_none());
    }

    #[tokio::test]
    async fn test_parse_extraction_response_null_strings() {
        let json = serde_json::json!({
            "memories": [
                {
                    "lossless_restatement": "Memory with null string",
                    "keywords": [],
                    "persons": [],
                    "entities": [],
                    "location": "null",
                    "topic": "null"
                }
            ]
        });

        let entries = Compressor::parse_extraction_response(&json).unwrap();
        assert!(entries[0].location.is_none());
        assert!(entries[0].topic.is_none());
    }

    #[test]
    fn test_format_conversation() {
        let d1 = make_dialogue("User", "Hello");
        let d2 = make_dialogue("Assistant", "Hi there");
        let dialogues = vec![&d1, &d2];

        let formatted = Compressor::format_conversation(&dialogues);
        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant: Hi there"));
    }
}
