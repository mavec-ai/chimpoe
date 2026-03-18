use crate::config::PipelineConfig;
use crate::error::{PipelineError, PipelineResult};
use crate::traits::{LlmClient, Message};
use crate::types::{Dialogue, MemoryEntry};
use std::sync::Arc;

const EXTRACTION_PROMPT: &str = r#"
Analyze the following conversation and extract key memories.

Conversation:
{conversation}

Extract memories in JSON format:
```json
{{
  "memories": [
    {{
      "lossless_restatement": "A neutral, factual restatement of the key information",
      "keywords": ["keyword1", "keyword2"],
      "persons": ["person name if mentioned"],
      "entities": ["organization/product if mentioned"],
      "location": "location if mentioned or null",
      "topic": "main topic or null"
    }}
  ]
}}
```

Requirements:
1. Extract factual information, preferences, and important context
2. lossless_restatement should be a complete, standalone statement
3. Include only information explicitly stated
4. Return ONLY valid JSON, no markdown or explanation
"#;

pub struct Compressor {
    llm: Arc<dyn LlmClient>,
    config: PipelineConfig,
    temperature: f32,
}

impl Compressor {
    pub fn new(llm: Arc<dyn LlmClient>, config: PipelineConfig, temperature: f32) -> Self {
        Self {
            llm,
            config,
            temperature,
        }
    }

    pub async fn compress_dialogues(
        &self,
        dialogues: &[Dialogue],
    ) -> PipelineResult<Vec<MemoryEntry>> {
        let windows = self.create_windows(dialogues);
        let mut all_entries = Vec::new();

        for window in windows {
            let entries = self.process_window(&window).await?;
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

    async fn process_window(&self, window: &[&Dialogue]) -> PipelineResult<Vec<MemoryEntry>> {
        let conversation = self.format_conversation(window);
        let prompt = EXTRACTION_PROMPT.replace("{conversation}", &conversation);

        let messages = vec![
            Message {
                role: crate::traits::MessageRole::System,
                content:
                    "You are a memory extraction assistant. You must output valid JSON format."
                        .to_string(),
            },
            Message {
                role: crate::traits::MessageRole::User,
                content: prompt,
            },
        ];

        let response = self
            .llm
            .chat_completion_with_json(&messages, self.temperature)
            .await?;
        let entries = self.parse_extraction_response(&response)?;

        Ok(entries)
    }

    fn format_conversation(&self, dialogues: &[&Dialogue]) -> String {
        dialogues
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn parse_extraction_response(
        &self,
        json: &serde_json::Value,
    ) -> PipelineResult<Vec<MemoryEntry>> {
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

            entries.push(entry);
        }

        Ok(entries)
    }
}
