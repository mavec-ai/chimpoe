use crate::chimpoe::MemoryHit;
use crate::error::LlmResult;
use crate::traits::{LlmClient, Message, MessageRole};
use std::sync::Arc;

const MAX_RETRIES: usize = 3;
const DEFAULT_TEMPERATURE: f32 = 0.1;

pub struct AnswerGenerator {
    llm: Arc<dyn LlmClient>,
}

impl AnswerGenerator {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    pub async fn generate(&self, query: &str, contexts: &[MemoryHit]) -> LlmResult<String> {
        if contexts.is_empty() {
            return Ok("No relevant information found.".to_string());
        }

        let context_str = Self::format_contexts(contexts);
        let prompt = Self::build_prompt(query, &context_str);

        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are a professional Q&A assistant. Extract concise answers from context. CRITICAL: You MUST answer in the SAME LANGUAGE as the user's question. You must output valid JSON format.".to_string(),
            },
            Message {
                role: MessageRole::User,
                content: prompt,
            },
        ];

        let mut last_raw_response: Option<String> = None;

        for _attempt in 1..=MAX_RETRIES {
            match self
                .llm
                .chat_completion_with_json(&messages, DEFAULT_TEMPERATURE)
                .await
            {
                Ok(response) => {
                    if let Some(answer) = Self::extract_answer(&response) {
                        return Ok(answer);
                    }
                    last_raw_response = Some(response.to_string());
                }
                Err(e) => {
                    tracing::warn!("JSON completion failed: {e}, falling back to plain completion");
                    if let Ok(raw) = self
                        .llm
                        .chat_completion(&messages, DEFAULT_TEMPERATURE)
                        .await
                    {
                        last_raw_response = Some(raw);
                    }
                }
            }
        }

        if let Some(raw) = last_raw_response {
            if let Some(answer) = Self::try_extract_json_from_text(&raw) {
                return Ok(answer);
            }
            tracing::warn!("Failed to extract JSON from LLM response, returning raw text");
            return Ok(raw);
        }

        Ok("Failed to generate answer.".to_string())
    }

    fn try_extract_json_from_text(text: &str) -> Option<String> {
        let text = text.trim();

        if text.starts_with('{')
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(text)
        {
            return Self::extract_answer(&json);
        }

        if let Some(start) = text.find("```json") {
            let start = start + 7;
            if let Some(end) = text[start..].find("```") {
                let json_str = text[start..start + end].trim();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return Self::extract_answer(&json);
                }
            }
        }

        if let Some(start) = text.find("```") {
            let start = start + 3;
            let newline_pos = text[start..].find('\n').unwrap_or(0);
            let start = start + newline_pos + 1;
            if let Some(end) = text[start..].find("```") {
                let json_str = text[start..start + end].trim();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                    return Self::extract_answer(&json);
                }
            }
        }

        None
    }

    fn extract_answer(response: &serde_json::Value) -> Option<String> {
        response
            .as_object()
            .and_then(|obj| obj.get("answer"))
            .and_then(|a| a.as_str())
            .map(std::string::ToString::to_string)
    }

    fn format_contexts(contexts: &[MemoryHit]) -> String {
        contexts
            .iter()
            .enumerate()
            .map(|(i, hit)| {
                let mut parts = vec![format!("[Context {}]", i + 1)];
                parts.push(format!("Content: {}", hit.memory));

                if let Some(ref ts) = hit.timestamp {
                    parts.push(format!("Time: {ts}"));
                }
                if let Some(ref loc) = hit.location {
                    parts.push(format!("Location: {loc}"));
                }
                if !hit.persons.is_empty() {
                    parts.push(format!("Persons: {}", hit.persons.join(", ")));
                }
                if !hit.entities.is_empty() {
                    parts.push(format!("Related Entities: {}", hit.entities.join(", ")));
                }
                if let Some(ref topic) = hit.topic {
                    parts.push(format!("Topic: {topic}"));
                }

                parts.join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_prompt(question: &str, context: &str) -> String {
        format!(
            r#"Answer the user's question based on the provided context.

User Question: {question}

Relevant Context:
{context}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. CRITICAL: You MUST write the answer in the SAME LANGUAGE as the user's question. If the question is in Indonesian, answer in Indonesian. If in English, answer in English. Always match the question's language.
6. If the context does not contain information to answer the question, say so in the question's language (e.g., "Tidak ada informasi tentang..." for Indonesian, "No information about..." for English)
7. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text."#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_answer() {
        let json = serde_json::json!({"answer": "Test answer"});
        assert_eq!(
            AnswerGenerator::extract_answer(&json),
            Some("Test answer".to_string())
        );

        let json_with_reasoning = serde_json::json!({
            "reasoning": "Some reasoning",
            "answer": "Final answer"
        });
        assert_eq!(
            AnswerGenerator::extract_answer(&json_with_reasoning),
            Some("Final answer".to_string())
        );

        let json_no_answer = serde_json::json!({"reasoning": "Only reasoning"});
        assert_eq!(AnswerGenerator::extract_answer(&json_no_answer), None);

        let json_null_answer = serde_json::json!({"answer": null});
        assert_eq!(AnswerGenerator::extract_answer(&json_null_answer), None);
    }

    #[test]
    fn test_try_extract_json_from_text_direct_json() {
        let text = r#"{"answer": "Direct answer"}"#;
        assert_eq!(
            AnswerGenerator::try_extract_json_from_text(text),
            Some("Direct answer".to_string())
        );
    }

    #[test]
    fn test_try_extract_json_from_text_json_block() {
        let text = r#"Here is the response:
```json
{"answer": "Block answer"}
```
Some extra text"#;
        assert_eq!(
            AnswerGenerator::try_extract_json_from_text(text),
            Some("Block answer".to_string())
        );
    }

    #[test]
    fn test_try_extract_json_from_text_code_block() {
        let text = r#"Response:
```
{"answer": "Code block answer"}
```"#;
        assert_eq!(
            AnswerGenerator::try_extract_json_from_text(text),
            Some("Code block answer".to_string())
        );
    }

    #[test]
    fn test_try_extract_json_from_text_with_whitespace() {
        let text = r#"
```json
{
  "reasoning": "Complex reasoning",
  "answer": "Whitespace answer"
}
```
"#;
        assert_eq!(
            AnswerGenerator::try_extract_json_from_text(text),
            Some("Whitespace answer".to_string())
        );
    }

    #[test]
    fn test_try_extract_json_from_text_invalid_json() {
        let text = "This is not JSON at all";
        assert_eq!(AnswerGenerator::try_extract_json_from_text(text), None);
    }

    #[test]
    fn test_try_extract_json_from_text_empty_answer() {
        let text = r#"{"reasoning": "No answer field"}"#;
        assert_eq!(AnswerGenerator::try_extract_json_from_text(text), None);
    }

    #[test]
    fn test_format_contexts_basic() {
        let contexts = vec![MemoryHit {
            memory: "Alice likes pizza".to_string(),
            timestamp: None,
            location: None,
            persons: vec![],
            entities: vec![],
            topic: None,
            source: "test".to_string(),
        }];

        let result = AnswerGenerator::format_contexts(&contexts);
        assert!(result.contains("[Context 1]"));
        assert!(result.contains("Content: Alice likes pizza"));
    }

    #[test]
    fn test_format_contexts_with_all_fields() {
        let contexts = vec![MemoryHit {
            memory: "Meeting scheduled".to_string(),
            timestamp: Some("2025-11-16T14:00:00".to_string()),
            location: Some("Office".to_string()),
            persons: vec!["Alice".to_string(), "Bob".to_string()],
            entities: vec!["Project X".to_string()],
            topic: Some("Planning".to_string()),
            source: "test".to_string(),
        }];

        let result = AnswerGenerator::format_contexts(&contexts);
        assert!(result.contains("[Context 1]"));
        assert!(result.contains("Content: Meeting scheduled"));
        assert!(result.contains("Time: 2025-11-16T14:00:00"));
        assert!(result.contains("Location: Office"));
        assert!(result.contains("Persons: Alice, Bob"));
        assert!(result.contains("Related Entities: Project X"));
        assert!(result.contains("Topic: Planning"));
    }

    #[test]
    fn test_format_contexts_multiple() {
        let contexts = vec![
            MemoryHit {
                memory: "First memory".to_string(),
                timestamp: None,
                location: None,
                persons: vec![],
                entities: vec![],
                topic: None,
                source: "test".to_string(),
            },
            MemoryHit {
                memory: "Second memory".to_string(),
                timestamp: None,
                location: None,
                persons: vec![],
                entities: vec![],
                topic: None,
                source: "test".to_string(),
            },
        ];

        let result = AnswerGenerator::format_contexts(&contexts);
        assert!(result.contains("[Context 1]"));
        assert!(result.contains("[Context 2]"));
        assert!(result.contains("First memory"));
        assert!(result.contains("Second memory"));
    }

    #[test]
    fn test_build_prompt() {
        let question = "What is the meeting time?";
        let context = "[Context 1]\nContent: Meeting at 2pm";

        let prompt = AnswerGenerator::build_prompt(question, context);

        assert!(prompt.contains("What is the meeting time?"));
        assert!(prompt.contains("[Context 1]"));
        assert!(prompt.contains("Meeting at 2pm"));
        assert!(prompt.contains("reasoning"));
        assert!(prompt.contains("answer"));
    }

    #[tokio::test]
    async fn test_generate_empty_contexts() {
        use crate::mocks::MockLlmClient;
        let llm = Arc::new(MockLlmClient::new());
        let generator = AnswerGenerator::new(llm);

        let result = generator.generate("What is X?", &[]).await.unwrap();
        assert_eq!(result, "No relevant information found.");
    }

    #[tokio::test]
    async fn test_generate_success_on_first_try() {
        use crate::mocks::MockLlmClient;
        let response = serde_json::json!({
            "reasoning": "Based on context",
            "answer": "The meeting is at 3pm"
        });
        let llm = Arc::new(MockLlmClient::with_responses(vec![response]));
        let generator = AnswerGenerator::new(llm);

        let contexts = vec![MemoryHit {
            memory: "Meeting scheduled at 3pm".to_string(),
            persons: vec![],
            entities: vec![],
            location: None,
            topic: None,
            timestamp: None,
            source: "test".to_string(),
        }];

        let result = generator
            .generate("When is the meeting?", &contexts)
            .await
            .unwrap();
        assert_eq!(result, "The meeting is at 3pm");
    }

    #[tokio::test]
    async fn test_generate_fallback_to_raw_text() {
        use crate::mocks::MockLlmClient;
        let llm = Arc::new(MockLlmClient::new());
        let generator = AnswerGenerator::new(llm);

        let contexts = vec![MemoryHit {
            memory: "Some memory".to_string(),
            persons: vec![],
            entities: vec![],
            location: None,
            topic: None,
            timestamp: None,
            source: "test".to_string(),
        }];

        let result = generator.generate("What?", &contexts).await.unwrap();
        assert_eq!(result, "mock response");
    }

    #[tokio::test]
    async fn test_generate_retry_then_fallback_text() {
        use crate::mocks::MockLlmClient;
        let llm = Arc::new(MockLlmClient::with_both(
            vec![serde_json::json!({"reasoning": "no answer field"})],
            vec!["{\"answer\": \"recovered answer\"}".to_string()],
        ));
        let generator = AnswerGenerator::new(llm);

        let contexts = vec![MemoryHit {
            memory: "Some context".to_string(),
            persons: vec![],
            entities: vec![],
            location: None,
            topic: None,
            timestamp: None,
            source: "test".to_string(),
        }];

        let result = generator
            .generate("Tell me something", &contexts)
            .await
            .unwrap();
        assert_eq!(result, "recovered answer");
    }

    #[test]
    fn test_build_prompt_indonesian_language() {
        let question = "Apa waktu meetingnya?";
        let context = "[Context 1]\nContent: Meeting jam 3 siang";

        let prompt = AnswerGenerator::build_prompt(question, context);

        assert!(prompt.contains("Apa waktu meetingnya?"));
        assert!(prompt.contains("Meeting jam 3 siang"));
        assert!(prompt.contains("reasoning"));
        assert!(prompt.contains("answer"));
    }

    #[test]
    fn test_build_prompt_japanese_language() {
        let question = "会議は何時ですか？";
        let context = "[Context 1]\nContent: Meeting is at 3pm";

        let prompt = AnswerGenerator::build_prompt(question, context);

        assert!(prompt.contains("会議は何時ですか？"));
        assert!(prompt.contains("Meeting is at 3pm"));
    }
}
