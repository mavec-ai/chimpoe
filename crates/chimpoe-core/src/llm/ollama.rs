use crate::config::{LlmConfig, OLLAMA_LLM_BASE_URL};
use crate::error::{LlmError, LlmResult};
use crate::traits::{LlmClient, Message, MessageRole};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaLlm {
    client: Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

impl OllamaLlm {
    #[must_use]
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            client: Client::new(),
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| OLLAMA_LLM_BASE_URL.to_string()),
            model: config.model.clone(),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    fn convert_messages(messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|m| ChatMessage {
                role: match m.role {
                    MessageRole::System => "system".to_string(),
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
            })
            .collect()
    }

    fn extract_json(text: &str) -> LlmResult<serde_json::Value> {
        let text = text.trim();

        let extract_and_parse = |json_str: &str| -> LlmResult<serde_json::Value> {
            let cleaned = json_str
                .replace(":=", ":")
                .replace(",]", "]")
                .replace(",}", "}");
            serde_json::from_str(&cleaned)
                .map_err(|e| LlmError::JsonExtractionFailed(e.to_string()))
        };

        if text.starts_with('{') || text.starts_with('[') {
            return extract_and_parse(text);
        }

        if let Some(start) = text.find("```json") {
            let start = start + 7;
            if let Some(end) = text[start..].find("```") {
                return extract_and_parse(text[start..start + end].trim());
            }
        }

        if let Some(start) = text.find("```") {
            let start = start + 3;
            let newline_pos = text[start..].find('\n').unwrap_or(0);
            let start = start + newline_pos + 1;
            if let Some(end) = text[start..].find("```") {
                return extract_and_parse(text[start..start + end].trim());
            }
        }

        extract_and_parse(text)
    }
}

#[async_trait]
impl LlmClient for OllamaLlm {
    async fn chat_completion(&self, messages: &[Message], temperature: f32) -> LlmResult<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            temperature,
            format: Some(serde_json::json!("json")),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("HTTP {status}: {body}")));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))
    }

    async fn chat_completion_with_json(
        &self,
        messages: &[Message],
        temperature: f32,
    ) -> LlmResult<serde_json::Value> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            temperature,
            format: Some(serde_json::json!({"type": "object"})),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(format!("HTTP {status}: {body}")));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))?;

        Self::extract_json(&content)
    }
}
