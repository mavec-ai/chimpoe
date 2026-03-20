use crate::config::{LlmConfig, OPENAI_LLM_BASE_URL};
use crate::error::{LlmError, LlmResult};
use crate::traits::{LlmClient, Message, MessageRole};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenAILlm {
    client: Client,
    api_key: String,
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
    response_format: Option<serde_json::Value>,
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

#[derive(Deserialize)]
struct ErrorResponse {
    error: Option<OpenAIError>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

impl OpenAILlm {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            client: Client::new(),
            api_key: config.api_key.clone().unwrap_or_default(),
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| OPENAI_LLM_BASE_URL.to_string()),
            model: config.model.clone(),
        }
    }

    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    fn convert_messages(&self, messages: &[Message]) -> Vec<ChatMessage> {
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

    fn extract_json(&self, text: &str) -> LlmResult<serde_json::Value> {
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
impl LlmClient for OpenAILlm {
    async fn chat_completion(&self, messages: &[Message], temperature: f32) -> LlmResult<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: self.convert_messages(messages),
            stream: false,
            temperature,
            response_format: None,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body)
                && let Some(error) = error_response.error
            {
                if error.error_type.as_deref() == Some("rate_limit_exceeded") {
                    return Err(LlmError::RateLimited);
                }
                return Err(LlmError::ApiError(error.message));
            }
            return Err(LlmError::ApiError(format!("HTTP {}: {}", status, body)));
        }

        let chat_response: ChatResponse =
            serde_json::from_str(&body).map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

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
            messages: self.convert_messages(messages),
            stream: false,
            temperature,
            response_format: Some(serde_json::json!({"type": "json_object"})),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| LlmError::ApiError(e.to_string()))?;

        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body)
                && let Some(error) = error_response.error
            {
                if error.error_type.as_deref() == Some("rate_limit_exceeded") {
                    return Err(LlmError::RateLimited);
                }
                return Err(LlmError::ApiError(error.message));
            }
            return Err(LlmError::ApiError(format!("HTTP {}: {}", status, body)));
        }

        let chat_response: ChatResponse =
            serde_json::from_str(&body).map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))?;

        self.extract_json(&content)
    }
}
