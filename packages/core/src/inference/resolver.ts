import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import type { LanguageModel } from "ai";
import type { Provider } from "@chimpoe/types";

export interface ResolvedModel {
  model: LanguageModel;
  provider: Provider;
  modelId: string;
}

const PROVIDER_ENV: Record<Provider, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
  xai: "XAI_API_KEY",
  groq: "GROQ_API_KEY",
  ollama: "OLLAMA_BASE_URL",
  glm: "ZAI_API_KEY",
  "glm-coding": "ZAI_API_KEY",
};

const PROVIDER_BASE_URL: Partial<Record<Provider, string>> = {
  glm: "https://api.z.ai/api/paas/v4/",
  "glm-coding": "https://api.z.ai/api/coding/paas/v4",
};

const PROVIDER_DEFAULT_MODEL: Partial<Record<Provider, string>> = {
  glm: "glm-4.6",
  "glm-coding": "glm-4.6",
};

export function resolveModel(
  provider: Provider,
  modelId: string,
  options: { apiKey?: string; baseUrl?: string } = {},
): ResolvedModel {
  const envKey = PROVIDER_ENV[provider];
  const apiKey = options.apiKey ?? process.env[envKey];

  if (provider === "ollama") {
    const openai = createOpenAI({
      baseURL: options.baseUrl ?? process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1",
      apiKey: "ollama",
    });
    return { model: openai(modelId), provider, modelId };
  }

  if (!apiKey) {
    const hint = PROVIDER_DEFAULT_MODEL[provider]
      ? ` (default model: ${PROVIDER_DEFAULT_MODEL[provider]})`
      : "";
    throw new Error(
      `No API key for provider "${provider}". Set ${envKey} env var or run "chimpoe init".${hint}`,
    );
  }

  switch (provider) {
    case "openai": {
      const openai = createOpenAI({ apiKey, baseURL: options.baseUrl });
      return { model: openai(modelId), provider, modelId };
    }
    case "anthropic": {
      const anthropic = createAnthropic({ apiKey, baseURL: options.baseUrl });
      return { model: anthropic(modelId), provider, modelId };
    }
    case "glm":
    case "glm-coding": {
      const baseURL = options.baseUrl ?? PROVIDER_BASE_URL[provider];
      const openai = createOpenAI({
        apiKey,
        baseURL,
        headers: {
          authorization: apiKey,
          "x-api-key": apiKey,
        },
      });
      return { model: openai.chat(modelId), provider, modelId };
    }
    default:
      throw new Error(
        `Provider "${provider}" not yet wired up. For OpenAI-compatible providers, use "glm" with custom baseURL or open an issue.`,
      );
  }
}

export function listSupportedProviders(): Array<{
  provider: Provider;
  envKey: string;
  defaultModel?: string;
  baseUrl?: string;
}> {
  return (Object.keys(PROVIDER_ENV) as Provider[]).map((p) => ({
    provider: p,
    envKey: PROVIDER_ENV[p]!,
    defaultModel: PROVIDER_DEFAULT_MODEL[p],
    baseUrl: PROVIDER_BASE_URL[p],
  }));
}
