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
    throw new Error(
      `No API key for provider "${provider}". Set ${envKey} env var or run "chimpoe init".`,
    );
  }

  switch (provider) {
    case "openai": {
      const openai = createOpenAI({ apiKey });
      return { model: openai(modelId), provider, modelId };
    }
    case "anthropic": {
      const anthropic = createAnthropic({ apiKey });
      return { model: anthropic(modelId), provider, modelId };
    }
    default:
      throw new Error(`Provider "${provider}" not yet wired up. Open an issue.`);
  }
}
