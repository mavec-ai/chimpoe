export interface ExperimentConfig {
  name: string;
  preset?: string;
  description?: string;
  durationMs?: number;
  agents: AgentSpec[];
  taskPool?: TaskSpec[];
  metrics?: string[];
  policy?: {
    autoCullReputationUnder?: number;
    autoFundOnConservation?: boolean;
    blockSelfSpawn?: boolean;
  };
}

export interface AgentSpec {
  name: string;
  genesis: string;
  endowment?: number;
  provider?: string;
  model?: string;
  role?: "root" | "leaf" | "orchestrator";
}

export interface TaskSpec {
  prompt: string;
  reward?: number;
  assignTo?: string;
  delayMs?: number;
}

export function validateConfig(input: unknown): ExperimentConfig {
  if (typeof input !== "object" || input === null) {
    throw new Error("Config must be an object");
  }
  const cfg = input as Record<string, unknown>;
  if (typeof cfg.name !== "string" || cfg.name.trim().length === 0) {
    throw new Error("Config must have a non-empty 'name'");
  }
  if (!Array.isArray(cfg.agents) || cfg.agents.length === 0) {
    throw new Error("Config must have a non-empty 'agents' array");
  }
  for (let i = 0; i < cfg.agents.length; i++) {
    const a = cfg.agents[i] as Record<string, unknown>;
    if (typeof a.name !== "string" || typeof a.genesis !== "string") {
      throw new Error(`agents[${i}] must have name (string) and genesis (string)`);
    }
  }
  if (cfg.taskPool !== undefined && !Array.isArray(cfg.taskPool)) {
    throw new Error("'taskPool' must be an array if provided");
  }
  if (cfg.durationMs !== undefined && typeof cfg.durationMs !== "number") {
    throw new Error("'durationMs' must be a number");
  }
  return cfg as unknown as ExperimentConfig;
}

import { parse as parseYamlLib } from "yaml";

export function parseConfigYaml(text: string): ExperimentConfig {
  const trimmed = text.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const json = JSON.parse(trimmed);
      return validateConfig(json);
    } catch {
      // fall through to YAML
    }
  }
  const parsed = parseYamlLib(trimmed);
  return validateConfig(parsed);
}
