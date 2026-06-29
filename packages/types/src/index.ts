import { homedir } from "node:os";
import { join } from "node:path";

export const CHIMPOE_HOME_ENV = "CHIMPOE_HOME";
export const DEFAULT_CHIMPOE_HOME_NAME = ".chimpoe";
export const CHIMPOE_VERSION = "0.1.0";

export function getChimpoeHome(): string {
  const env = process.env[CHIMPOE_HOME_ENV];
  if (env && env.trim().length > 0) return env;
  return join(homedir(), DEFAULT_CHIMPOE_HOME_NAME);
}

export function getAgentsDir(): string {
  return join(getChimpoeHome(), "agents");
}

export function getAgentHome(agentId: string): string {
  return join(getAgentsDir(), agentId);
}

export function getFossilsDir(): string {
  return join(getChimpoeHome(), "fossils");
}

export function getSharedStateDbPath(): string {
  return join(getChimpoeHome(), "state.db");
}

export function getConfigPath(): string {
  return join(getChimpoeHome(), "chimpoe.json");
}

export function getEnvPath(): string {
  return join(getChimpoeHome(), ".env");
}

export type Provider = "openai" | "anthropic" | "google" | "xai" | "groq" | "ollama";

export interface ChimpoeConfig {
  version: string;
  defaultProvider: Provider;
  defaultModel: string;
  createdAt: number;
}

export type AgentTier = "thriving" | "normal" | "conservation" | "dormant" | "dead";
export type AgentStatus = "running" | "idle" | "sleeping" | "dead";

export interface AgentConfig {
  id: string;
  name: string;
  provider: Provider;
  modelId: string;
  genesisPrompt: string;
  parentId: string | null;
  generation: number;
  budgetTokens: number;
  tier: AgentTier;
  status: AgentStatus;
  createdAt: number;
}

export type AgentRole = "root" | "leaf" | "orchestrator";

export type TurnStatus = "complete" | "failed" | "interrupted";

export interface ToolCallRecord {
  name: string;
  input: unknown;
  output: unknown;
  durationMs?: number;
  error?: string;
}

export interface TurnRecord {
  id: string;
  agentId: string;
  sessionId: string;
  userMessage: string;
  assistantText: string | null;
  toolCalls: ToolCallRecord[];
  inputTokens: number | null;
  outputTokens: number | null;
  durationMs: number | null;
  status: TurnStatus;
  createdAt: number;
}

export interface SessionInfo {
  id: string;
  agentId: string;
  startedAt: number;
  endedAt: number | null;
  turnCount: number;
}
