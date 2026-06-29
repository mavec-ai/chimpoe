import { randomUUID } from "node:crypto";
import type { AgentConfig, AgentStatus, AgentTier, Provider } from "@chimpoe/types";
import { withDb } from "./db.ts";

interface AgentRow {
  id: string;
  name: string;
  provider: string;
  model_id: string;
  genesis_prompt: string;
  parent_id: string | null;
  generation: number;
  budget_tokens: number;
  tier: string;
  status: string;
  created_at: number;
  updated_at: number;
}

function rowToConfig(row: AgentRow): AgentConfig {
  return {
    id: row.id,
    name: row.name,
    provider: row.provider as Provider,
    modelId: row.model_id,
    genesisPrompt: row.genesis_prompt,
    parentId: row.parent_id,
    generation: row.generation,
    budgetTokens: row.budget_tokens,
    tier: row.tier as AgentTier,
    status: row.status as AgentStatus,
    createdAt: row.created_at,
  };
}

export interface CreateAgentInput {
  name: string;
  provider: Provider;
  modelId: string;
  genesisPrompt: string;
  parentId?: string | null;
  generation?: number;
}

export async function registerAgent(input: CreateAgentInput): Promise<AgentConfig> {
  const now = Date.now();
  const id = randomUUID();
  const config: AgentConfig = {
    id,
    name: input.name,
    provider: input.provider,
    modelId: input.modelId,
    genesisPrompt: input.genesisPrompt,
    parentId: input.parentId ?? null,
    generation: input.generation ?? 0,
    budgetTokens: 0,
    tier: "normal",
    status: "idle",
    createdAt: now,
  };
  await withDb((db) => {
    db.prepare(
      `INSERT INTO agents (id, name, provider, model_id, genesis_prompt, parent_id, generation, budget_tokens, tier, status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      config.id,
      config.name,
      config.provider,
      config.modelId,
      config.genesisPrompt,
      config.parentId,
      config.generation,
      config.budgetTokens,
      config.tier,
      config.status,
      now,
      now,
    );
  });
  return config;
}

export async function getAgent(id: string): Promise<AgentConfig | null> {
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM agents WHERE id = ?").get(id) as AgentRow | null;
    return row ? rowToConfig(row) : null;
  });
}

export async function listAgents(): Promise<AgentConfig[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM agents WHERE id != 'user' ORDER BY created_at DESC")
      .all() as AgentRow[];
    return rows.map(rowToConfig);
  });
}

export async function updateAgentStatus(
  id: string,
  status: AgentStatus,
  tier?: AgentTier,
): Promise<void> {
  await withDb((db) => {
    if (tier !== undefined) {
      db.prepare("UPDATE agents SET status = ?, tier = ?, updated_at = ? WHERE id = ?").run(
        status,
        tier,
        Date.now(),
        id,
      );
    } else {
      db.prepare("UPDATE agents SET status = ?, updated_at = ? WHERE id = ?").run(
        status,
        Date.now(),
        id,
      );
    }
  });
}

export async function deleteAgent(id: string): Promise<void> {
  await withDb((db) => {
    db.prepare("DELETE FROM agents WHERE id = ?").run(id);
  });
}
