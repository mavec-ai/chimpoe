import type { AgentTier } from "@chimpoe/types";
import { adjustBudget, getAgent, updateAgentStatus } from "../state/agents.ts";
import { withDb } from "../state/db.ts";

export interface ModelPrice {
  inputPerToken: number;
  outputPerToken: number;
}

const DEFAULT_PRICE: ModelPrice = {
  inputPerToken: 1,
  outputPerToken: 3,
};

const PRICE_OVERRIDES: Record<string, ModelPrice> = {
  "gpt-5": { inputPerToken: 5, outputPerToken: 15 },
  "gpt-5-mini": { inputPerToken: 1, outputPerToken: 3 },
  "claude-opus-4": { inputPerToken: 15, outputPerToken: 75 },
  "claude-sonnet-4": { inputPerToken: 3, outputPerToken: 15 },
  "claude-haiku-4-5": { inputPerToken: 1, outputPerToken: 5 },
  "llama3.2": { inputPerToken: 0, outputPerToken: 0 },
};

export function getModelPrice(modelId: string): ModelPrice {
  const lower = modelId.toLowerCase();
  if (PRICE_OVERRIDES[lower]) return PRICE_OVERRIDES[lower];
  for (const [key, price] of Object.entries(PRICE_OVERRIDES)) {
    if (lower.startsWith(key.toLowerCase() + "-") || lower.startsWith(key.toLowerCase() + ".")) {
      return price;
    }
  }
  return DEFAULT_PRICE;
}

export interface ChargeInput {
  agentId: string;
  modelId: string;
  inputTokens: number;
  outputTokens: number;
}

export interface ChargeResult {
  costTokens: number;
  newBalance: number;
  newTier: AgentTier;
  tierChanged: boolean;
}

export async function chargeInference(input: ChargeInput): Promise<ChargeResult> {
  const price = getModelPrice(input.modelId);
  const cost = Math.ceil(
    input.inputTokens * price.inputPerToken + input.outputTokens * price.outputPerToken,
  );
  if (cost <= 0) {
    const agent = await getAgent(input.agentId);
    return {
      costTokens: 0,
      newBalance: agent?.budgetTokens ?? 0,
      newTier: agent?.tier ?? "normal",
      tierChanged: false,
    };
  }
  const newBalance = await adjustBudget(input.agentId, -cost);
  const tierResult = await applyTierTransition(input.agentId, newBalance);
  return {
    costTokens: cost,
    newBalance,
    newTier: tierResult.tier,
    tierChanged: tierResult.changed,
  };
}

export const TIER_THRESHOLDS = {
  thriving: 50_000,
  normal: 5_000,
  conservation: 500,
  dormant: 1,
} as const;

export function calculateTier(budgetTokens: number): AgentTier {
  if (budgetTokens >= TIER_THRESHOLDS.thriving) return "thriving";
  if (budgetTokens >= TIER_THRESHOLDS.normal) return "normal";
  if (budgetTokens >= TIER_THRESHOLDS.conservation) return "conservation";
  if (budgetTokens >= TIER_THRESHOLDS.dormant) return "dormant";
  return "dead";
}

export interface TierTransitionResult {
  tier: AgentTier;
  changed: boolean;
  previous: AgentTier | null;
}

export async function applyTierTransition(
  agentId: string,
  currentBudget: number,
): Promise<TierTransitionResult> {
  const newTier = calculateTier(currentBudget);
  const agent = await getAgent(agentId);
  if (!agent) return { tier: newTier, changed: false, previous: null };
  if (agent.tier === newTier) return { tier: newTier, changed: false, previous: agent.tier };

  const status = newTier === "dead" ? "dead" : newTier === "dormant" ? "sleeping" : agent.status;
  await updateAgentStatus(agentId, status, newTier);
  return { tier: newTier, changed: true, previous: agent.tier };
}

export interface BudgetSnapshot {
  balance: number;
  tier: AgentTier;
  projectedInputsAt: { thriving: number; normal: number; conservation: number; dormant: number };
}

export async function getBudgetSnapshot(agentId: string): Promise<BudgetSnapshot | null> {
  const agent = await getAgent(agentId);
  if (!agent) return null;
  const price = getModelPrice(agent.modelId);
  const perInput = price.inputPerToken + price.outputPerToken * 2;
  const safePerInput = perInput > 0 ? perInput : 1;
  return {
    balance: agent.budgetTokens,
    tier: agent.tier,
    projectedInputsAt: {
      thriving: Math.floor(agent.budgetTokens / (TIER_THRESHOLDS.thriving * safePerInput)),
      normal: Math.floor(agent.budgetTokens / (TIER_THRESHOLDS.normal * safePerInput)),
      conservation: Math.floor(agent.budgetTokens / (TIER_THRESHOLDS.conservation * safePerInput)),
      dormant: Math.floor(agent.budgetTokens / (TIER_THRESHOLDS.dormant * safePerInput)),
    },
  };
}

export async function fundAgent(agentId: string, amountTokens: number): Promise<number> {
  return adjustBudget(agentId, Math.max(0, Math.floor(amountTokens)));
}

export async function getAgentBudgets(): Promise<
  Array<{ id: string; name: string; balance: number; tier: AgentTier }>
> {
  return withDb((db) => {
    const rows = db
      .prepare(
        "SELECT id, name, budget_tokens, tier FROM agents WHERE id != 'user' ORDER BY budget_tokens DESC",
      )
      .all() as Array<{ id: string; name: string; budget_tokens: number; tier: string }>;
    return rows.map((r) => ({
      id: r.id,
      name: r.name,
      balance: r.budget_tokens,
      tier: r.tier as AgentTier,
    }));
  });
}
