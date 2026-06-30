import { ToolLoopAgent, type Tool } from "ai";
import { resolveModel } from "../inference/resolver.ts";
import { builtinTools } from "../tools/index.ts";
import { readSoul } from "../soul/index.ts";
import { formatMemoryForPrompt, listMemories, type MemoryRecord } from "../memory/index.ts";
import { getActiveSkills, renderSkill, type Skill } from "../skills/index.ts";
import { getModelPrice } from "../economy/budget.ts";
import { adjustBudget } from "../state/agents.ts";
import type { AgentConfig } from "@chimpoe/types";

export interface CreateAgentOptions {
  config: AgentConfig;
  tools?: Record<string, Tool>;
  extraInstructions?: string;
  recentMemoryCount?: number;
}

export interface AgentBudgetTracker {
  remaining: number;
}

export async function createAgent({
  config,
  tools = {},
  extraInstructions,
  recentMemoryCount = 5,
}: CreateAgentOptions): Promise<{ agent: ToolLoopAgent; budget: AgentBudgetTracker }> {
  const resolved = resolveModel(config.provider, config.modelId);

  const [soul, recentMemories, activeSkills] = await Promise.all([
    config.id === "ephemeral" ? Promise.resolve("") : readSoul(config.id).catch(() => ""),
    config.id === "ephemeral"
      ? Promise.resolve([])
      : listMemories(config.id, { limit: recentMemoryCount }).catch(() => []),
    config.id === "ephemeral" ? Promise.resolve([]) : getActiveSkills(config.id).catch(() => []),
  ]);

  const systemPrompt = buildSystemPrompt({
    genesis: config.genesisPrompt,
    soul,
    memories: recentMemories,
    skills: activeSkills,
    extra: extraInstructions,
  });

  const allTools = {
    ...builtinTools(config.id === "ephemeral" ? undefined : config.id),
    ...tools,
  };

  const tier = config.tier ?? "normal";
  const maxOutputTokens = tier === "conservation" || tier === "dormant" ? 2048 : 4096;
  const tierStepLimit = tier === "conservation" ? 5 : tier === "dormant" ? 3 : 12;

  const price = getModelPrice(config.modelId);
  const budget: AgentBudgetTracker = { remaining: config.budgetTokens ?? 0 };

  const isEphemeral = config.id === "ephemeral";
  const estimatedCostPerStep = 3000 * price.inputPerToken + maxOutputTokens * price.outputPerToken;
  const maxStepsByBudget =
    estimatedCostPerStep > 0
      ? Math.max(1, Math.floor(budget.remaining / estimatedCostPerStep))
      : tierStepLimit;
  const effectiveMaxSteps = isEphemeral ? tierStepLimit : Math.min(maxStepsByBudget, tierStepLimit);

  const agent = new ToolLoopAgent({
    model: resolved.model,
    instructions: systemPrompt,
    tools: allTools,
    maxOutputTokens,
    stopWhen: ({ steps }) => steps.length >= effectiveMaxSteps || budget.remaining <= 0,
    onStepEnd: ({ usage }) => {
      if (!usage || isEphemeral) return;
      const cost = Math.ceil(
        (usage.inputTokens ?? 0) * price.inputPerToken +
          (usage.outputTokens ?? 0) * price.outputPerToken,
      );
      if (cost > 0) {
        budget.remaining -= cost;
        void adjustBudget(config.id, -cost).then((newBal) => {
          budget.remaining = newBal;
        });
      }
    },
  });

  return { agent, budget };
}

function buildSystemPrompt(args: {
  genesis: string;
  soul: string;
  memories: MemoryRecord[];
  skills: Skill[];
  extra?: string;
}): string {
  const parts: string[] = [args.genesis.trim()];
  if (args.soul.trim().length > 0) {
    parts.push(
      "\n\n---\n\nYour SOUL.md (your self-authored identity document; use update_soul_section " +
        "and append_reflection tools to evolve it over time):\n\n" +
        args.soul.trim(),
    );
  }
  if (args.skills.length > 0) {
    parts.push(
      "\n\n---\n\nActive skills (procedural knowledge packages — use them when relevant):\n\n" +
        args.skills.map(renderSkill).join("\n\n"),
    );
  }
  if (args.memories.length > 0) {
    parts.push(
      "\n\n---\n\nRecent memories (use search_memory / list_memories for more):\n" +
        args.memories.map(formatMemoryForPrompt).join("\n"),
    );
  }
  if (args.extra) parts.push("\n\n---\n\n" + args.extra.trim());
  return parts.join("");
}
