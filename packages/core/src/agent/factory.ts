import { ToolLoopAgent, type Tool } from "ai";
import { resolveModel } from "../inference/resolver.ts";
import { builtinTools } from "../tools/index.ts";
import { readSoul } from "../soul/index.ts";
import { formatMemoryForPrompt, listMemories, type MemoryRecord } from "../memory/index.ts";
import type { AgentConfig } from "@chimpoe/types";

export interface CreateAgentOptions {
  config: AgentConfig;
  tools?: Record<string, Tool>;
  extraInstructions?: string;
  recentMemoryCount?: number;
}

export async function createAgent({
  config,
  tools = {},
  extraInstructions,
  recentMemoryCount = 5,
}: CreateAgentOptions): Promise<ToolLoopAgent> {
  const resolved = resolveModel(config.provider, config.modelId);

  const [soul, recentMemories] = await Promise.all([
    config.id === "ephemeral" ? Promise.resolve("") : readSoul(config.id).catch(() => ""),
    config.id === "ephemeral"
      ? Promise.resolve([])
      : listMemories(config.id, { limit: recentMemoryCount }).catch(() => []),
  ]);

  const systemPrompt = buildSystemPrompt({
    genesis: config.genesisPrompt,
    soul,
    memories: recentMemories,
    extra: extraInstructions,
  });

  const allTools = {
    ...builtinTools(config.id === "ephemeral" ? undefined : config.id),
    ...tools,
  };

  return new ToolLoopAgent({
    model: resolved.model,
    instructions: systemPrompt,
    tools: allTools,
  });
}

function buildSystemPrompt(args: {
  genesis: string;
  soul: string;
  memories: MemoryRecord[];
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
  if (args.memories.length > 0) {
    parts.push(
      "\n\n---\n\nRecent memories (use search_memory / list_memories for more):\n" +
        args.memories.map(formatMemoryForPrompt).join("\n"),
    );
  }
  if (args.extra) parts.push("\n\n---\n\n" + args.extra.trim());
  return parts.join("");
}
