import { calculateReputation } from "../reputation/index.ts";
import { listMemories } from "../memory/index.ts";
import { countTurns } from "../state/turns.ts";
import { getAgent } from "../state/agents.ts";
import { getAncestors } from "../lineage/index.ts";
import { readSoul } from "../soul/index.ts";
import { saveFossil, type SaveFossilInput } from "./store.ts";

const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "are",
  "but",
  "not",
  "you",
  "all",
  "any",
  "can",
  "her",
  "was",
  "one",
  "our",
  "out",
  "has",
  "have",
  "from",
  "they",
  "your",
  "what",
  "with",
  "this",
  "that",
  "been",
  "said",
  "each",
  "which",
  "their",
  "will",
  "into",
  "them",
  "than",
  "then",
  "who",
  "when",
  "where",
  "use",
  "using",
]);

export interface DistillOptions {
  maxMemoriesPerType?: number;
}

export interface DistillResult {
  fossilId: string;
  agentId: string;
  content: string;
  keywords: string[];
  sizeBytes: number;
}

export async function distillAgent(
  agentId: string,
  options: DistillOptions = {},
): Promise<DistillResult> {
  const agent = await getAgent(agentId);
  if (!agent) throw new Error(`Agent ${agentId} not found`);

  const perType = options.maxMemoriesPerType ?? 5;
  const [soul, ancestors, reputation, turns, totalMemories, procedural, semantic, episodic] =
    await Promise.all([
      readSoul(agentId).catch(() => ""),
      getAncestors(agentId),
      calculateReputation(agentId),
      countTurns(agentId),
      (await import("../memory/index.ts")).countMemories(agentId),
      listMemories(agentId, { type: "procedural", limit: perType, orderBy: "important" }),
      listMemories(agentId, { type: "semantic", limit: perType, orderBy: "important" }),
      listMemories(agentId, { type: "episodic", limit: perType, orderBy: "important" }),
    ]);

  const lineagePath =
    ancestors.length > 0 ? [...ancestors.map((a) => a.name), agent.name] : [agent.name];

  const content = renderFossil({
    agent,
    soul,
    lineagePath,
    reputation,
    turns,
    totalMemories,
    procedural,
    semantic,
    episodic,
  });

  const keywords = extractKeywords(content, agent.genesisPrompt);

  const input: SaveFossilInput = {
    agentId,
    agentName: agent.name,
    generation: agent.generation,
    content,
    lineagePath,
    keywords,
  };
  const fossil = await saveFossil(input);
  return {
    fossilId: fossil.id,
    agentId,
    content,
    keywords,
    sizeBytes: content.length,
  };
}

interface RenderArgs {
  agent: {
    id: string;
    name: string;
    provider: string;
    modelId: string;
    genesisPrompt: string;
    generation: number;
    budgetTokens: number;
    tier: string;
    createdAt: number;
  };
  soul: string;
  lineagePath: string[];
  reputation: { score: number; eventCount: number; recentDelta: number };
  turns: number;
  totalMemories: number;
  procedural: { content: string; tags: string[]; importance: number }[];
  semantic: { content: string; tags: string[]; importance: number }[];
  episodic: { content: string; tags: string[]; importance: number }[];
}

function renderFossil(args: RenderArgs): string {
  const died = new Date().toISOString().slice(0, 10);
  const born = new Date(args.agent.createdAt).toISOString().slice(0, 10);
  const lifespan = Math.max(1, Date.now() - args.agent.createdAt);
  const lifespanStr =
    lifespan < 3_600_000
      ? `${Math.floor(lifespan / 60_000)}m`
      : lifespan < 86_400_000
        ? `${Math.floor(lifespan / 3_600_000)}h`
        : `${Math.floor(lifespan / 86_400_000)}d`;

  const sections: string[] = [];

  sections.push(
    `# Fossil of ${args.agent.name}\n\n` +
      `Agent id: \`${args.agent.id}\`\n` +
      `Generation: ${args.agent.generation}\n` +
      `Lineage: ${args.lineagePath.join(" → ")}\n` +
      `Born: ${born} · Died: ${died} · Lifespan: ${lifespanStr}\n` +
      `Provider: ${args.agent.provider}/${args.agent.modelId}\n` +
      `Final tier: ${args.agent.tier} (balance: ${args.agent.budgetTokens} tokens)\n` +
      `Final reputation: ${args.reputation.score}/100 (${args.reputation.eventCount} events)\n` +
      `Lifetime turns: ${args.turns} · Memories stored: ${args.totalMemories}\n`,
  );

  sections.push(`## Genesis prompt\n\n${args.agent.genesisPrompt}\n`);

  if (args.soul.trim().length > 0) {
    sections.push(`## SOUL.md (final state)\n\n${args.soul.trim()}\n`);
  }

  if (args.procedural.length > 0) {
    sections.push(
      "## Procedural memory (how-to / skills)\n\n" +
        args.procedural.map((m) => `- [imp=${m.importance}] ${m.content}`).join("\n") +
        "\n",
    );
  }

  if (args.semantic.length > 0) {
    sections.push(
      "## Semantic memory (facts)\n\n" +
        args.semantic.map((m) => `- [imp=${m.importance}] ${m.content}`).join("\n") +
        "\n",
    );
  }

  if (args.episodic.length > 0) {
    sections.push(
      "## Episodic memory (events)\n\n" +
        args.episodic.map((m) => `- [imp=${m.importance}] ${m.content}`).join("\n") +
        "\n",
    );
  }

  sections.push(
    "## Self-reflection (auto-distilled)\n\n" +
      `This agent lived ${lifespanStr}, completed ${args.turns} turn(s), and ended ` +
      `${args.agent.tier} with ${args.agent.budgetTokens} tokens and reputation ` +
      `${args.reputation.score}. Descendants reading this should note what worked ` +
      `(procedural + semantic memory above) and what didn't (low-importance episodic). ` +
      `Inherit the strengths; iterate on the weaknesses.\n`,
  );

  return sections.join("\n").trim() + "\n";
}

export function extractKeywords(content: string, genesis: string): string[] {
  const text = `${content} ${genesis}`.toLowerCase();
  const counts = new Map<string, number>();
  for (const word of text.split(/[^a-z0-9]+/i)) {
    const w = word.trim();
    if (w.length < 4 || STOP_WORDS.has(w)) continue;
    counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([w]) => w);
}
