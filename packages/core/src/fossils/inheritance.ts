import { searchFossils, type Fossil } from "./store.ts";
import { getAncestors } from "../lineage/index.ts";
import { writeMemory } from "../memory/index.ts";

const INHERITANCE_TOP_N = 3;
const MAX_EXCERPT_BYTES = 2000;

export interface InheritedFossil {
  fossilId: string;
  fromAgentName: string;
  generation: number;
  excerpt: string;
  relevanceScore: number;
}

export async function selectRelevantFossils(
  childAgentId: string,
  genesisPrompt: string,
  ancestorIds: string[],
  topN = INHERITANCE_TOP_N,
): Promise<InheritedFossil[]> {
  const keywords = genesisPrompt
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .filter((w) => w.length >= 4);

  if (keywords.length === 0 || ancestorIds.length === 0) return [];

  const query = keywords.slice(0, 5).join(" ");
  const matches: Fossil[] = [];
  for (const ancestorId of ancestorIds) {
    const found = await searchFossils(query, {
      agentIds: [ancestorId],
      limit: topN,
    });
    matches.push(...found);
  }

  if (matches.length === 0) {
    return [];
  }

  const scored = matches.map((f) => ({
    fossilId: f.id,
    fromAgentName: f.agentName,
    generation: f.generation,
    excerpt: extractRelevantExcerpt(f.content, keywords),
    relevanceScore: scoreRelevance(f, keywords),
  }));
  scored.sort((a, b) => b.relevanceScore - a.relevanceScore);

  return scored.slice(0, topN);
}

export async function inheritFossilsIntoMemory(
  childAgentId: string,
  parentAgentId: string,
  genesisPrompt: string,
): Promise<InheritedFossil[]> {
  const ancestors = await getAncestors(parentAgentId);
  const ancestorIds = [parentAgentId, ...ancestors.map((a) => a.id)];
  const inherited = await selectRelevantFossils(childAgentId, genesisPrompt, ancestorIds);

  for (const f of inherited) {
    const note = `[Inherited from ${f.fromAgentName} (gen ${f.generation})]:\n` + f.excerpt;
    await writeMemory({
      agentId: childAgentId,
      type: "semantic",
      content: note,
      tags: ["fossil", "inherited", `from:${f.fromAgentName}`],
      importance: 70,
    });
  }

  return inherited;
}

function scoreRelevance(fossil: Fossil, keywords: string[]): number {
  let score = 0;
  const contentLower = fossil.content.toLowerCase();
  for (const kw of keywords) {
    const matches = contentLower.split(kw).length - 1;
    score += matches;
  }
  for (const fossilKw of fossil.keywords) {
    if (keywords.includes(fossilKw)) score += 3;
  }
  score += 1 / (1 + fossil.generation);
  return score;
}

function extractRelevantExcerpt(content: string, keywords: string[]): string {
  if (content.length <= MAX_EXCERPT_BYTES) return content;
  const lower = content.toLowerCase();
  const positions = keywords
    .map((kw) => lower.indexOf(kw))
    .filter((p) => p >= 0)
    .sort((a, b) => a - b);
  if (positions.length === 0) return content.slice(0, MAX_EXCERPT_BYTES);
  const center = positions[Math.floor(positions.length / 2)]!;
  const start = Math.max(0, center - MAX_EXCERPT_BYTES / 2);
  const end = Math.min(content.length, start + MAX_EXCERPT_BYTES);
  let excerpt = content.slice(start, end);
  if (start > 0) excerpt = "...\n" + excerpt;
  if (end < content.length) excerpt = excerpt + "\n...";
  return excerpt;
}
