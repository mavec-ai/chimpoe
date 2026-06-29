import type { AgentConfig } from "@chimpoe/types";

export function resolveAgent(agents: AgentConfig[], query: string): AgentConfig | undefined {
  const q = query.trim().toLowerCase();
  return agents.find((a) => a.id === q || a.id.startsWith(q) || a.name.toLowerCase() === q);
}
