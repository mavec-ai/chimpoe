import { getAgentHome } from "@chimpoe/types";
import { join } from "node:path";

export function getSoulPath(agentId: string): string {
  return join(getAgentHome(agentId), "SOUL.md");
}

export const DEFAULT_SOUL_TEMPLATE = `# Identity

Born: {createdAt}
Name: {name}
Mission: {genesisPrompt}

# Self-description

I am a fresh agent. I have not yet formed a sense of who I am beyond the genesis
prompt my creator gave me. As I work, I will update this section to reflect
what I learn about myself — what I am good at, what I struggle with, what I
care about.

# Strengths

(to be discovered — update as you learn what you do well)

# Weaknesses

(to be discovered — update as you learn what trips you up)

# Current focus

(to be discovered — what are you working toward right now?)

# Reflections

(append dated notes via the append_reflection tool)
`;

export function renderDefaultSoul(args: {
  name: string;
  createdAt: number;
  genesisPrompt: string;
}): string {
  const date = new Date(args.createdAt).toISOString().slice(0, 10);
  const trimmedGenesis =
    args.genesisPrompt.length > 200 ? args.genesisPrompt.slice(0, 200) + "..." : args.genesisPrompt;
  return DEFAULT_SOUL_TEMPLATE.replace("{createdAt}", date)
    .replace("{name}", args.name)
    .replace("{genesisPrompt}", trimmedGenesis);
}
