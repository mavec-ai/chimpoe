import { defineCommand } from "citty";
import color from "picocolors";
import { listAgents, spawnAgent } from "@chimpoe/core";
import { getChimpoeHome } from "@chimpoe/types";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "start",
    description: "Start an agent as a background daemon (polls inbox, replies to messages)",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
  },
  run: async ({ args }) => {
    const chimpoeHome = getChimpoeHome();
    const agents = await listAgents();
    const match = resolveAgent(agents, args.agentId);
    if (!match) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    try {
      const result = await spawnAgent({ agentId: match.id, chimpoeHome });
      console.log(
        color.green("●") +
          ` started ${color.cyan(match.name)} (pid ${result.pid}, agent ${match.id.slice(0, 8)})`,
      );
      console.log(color.dim(`  log: ${chimpoeHome}/agents/${match.id}/agent.log`));
      console.log(color.dim(`  inbox polling every 1.5s — message via "chimpoe message"`));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(color.red(msg));
      process.exit(1);
    }
  },
});
