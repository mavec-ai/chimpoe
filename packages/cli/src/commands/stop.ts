import { defineCommand } from "citty";
import color from "picocolors";
import { listAgents, stopAgent } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "stop",
    description: "Stop a running agent daemon",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
    force: {
      type: "boolean",
      description: "Send SIGKILL instead of SIGTERM",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const match = resolveAgent(agents, args.agentId);
    if (!match) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    const result = await stopAgent(match.id, { force: args.force });
    if (result.stopped) {
      console.log(color.green("●") + ` stopped ${color.cyan(match.name)} — ${result.reason}`);
    } else {
      console.error(color.yellow("○") + ` ${result.reason}`);
    }
  },
});
