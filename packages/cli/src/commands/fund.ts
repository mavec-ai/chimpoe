import { defineCommand } from "citty";
import color from "picocolors";
import { fundAgent, getAgent, listAgents } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "fund",
    description: "Top up an agent's token budget",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
    amount: {
      type: "positional",
      required: true,
      description: "Tokens to add (positive integer)",
    },
  },
  run: async ({ args }) => {
    const amount = Number.parseInt(args.amount, 10);
    if (!Number.isFinite(amount) || amount <= 0) {
      console.error(color.red("Amount must be a positive integer."));
      process.exit(2);
    }
    const agents = await listAgents();
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    const newBalance = await fundAgent(me.id, amount);
    const fresh = await getAgent(me.id);
    console.log(
      color.green("●") + ` funded ${color.cyan(me.name)} with ${color.cyan(String(amount))} tokens`,
    );
    console.log(color.dim(`  balance: ${newBalance}  tier: ${fresh?.tier ?? "?"}`));
  },
});
