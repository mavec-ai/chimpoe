import { defineCommand } from "citty";
import color from "picocolors";
import {
  abandonAllClaimedByAgent,
  distillAgent,
  getAgent,
  listAgents,
  updateAgentStatus,
} from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "kill",
    description: "Mark an agent dead, distill a fossil, stop its daemon if running",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
    noDistill: {
      type: "boolean",
      description: "Skip fossil distillation",
    },
    reason: {
      type: "string",
      description: "Reason recorded in the modification/event log",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    const fresh = await getAgent(me.id);
    if (fresh?.status === "dead") {
      console.error(color.yellow(`${me.name} is already dead.`));
      process.exit(0);
    }
    try {
      const { stopAgent } = await import("@chimpoe/core");
      await stopAgent(me.id).catch(() => {});
    } catch {
      // best effort
    }
    if (!args.noDistill) {
      try {
        const d = await distillAgent(me.id);
        console.log(
          color.dim(`  fossil distilled (${d.sizeBytes}b, ${d.keywords.length} keywords)`),
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(color.yellow(`  fossil distill failed: ${msg}`));
      }
    }
    await updateAgentStatus(me.id, "dead", "dead");
    const abandoned = await abandonAllClaimedByAgent(me.id);
    if (abandoned > 0) console.log(color.dim(`  ${abandoned} task(s) returned to pool`));
    console.log(color.red("✝") + ` killed ${color.cyan(me.name)} (${me.id.slice(0, 8)})`);
    if (args.reason) console.log(color.dim(`  reason: ${args.reason}`));
  },
});
