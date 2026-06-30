import { defineCommand } from "citty";
import color from "picocolors";
import { isProtected, listProtected, protectAgent, unprotectAgent } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "protect",
    description: "Protect an agent from auto-cull, or list/remove protections",
  },
  args: {
    agentId: {
      type: "positional",
      required: false,
      description: "Agent id/name to protect/unprotect. Omit to list.",
    },
    reason: {
      type: "string",
      description: "Reason for protection (default 'manual override')",
    },
    remove: {
      type: "boolean",
      description: "Remove protection instead of adding",
    },
    list: {
      type: "boolean",
      description: "List protected agents",
    },
  },
  run: async ({ args }) => {
    if (args.list || !args.agentId) {
      const list = await listProtected();
      if (list.length === 0) {
        console.log(color.gray("No protected agents."));
        return;
      }
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      for (const p of list) {
        const ag = agents.find((a) => a.id === p.agentId);
        const name = ag?.name ?? p.agentId.slice(0, 8);
        const when = new Date(p.protectedAt).toISOString().slice(0, 19);
        console.log(
          `  ${color.cyan(p.agentId.slice(0, 8))} ${name.padEnd(16)} ${color.dim(when)} ${color.dim(p.reason)}`,
        );
      }
      return;
    }
    const agents = await import("@chimpoe/core").then((m) => m.listAgents());
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    if (args.remove) {
      await unprotectAgent(me.id);
      console.log(color.green("●") + ` unprotected ${color.cyan(me.name)}`);
      return;
    }
    await protectAgent(me.id, args.reason ?? "manual override");
    const check = await isProtected(me.id);
    console.log(
      color.green("●") +
        ` protected ${color.cyan(me.name)} (reason: ${args.reason ?? "manual override"}, active: ${check})`,
    );
  },
});
