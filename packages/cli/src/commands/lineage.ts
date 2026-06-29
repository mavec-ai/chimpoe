import { defineCommand } from "citty";
import color from "picocolors";
import { getAncestors, getDescendants, getSiblings, listAgents } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

function ageLabel(ms: number): string {
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h`;
  return `${Math.floor(hr / 24)}d`;
}

export default defineCommand({
  meta: {
    name: "lineage",
    description: "Inspect an agent's ancestors, siblings, and descendants",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }

    const [ancestors, siblings, descendants] = await Promise.all([
      getAncestors(me.id),
      getSiblings(me.id),
      getDescendants(me.id),
    ]);

    console.log(
      color.cyan(me.name) +
        color.dim(` · ${me.id.slice(0, 8)} · gen ${me.generation} · ${me.status}`),
    );
    console.log(color.gray("-".repeat(60)));

    if (ancestors.length > 0) {
      console.log(color.yellow("Ancestors (root → parent):"));
      for (const a of ancestors) {
        console.log(
          `  ${a.name} ${color.dim(a.id.slice(0, 8))} gen ${a.generation} (${ageLabel(Date.now() - a.createdAt)} ago)`,
        );
      }
      console.log("");
    }

    if (siblings.length > 0) {
      console.log(color.yellow("Siblings:"));
      for (const s of siblings) {
        console.log(`  ${s.name} ${color.dim(s.id.slice(0, 8))} (${s.status})`);
      }
      console.log("");
    }

    if (descendants.length > 0) {
      console.log(color.yellow(`Descendants (${descendants.length}):`));
      for (const d of descendants) {
        const indent = "  ".repeat(Math.max(0, d.generation - me.generation));
        console.log(
          `${indent}${d.name} ${color.dim(d.id.slice(0, 8))} gen ${d.generation} (${d.status})`,
        );
      }
      console.log("");
    }

    if (ancestors.length === 0 && siblings.length === 0 && descendants.length === 0) {
      console.log(
        color.gray("No ancestors, siblings, or descendants. This is a root with no children yet."),
      );
    }
  },
});
