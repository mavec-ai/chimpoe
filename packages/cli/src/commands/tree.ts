import { defineCommand } from "citty";
import color from "picocolors";
import { getChildren, getLineageSummary, getRoots } from "@chimpoe/core";
import type { AgentConfig } from "@chimpoe/types";

const STATUS_DOT: Record<string, string> = {
  running: color.green("●"),
  idle: color.gray("○"),
  sleeping: color.dim("⏾"),
  dead: color.red("✝"),
};

function agentLabel(a: AgentConfig): string {
  const dot = STATUS_DOT[a.status] ?? color.gray("?");
  const shortId = color.dim(a.id.slice(0, 8));
  return `${dot} ${a.name} ${shortId} (g${a.generation})`;
}

async function collectTree(
  agent: AgentConfig,
  prefix: string,
  isLast: boolean,
  isRoot: boolean,
  lines: string[],
): Promise<void> {
  const branch = isRoot ? "" : isLast ? "└─ " : "├─ ";
  lines.push(`${prefix}${branch}${agentLabel(agent)}`);
  const children = await getChildren(agent.id);
  if (children.length === 0) return;
  const newPrefix = isRoot ? "" : prefix + (isLast ? "   " : "│  ");
  for (let i = 0; i < children.length; i++) {
    await collectTree(children[i]!, newPrefix, i === children.length - 1, false, lines);
  }
}

export default defineCommand({
  meta: {
    name: "tree",
    description: "Show agent lineage tree",
  },
  args: {},
  run: async () => {
    const [roots, summary] = await Promise.all([getRoots(), getLineageSummary()]);
    if (roots.length === 0) {
      console.log(
        color.gray("No agents yet. Run ") +
          color.cyan("chimpoe spawn") +
          color.gray(" to create one."),
      );
      return;
    }

    const lines: string[] = [];
    for (const r of roots) {
      await collectTree(r, "", true, true, lines);
    }
    for (const line of lines) console.log(line);

    console.log("");
    console.log(
      color.dim(
        `  ${summary.totalAgents} agent(s) · ${summary.roots} root(s) · ${summary.generations} generation(s) · ${summary.living} alive · ${summary.dead} dead`,
      ),
    );
  },
});
