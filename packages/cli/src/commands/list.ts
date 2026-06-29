import { defineCommand } from "citty";
import color from "picocolors";
import { listAgents } from "@chimpoe/core";

const TIER_COLOR: Record<string, (s: string) => string> = {
  thriving: color.green,
  normal: color.white,
  conservation: color.yellow,
  dormant: color.gray,
  dead: color.red,
};

const STATUS_COLOR: Record<string, (s: string) => string> = {
  running: color.green,
  idle: color.white,
  sleeping: color.gray,
  dead: color.red,
};

function ageLabel(ms: number): string {
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h`;
  const day = Math.floor(hr / 24);
  return `${day}d`;
}

export default defineCommand({
  meta: {
    name: "list",
    description: "List all spawned agents",
  },
  args: {},
  run: async () => {
    const agents = await listAgents();
    if (agents.length === 0) {
      console.log(
        color.gray("No agents yet. Run ") +
          color.cyan("chimpoe spawn") +
          color.gray(" to create one."),
      );
      return;
    }

    const rows = agents.map((a) => {
      const age = ageLabel(Date.now() - a.createdAt);
      const tier = (TIER_COLOR[a.tier] ?? color.white)(a.tier);
      const status = (STATUS_COLOR[a.status] ?? color.white)(a.status);
      const name = a.name.length > 20 ? a.name.slice(0, 19) + "…" : a.name;
      const shortId = a.id.slice(0, 8);
      return {
        id: shortId,
        name,
        provider: `${a.provider}/${a.modelId}`.slice(0, 30),
        gen: `g${a.generation}`,
        tier,
        status,
        age,
      };
    });

    const idW = Math.max(2, ...rows.map((r) => r.id.length));
    const nameW = Math.max(4, ...rows.map((r) => r.name.length));
    const provW = Math.max(8, ...rows.map((r) => r.provider.length));

    const header = [
      "id".padEnd(idW),
      "name".padEnd(nameW),
      "provider".padEnd(provW),
      "gen".padStart(3),
      "tier".padEnd(12),
      "status".padEnd(10),
      "age".padStart(4),
    ].join("  ");

    console.log(color.gray(header));
    console.log(color.gray("-".repeat(header.length)));
    for (const r of rows) {
      console.log(
        [
          color.cyan(r.id.padEnd(idW)),
          r.name.padEnd(nameW),
          color.dim(r.provider.padEnd(provW)),
          color.dim(r.gen.padStart(3)),
          r.tier.padEnd(12),
          r.status.padEnd(10),
          color.dim(r.age.padStart(4)),
        ].join("  "),
      );
    }
    console.log("");
    console.log(color.dim(`Showing ${agents.length} agent(s). Use full id for chimpoe chat <id>.`));
  },
});
