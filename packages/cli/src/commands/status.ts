import { defineCommand } from "citty";
import color from "picocolors";
import {
  calculateReputation,
  countMemories,
  countTurns,
  getAgent,
  getBudgetSnapshot,
  getProcessInfo,
  listReputationEvents,
} from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

const TIER_COLOR: Record<string, (s: string) => string> = {
  thriving: color.green,
  normal: color.white,
  conservation: color.yellow,
  dormant: color.gray,
  dead: color.red,
};

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
    name: "status",
    description: "Show detailed vitals for an agent (budget, tier, reputation, activity)",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
  },
  run: async ({ args }) => {
    const agents = await import("@chimpoe/core").then((m) => m.listAgents());
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }

    const fresh = await getAgent(me.id);
    if (!fresh) {
      console.error(color.red("Agent record vanished."));
      process.exit(1);
    }

    const [budget, reputation, events, turns, memories, proc] = await Promise.all([
      getBudgetSnapshot(me.id),
      calculateReputation(me.id),
      listReputationEvents(me.id, 5),
      countTurns(me.id),
      countMemories(me.id),
      getProcessInfo(me.id),
    ]);

    const tier = TIER_COLOR[fresh.tier]?.(fresh.tier) ?? fresh.tier;
    const uptime = proc?.alive
      ? color.green(`running (pid ${proc.pid})`)
      : color.gray("not running");

    console.log(color.cyan(fresh.name) + color.dim(` · ${fresh.id.slice(0, 8)} · ${uptime}`));
    console.log(color.gray("-".repeat(60)));
    console.log(`provider   ${fresh.provider}/${fresh.modelId}`);
    console.log(
      `generation ${fresh.generation}` +
        (fresh.parentId ? color.dim(` (parent ${fresh.parentId.slice(0, 8)})`) : " (root)"),
    );
    console.log(`created    ${ageLabel(Date.now() - fresh.createdAt)} ago`);
    console.log("");
    console.log(color.yellow("Economy"));
    console.log(`  budget    ${budget?.balance ?? 0} tokens`);
    console.log(`  tier      ${tier}`);
    console.log(`  status    ${fresh.status}`);
    console.log("");
    console.log(color.yellow("Reputation"));
    console.log(
      `  score     ${color.cyan(String(reputation.score))}/100 ${reputation.recentDelta >= 0 ? color.green(`(+${reputation.recentDelta} recent)`) : color.red(`(${reputation.recentDelta} recent)`)}`,
    );
    console.log(`  events    ${reputation.eventCount} total`);
    if (events.length > 0) {
      for (const e of events) {
        const sign = e.delta >= 0 ? "+" : "";
        console.log(
          color.dim(
            `    ${e.createdAt.toString().slice(0, 10)} ${sign}${e.delta} ${e.eventType}${e.reason ? ` — ${e.reason.slice(0, 60)}` : ""}`,
          ),
        );
      }
    }
    console.log("");
    console.log(color.yellow("Activity"));
    console.log(`  turns     ${turns}`);
    console.log(`  memories  ${memories}`);
  },
});
