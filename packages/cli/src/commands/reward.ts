import { defineCommand } from "citty";
import color from "picocolors";
import {
  calculateReputation,
  fundAgent,
  listAgents,
  recordReputationEvent,
  type ReputationEventType,
} from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

const VALID_TYPES = new Set<ReputationEventType>([
  "user_praise",
  "task_completed",
  "bug_introduced",
  "manual_adjustment",
]);

export default defineCommand({
  meta: {
    name: "reward",
    description: "Reward an agent: +reputation and optionally +budget",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
    eventType: {
      type: "string",
      description: "Reputation event type (default user_praise)",
    },
    reason: {
      type: "string",
      description: "Why (required)",
    },
    bonus: {
      type: "string",
      description: "Optional budget bonus tokens to add",
    },
  },
  run: async ({ args }) => {
    const eventType = (args.eventType ?? "user_praise") as ReputationEventType;
    if (!VALID_TYPES.has(eventType)) {
      console.error(color.red(`Invalid type. Use: ${[...VALID_TYPES].join(", ")}`));
      process.exit(2);
    }
    if (!args.reason) {
      console.error(color.red("--reason is required."));
      process.exit(2);
    }
    const agents = await listAgents();
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }

    const event = await recordReputationEvent({
      agentId: me.id,
      eventType,
      reason: `Manual: ${args.reason}`,
    });
    let bonusLine = "";
    if (args.bonus) {
      const n = Number.parseInt(args.bonus, 10);
      if (Number.isFinite(n) && n > 0) {
        const bal = await fundAgent(me.id, n);
        bonusLine = color.dim(` (bonus ${n} tokens → balance ${bal})`);
      }
    }
    const newScore = await calculateReputation(me.id);
    console.log(
      color.green("●") +
        ` ${event.eventType} ${event.delta >= 0 ? "+" : ""}${event.delta} → ${color.cyan(me.name)}${bonusLine}`,
    );
    console.log(color.dim(`  reputation now ${newScore.score}/100`));
  },
});
