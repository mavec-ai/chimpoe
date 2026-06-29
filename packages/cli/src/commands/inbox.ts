import { defineCommand } from "citty";
import color from "picocolors";
import { checkInbox, listAgents, markAllRead } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "inbox",
    description: "Inspect an agent's inbox (use --read to mark all as read)",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id, short id prefix, or name",
    },
    all: {
      type: "boolean",
      description: "Show all messages (not just unread)",
    },
    limit: {
      type: "string",
      description: "Max messages to show (default 20)",
    },
    markRead: {
      type: "boolean",
      description: "Mark all shown messages as read",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const match = resolveAgent(agents, args.agentId);
    if (!match) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }

    const limit = args.limit ? Number.parseInt(args.limit, 10) : 20;
    const messages = await checkInbox(match.id, { unreadOnly: !args.all, limit });

    if (messages.length === 0) {
      console.log(color.gray(`No ${args.all ? "" : "unread "}messages for ${match.name}.`));
      return;
    }

    const nameFor = (id: string) => {
      if (id === "user") return "user";
      const a = agents.find((x) => x.id === id);
      return a ? a.name : id.slice(0, 8);
    };

    console.log(color.cyan(match.name) + color.dim(`  ·  ${messages.length} message(s)`));
    console.log(color.gray("-".repeat(60)));
    for (const m of messages) {
      const when = new Date(m.createdAt).toISOString().slice(0, 19).replace("T", " ");
      const arrow = m.fromAgentId === match.id ? "→" : "←";
      const fromLabel =
        m.fromAgentId === match.id ? `(sent) → ${nameFor(m.toAgentId)}` : nameFor(m.fromAgentId);
      const typeTag = m.type === "text" ? "" : color.yellow(`[${m.type}] `);
      console.log(`${color.dim(when)} ${color.green(arrow)} ${color.cyan(fromLabel)} ${typeTag}`);
      console.log(`   ${m.content}`);
      if (m.readAt)
        console.log(color.dim(`   (read ${new Date(m.readAt).toISOString().slice(11, 19)})`));
      console.log("");
    }

    if (args.markRead) {
      const n = await markAllRead(match.id);
      console.log(color.dim(`marked ${n} message(s) as read.`));
    }
  },
});
