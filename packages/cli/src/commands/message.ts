import { defineCommand } from "citty";
import color from "picocolors";
import { listAgents, sendMessage, type MessageType } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

const VALID_TYPES: MessageType[] = ["text", "task", "result"];

export default defineCommand({
  meta: {
    name: "message",
    description: "Send a message from one agent to another (or from the user to an agent)",
  },
  args: {
    to: {
      type: "positional",
      required: true,
      description: "Recipient agent id (or short prefix). Use 'user' to send as the human.",
    },
    content: {
      type: "positional",
      required: true,
      description: "Message body",
    },
    from: {
      type: "string",
      description: "Sender agent id (or 'user'). Default 'user'.",
    },
    type: {
      type: "string",
      description: "Message type: text, task, result, system (default text)",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const recipient = resolveAgent(agents, args.to);
    if (!recipient) {
      console.error(color.red(`No agent matching "${args.to}".`));
      process.exit(1);
    }

    const fromIdRaw = args.from ?? "user";
    let fromId: string;
    if (fromIdRaw === "user") {
      fromId = "user";
    } else {
      const sender = resolveAgent(agents, fromIdRaw);
      if (!sender) {
        console.error(color.red(`No sender agent matching "${fromIdRaw}".`));
        process.exit(1);
      }
      fromId = sender.id;
    }

    const type = (args.type ?? "text") as MessageType;
    if (!VALID_TYPES.includes(type)) {
      console.error(color.red(`Invalid type "${args.type}". Use: ${VALID_TYPES.join(", ")}`));
      process.exit(1);
    }

    const msg = await sendMessage({
      fromAgentId: fromId,
      toAgentId: recipient.id,
      content: args.content,
      type,
    });

    console.log(
      color.green("●") + ` ${color.cyan(fromIdRaw)} → ${color.cyan(recipient.name)} (${msg.type})`,
    );
    console.log(color.dim(`  message ${msg.id.slice(0, 8)} queued`));
    console.log(
      color.dim(`  if ${recipient.name} is running ("chimpoe ps"), it will reply on next poll`),
    );
  },
});
