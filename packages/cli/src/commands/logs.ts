import { defineCommand } from "citty";
import color from "picocolors";
import { getAgentHome } from "@chimpoe/types";
import { join } from "node:path";
import { listAgents } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

async function tailFile(path: string, follow: boolean, n: number): Promise<void> {
  try {
    const text = await Bun.file(path).text();
    const lines = text.split("\n");
    const start = Math.max(0, lines.length - n);
    const slice = lines.slice(start).join("\n");
    process.stdout.write(slice);
    if (!slice.endsWith("\n")) process.stdout.write("\n");
    if (follow) {
      let offset = (await Bun.file(path).size) ?? 0;
      while (true) {
        await Bun.sleep(500);
        const newSize = await Bun.file(path).size;
        if (newSize > offset) {
          const buf = await Bun.file(path).slice(offset, newSize).bytes();
          process.stdout.write(buf);
          offset = newSize;
        }
      }
    }
  } catch {
    console.error(color.red(`Cannot read ${path}`));
    process.exit(1);
  }
}

export default defineCommand({
  meta: {
    name: "logs",
    description: "Tail an agent's log file",
  },
  args: {
    agentId: {
      type: "positional",
      required: true,
      description: "Agent id/name",
    },
    tail: {
      type: "string",
      description: "Number of lines to show initially (default 30)",
    },
    follow: {
      type: "boolean",
      alias: "f",
      description: "Follow log updates",
    },
  },
  run: async ({ args }) => {
    const agents = await listAgents();
    const me = resolveAgent(agents, args.agentId);
    if (!me) {
      console.error(color.red(`No agent matching "${args.agentId}".`));
      process.exit(1);
    }
    const logPath = join(getAgentHome(me.id), "agent.log");
    const n = args.tail ? Number.parseInt(args.tail, 10) : 30;
    await tailFile(logPath, args.follow ?? false, n);
  },
});
