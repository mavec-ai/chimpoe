import { defineCommand } from "citty";
import color from "picocolors";
import {
  countFossils,
  distillAgent,
  getFossilByAgent,
  listFossils,
  searchFossils,
} from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "fossils",
    description: "Browse or trigger knowledge fossils from dead agents",
  },
  args: {
    action: {
      type: "positional",
      required: false,
      description: "list | show <agent> | search <query> | distill <agent>",
    },
    target: {
      type: "positional",
      required: false,
      description: "Agent id/name (for show/distill) or query string (for search)",
    },
    limit: {
      type: "string",
      description: "Max results (default 20)",
    },
  },
  run: async ({ args }) => {
    const action = args.action ?? "list";

    if (action === "list") {
      const limit = args.limit ? Number.parseInt(args.limit, 10) : 20;
      const fossils = await listFossils(limit);
      const total = await countFossils();
      if (fossils.length === 0) {
        console.log(color.gray("No fossils yet. Agents distill one when they die (budget = 0)."));
        return;
      }
      for (const f of fossils) {
        const when = new Date(f.createdAt).toISOString().slice(0, 10);
        const kw =
          f.keywords.length > 0 ? color.dim(` [${f.keywords.slice(0, 5).join(", ")}]`) : "";
        console.log(
          `${color.cyan(f.id.slice(0, 8))} ${color.yellow(`g${f.generation}`)} ${f.agentName.padEnd(16)} ${color.dim(when)} ${color.dim(`${f.content.length}b`)}${kw}`,
        );
      }
      console.log("");
      console.log(color.dim(`  ${total} fossil(s) total`));
      return;
    }

    if (action === "show") {
      if (!args.target) {
        console.error(color.red("show requires an agent id/name"));
        process.exit(2);
      }
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      const me = resolveAgent(agents, args.target);
      if (me) {
        const fossil = await getFossilByAgent(me.id);
        if (!fossil) {
          console.error(
            color.red(`No fossil for ${me.name} (still alive or distilled under different id).`),
          );
          process.exit(1);
        }
        console.log(fossil.content);
        return;
      }
      const byId = await import("@chimpoe/core").then((m) => m.getFossil(args.target!));
      if (byId) {
        console.log(byId.content);
        return;
      }
      console.error(color.red(`No agent or fossil matching "${args.target}".`));
      process.exit(1);
    }

    if (action === "search") {
      if (!args.target) {
        console.error(color.red("search requires a query"));
        process.exit(2);
      }
      const limit = args.limit ? Number.parseInt(args.limit, 10) : 5;
      const results = await searchFossils(args.target, { limit });
      if (results.length === 0) {
        console.log(color.gray(`No fossils matching "${args.target}".`));
        return;
      }
      for (const f of results) {
        console.log(
          color.cyan(f.id.slice(0, 8)) + " " + color.yellow(`g${f.generation}`) + " " + f.agentName,
        );
        console.log(color.dim("  keywords: " + f.keywords.slice(0, 8).join(", ")));
        console.log(color.dim("  excerpt:"));
        const excerpt = f.content.slice(0, 500).replace(/\n/g, "\n    ");
        console.log("    " + excerpt + (f.content.length > 500 ? color.dim("\n    ...") : ""));
        console.log("");
      }
      return;
    }

    if (action === "distill") {
      if (!args.target) {
        console.error(color.red("distill requires an agent id/name"));
        process.exit(2);
      }
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      const me = resolveAgent(agents, args.target);
      if (!me) {
        console.error(color.red(`No agent matching "${args.target}".`));
        process.exit(1);
      }
      const result = await distillAgent(me.id);
      console.log(color.green("●") + ` distilled fossil from ${color.cyan(me.name)}`);
      console.log(color.dim(`  fossil id: ${result.fossilId}`));
      console.log(color.dim(`  size: ${result.sizeBytes} bytes`));
      console.log(color.dim(`  keywords: ${result.keywords.slice(0, 8).join(", ")}`));
      return;
    }

    console.error(color.red(`Unknown action "${action}". Use list, show, search, or distill.`));
    process.exit(2);
  },
});
