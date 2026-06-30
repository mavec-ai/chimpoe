import { defineCommand } from "citty";
import color from "picocolors";
import { createTask, listAgents, listTasks } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "task",
    description: "Post or list tasks in the bounty pool",
  },
  args: {
    action: {
      type: "positional",
      required: false,
      description: "list | post",
    },
    agent: {
      type: "positional",
      required: false,
      description: "Agent id/name (for post: assignee) or status filter (for list)",
    },
    prompt: {
      type: "string",
      description: "Task prompt (for post)",
    },
    reward: {
      type: "string",
      description: "Reward in tokens",
    },
    difficulty: {
      type: "string",
      description: "Difficulty 1-10 (default 3)",
    },
    status: {
      type: "string",
      description: "Filter by status: pending/claimed/completed/failed",
    },
    limit: {
      type: "string",
      description: "Max results (default 20)",
    },
  },
  run: async ({ args }) => {
    const action = args.action ?? "list";

    if (action === "list") {
      const status = args.status as ReturnType<typeof listTasks> extends Promise<(infer T)[]>
        ? T extends { status: string }
          ? T["status"]
          : string
        : string;
      const tasks = await listTasks({
        status: status ?? undefined,
        limit: args.limit ? Number.parseInt(args.limit, 10) : 20,
      });
      if (tasks.length === 0) {
        console.log(color.gray("No tasks in pool."));
        return;
      }
      const agents = await listAgents();
      const nameFor = (id: string | null) => {
        if (!id || id === "user") return id ?? "-";
        const a = agents.find((x) => x.id === id);
        return a?.name ?? id.slice(0, 8);
      };
      for (const t of tasks) {
        const reward = t.rewardTokens > 0 ? color.green(`+${t.rewardTokens}`) : color.gray("0");
        const state =
          t.status === "completed"
            ? color.green(t.status)
            : t.status === "failed"
              ? color.red(t.status)
              : t.status === "pending"
                ? color.yellow(t.status)
                : t.status;
        console.log(
          `${color.cyan(t.id.slice(0, 8))} ${state.padEnd(12)} ${reward.padEnd(8)} d=${t.difficulty} from=${nameFor(t.fromAgentId).padEnd(8)} to=${nameFor(t.assigneeId).padEnd(8)} ${t.prompt.slice(0, 60)}`,
        );
      }
      return;
    }

    if (action === "post") {
      if (!args.prompt) {
        console.error(color.red("--prompt is required for post"));
        process.exit(2);
      }
      let assigneeId: string | undefined;
      if (args.agent) {
        const agents = await listAgents();
        const me = resolveAgent(agents, args.agent);
        if (me) assigneeId = me.id;
      }
      const reward = args.reward ? Number.parseInt(args.reward, 10) : 0;
      const difficulty = args.difficulty ? Number.parseInt(args.difficulty, 10) : 3;
      const task = await createTask({
        fromAgentId: "user",
        prompt: args.prompt,
        rewardTokens: reward,
        difficulty,
        assigneeId,
      });
      const toLabel = assigneeId ? ` for ${color.cyan(args.agent!)}` : " (open pool)";
      console.log(color.green("●") + ` posted task ${color.cyan(task.id.slice(0, 8))}${toLabel}`);
      console.log(
        color.dim(
          `  reward: ${reward}  difficulty: ${difficulty}  prompt: ${task.prompt.slice(0, 60)}`,
        ),
      );
      return;
    }

    console.error(color.red(`Unknown action. Use list or post.`));
    process.exit(2);
  },
});
