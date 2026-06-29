import { defineCommand } from "citty";
import color from "picocolors";
import { listAgents, listProcesses } from "@chimpoe/core";

export default defineCommand({
  meta: {
    name: "ps",
    description: "List running agent daemons",
  },
  args: {},
  run: async () => {
    const [processes, agents] = await Promise.all([listProcesses(), listAgents()]);
    if (processes.length === 0) {
      console.log(
        color.gray("No agent daemons running. Start one with ") + color.cyan("chimpoe start <id>"),
      );
      return;
    }
    const rows = processes.map((p) => {
      const agent = agents.find((a) => a.id === p.agentId);
      const name = agent?.name ?? p.agentId.slice(0, 8);
      const alive = p.alive ? color.green("alive") : color.red("dead");
      return { id: p.agentId.slice(0, 8), pid: String(p.pid), name, alive };
    });
    const idW = Math.max(2, ...rows.map((r) => r.id.length));
    const nameW = Math.max(4, ...rows.map((r) => r.name.length));
    const pidW = Math.max(3, ...rows.map((r) => r.pid.length));
    const header = ["id".padEnd(idW), "pid".padStart(pidW), "name".padEnd(nameW), "state"].join(
      "  ",
    );
    console.log(color.gray(header));
    console.log(color.gray("-".repeat(header.length)));
    for (const r of rows) {
      console.log(
        [color.cyan(r.id.padEnd(idW)), r.pid.padStart(pidW), r.name.padEnd(nameW), r.alive].join(
          "  ",
        ),
      );
    }
  },
});
