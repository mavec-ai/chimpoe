#!/usr/bin/env bun
import { defineCommand, runMain } from "citty";
import { CHIMPOE_VERSION } from "@chimpoe/types";

const argv = process.argv.slice(2);
if (argv.includes("--version") || argv.includes("-V")) {
  console.log(CHIMPOE_VERSION);
  process.exit(0);
}

const main = defineCommand({
  meta: {
    name: "chimpoe",
    version: CHIMPOE_VERSION,
    description: "Local-first sovereign agent colony runtime",
  },
  subCommands: {
    init: () => import("./commands/init.ts").then((m) => m.default),
    chat: () => import("./commands/chat.ts").then((m) => m.default),
    spawn: () => import("./commands/spawn.ts").then((m) => m.default),
    list: () => import("./commands/list.ts").then((m) => m.default),
    start: () => import("./commands/start.ts").then((m) => m.default),
    stop: () => import("./commands/stop.ts").then((m) => m.default),
    ps: () => import("./commands/ps.ts").then((m) => m.default),
    message: () => import("./commands/message.ts").then((m) => m.default),
    inbox: () => import("./commands/inbox.ts").then((m) => m.default),
    tree: () => import("./commands/tree.ts").then((m) => m.default),
    lineage: () => import("./commands/lineage.ts").then((m) => m.default),
    status: () => import("./commands/status.ts").then((m) => m.default),
    fund: () => import("./commands/fund.ts").then((m) => m.default),
    reward: () => import("./commands/reward.ts").then((m) => m.default),
    fossils: () => import("./commands/fossils.ts").then((m) => m.default),
    skills: () => import("./commands/skills.ts").then((m) => m.default),
    experiment: () => import("./commands/experiment.ts").then((m) => m.default),
    kill: () => import("./commands/kill.ts").then((m) => m.default),
    protect: () => import("./commands/protect.ts").then((m) => m.default),
    cull: () => import("./commands/cull.ts").then((m) => m.default),
    task: () => import("./commands/task.ts").then((m) => m.default),
    tasks: () => import("./commands/task.ts").then((m) => m.default),
    logs: () => import("./commands/logs.ts").then((m) => m.default),
    doctor: () => import("./commands/doctor.ts").then((m) => m.default),
    reset: () => import("./commands/reset.ts").then((m) => m.default),
  },
});

await runMain(main);
