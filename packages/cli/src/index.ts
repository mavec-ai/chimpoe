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
  },
});

await runMain(main);
