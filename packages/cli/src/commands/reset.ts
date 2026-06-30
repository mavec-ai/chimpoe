import { defineCommand } from "citty";
import color from "picocolors";
import { rm } from "node:fs/promises";
import {
  type Provider,
  getAgentsDir,
  getChimpoeHome,
  getFossilsDir,
  getConfigPath,
  getSharedStateDbPath,
} from "@chimpoe/types";
import { closeSharedDb } from "@chimpoe/core";

export default defineCommand({
  meta: {
    name: "reset",
    description: "Nuke chimpoe state. Use with extreme caution.",
  },
  args: {
    keepFossils: {
      type: "boolean",
      description: "Preserve ~/.chimpoe/fossils/",
    },
    keepConfig: {
      type: "boolean",
      description: "Preserve ~/.chimpoe/chimpoe.json and .env",
    },
    yes: {
      type: "boolean",
      description: "Skip confirmation prompt",
    },
  },
  run: async ({ args }) => {
    const home = getChimpoeHome();
    if (!args.yes) {
      console.error(color.red(`This will delete state under ${home}.`));
      console.error(color.red(`Pass --yes to confirm, or pick specific flags:`));
      console.error(color.dim("  --keep-fossils   preserve fossils/"));
      console.error(color.dim("  --keep-config    preserve chimpoe.json + .env"));
      process.exit(2);
    }
    closeSharedDb();
    const targets: string[] = [
      getSharedStateDbPath(),
      getSharedStateDbPath() + "-wal",
      getSharedStateDbPath() + "-shm",
      getAgentsDir(),
    ];
    if (!args.keepConfig) targets.push(getConfigPath(), `${home}/.env`);
    if (!args.keepFossils) targets.push(getFossilsDir());
    for (const target of targets) {
      try {
        await rm(target, { recursive: true, force: true });
        console.log(color.dim(`  removed ${target.replace(home, "~/.chimpoe")}`));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.log(color.yellow(`  ! could not remove ${target}: ${msg}`));
      }
    }
    console.log(color.green("●") + " reset complete");
  },
});

void ({} as Provider);
