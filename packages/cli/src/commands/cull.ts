import { defineCommand } from "citty";
import color from "picocolors";
import { executeCull, scanCullCandidates } from "@chimpoe/core";

export default defineCommand({
  meta: {
    name: "cull",
    description: "Scan for and (optionally) kill low-reputation agents",
  },
  args: {
    dryRun: {
      type: "boolean",
      description: "List candidates without killing",
    },
    bottomPercent: {
      type: "string",
      description: "Also cull bottom N% by reputation (e.g. 20)",
    },
    noDistill: {
      type: "boolean",
      description: "Skip fossil distillation for culled agents",
    },
    reputationFloor: {
      type: "string",
      description: "Override reputation floor (default 25)",
    },
  },
  run: async ({ args }) => {
    const thresholds = {
      reputationFloor: args.reputationFloor ? Number.parseInt(args.reputationFloor, 10) : 25,
      sustainedHours: 24,
      bottomPercentFloor: 10,
    };
    const bottomPercent = args.bottomPercent ? Number.parseInt(args.bottomPercent, 10) : undefined;
    const candidates = await scanCullCandidates(thresholds, { bottomPercent });
    if (candidates.length === 0) {
      console.log(color.gray("No cull candidates."));
      return;
    }
    console.log(color.yellow(`Found ${candidates.length} candidate(s):`));
    for (const c of candidates) {
      console.log(
        `  ${color.cyan(c.agentId.slice(0, 8))} ${c.name.padEnd(16)} rep=${String(c.reputation).padStart(3)} bal=${String(c.budget).padStart(7)} ${color.dim(c.reason)}`,
      );
    }
    if (args.dryRun) {
      console.log(color.dim("(dry-run, no changes made)"));
      return;
    }
    const result = await executeCull(candidates, { distill: !args.noDistill });
    console.log("");
    console.log(color.green(`● culled ${result.culled.length}, failed ${result.failed.length}`));
    for (const f of result.failed) {
      console.log(color.red(`  ⊗ ${f.agentId.slice(0, 8)} — ${f.reason}`));
    }
  },
});
