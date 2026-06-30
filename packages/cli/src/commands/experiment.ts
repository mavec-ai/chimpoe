import { defineCommand } from "citty";
import color from "picocolors";
import {
  computeMetrics,
  diffRuns,
  exportEventsJsonl,
  formatMetrics,
  getEvents,
  getReplayFrames,
  getRun,
  getRunStatus,
  listPresets,
  listRuns,
  parseConfigYaml,
  renderAsciiTreeAtTime,
  renderFrameSummary,
  resolveConfig,
  startRun,
} from "@chimpoe/core";

async function resolveRun(query: string) {
  let run = await getRun(query);
  if (!run) {
    const all = await listRuns();
    run = all.find((r) => r.id.startsWith(query)) ?? null;
  }
  return run;
}

function ageLabel(ms: number): string {
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  return `${hr}h`;
}

export default defineCommand({
  meta: {
    name: "experiment",
    description: "Run, observe, and analyze colony experiments",
  },
  args: {
    action: {
      type: "positional",
      required: false,
      description: "presets | run | status | list | export | replay | diff",
    },
    target: {
      type: "positional",
      required: false,
      description: "Run id, preset name, or config file path (depending on action)",
    },
    second: {
      type: "positional",
      required: false,
      description: "Second run id (for diff) or output path (for export)",
    },
    preset: {
      type: "string",
      description:
        "Preset name (for run): garden | high-pressure | mutation-storm | isolated-comparison",
    },
    config: {
      type: "string",
      description: "Path to YAML config file (alternative to --preset)",
    },
    duration: {
      type: "string",
      description: "Duration override (e.g. 30s, 5m, 1h)",
    },
    dryRun: {
      type: "boolean",
      description: "Validate config + create run record, but don't spawn agents",
    },
  },
  run: async ({ args }) => {
    const action = args.action ?? "list";

    if (action === "presets") {
      const presets = listPresets();
      console.log(color.cyan("Available presets:"));
      for (const p of presets) {
        console.log(`  ${color.green(p.name.padEnd(20))} ${p.description}`);
      }
      return;
    }

    if (action === "run") {
      let config;
      if (args.config) {
        try {
          const text = await Bun.file(args.config).text();
          config = parseConfigYaml(text);
        } catch (err) {
          console.error(
            color.red(`Cannot read/parse config: ${err instanceof Error ? err.message : err}`),
          );
          process.exit(1);
        }
      } else if (args.preset) {
        const overrides: Record<string, unknown> = {};
        if (args.duration) {
          overrides.durationMs = parseDuration(args.duration);
        }
        config = resolveConfig({ preset: args.preset, overrides });
      } else {
        console.error(color.red("Provide --preset <name> or --config <path>."));
        console.error(color.dim("See available presets: chimpoe experiment presets"));
        process.exit(2);
      }

      console.log(color.cyan(`Starting experiment: ${config.name}`));
      console.log(color.dim(`  agents: ${config.agents.length}`));
      console.log(color.dim(`  duration: ${formatDurationLabel(config.durationMs ?? 0)}`));
      console.log("");

      const result = await startRun({
        config,
        dryRun: args.dryRun,
        taskIntervalMs: 5_000,
      });

      console.log(color.green("●") + ` run ${color.cyan(result.run.id.slice(0, 8))} ended`);
      console.log(
        color.dim(`  events: ${result.totalEvents}  duration: ${ageLabel(result.durationMs)}`),
      );
      return;
    }

    if (action === "list") {
      const runs = await listRuns();
      if (runs.length === 0) {
        console.log(
          color.gray("No experiment runs yet. Try ") +
            color.cyan("chimpoe experiment presets") +
            color.gray(" then run one."),
        );
        return;
      }
      for (const r of runs) {
        const when = new Date(r.createdAt).toISOString().slice(0, 19);
        const duration = r.startedAt && r.endedAt ? ageLabel(r.endedAt - r.startedAt) : "-";
        const status =
          r.status === "completed"
            ? color.green(r.status)
            : r.status === "running"
              ? color.yellow(r.status)
              : r.status === "failed"
                ? color.red(r.status)
                : color.gray(r.status);
        console.log(
          `${color.cyan(r.id.slice(0, 8))} ${r.name.padEnd(24)} ${status.padEnd(12)} ${color.dim(when)} ${color.dim(duration.padStart(6))}`,
        );
      }
      return;
    }

    if (action === "status") {
      if (!args.target) {
        console.error(color.red("Usage: chimpoe experiment status <run-id>"));
        process.exit(2);
      }
      const status = await getRunStatus(args.target);
      if (!status) {
        console.error(color.red(`No run matching "${args.target}".`));
        process.exit(1);
      }
      const m = await computeMetrics(status.run.id);
      if (m) console.log(formatMetrics(m));
      return;
    }

    if (action === "export") {
      if (!args.target) {
        console.error(color.red("Usage: chimpoe experiment export <run-id> [output-path]"));
        process.exit(2);
      }
      const run = await resolveRun(args.target);
      if (!run) {
        console.error(color.red(`No run matching "${args.target}".`));
        process.exit(1);
      }
      const events = await getEvents(run.id, { limit: 100_000 });
      const jsonl = exportEventsJsonl(events, run);
      const outPath = args.second ?? `${run.name}-${run.id.slice(0, 8)}.jsonl`;
      await Bun.write(outPath, jsonl + "\n");
      console.log(color.green("●") + ` exported ${events.length} events → ${color.cyan(outPath)}`);
      return;
    }

    if (action === "replay") {
      if (!args.target) {
        console.error(color.red("Usage: chimpoe experiment replay <run-id>"));
        process.exit(2);
      }
      const run = await resolveRun(args.target);
      if (!run) {
        console.error(color.red(`No run matching "${args.target}".`));
        process.exit(1);
      }
      const frames = await getReplayFrames(run.id, { bucketMs: 60_000 });
      if (frames.length === 0) {
        console.log(color.gray("No events to replay."));
        return;
      }
      console.log(color.cyan(`Replaying ${run.name} (${frames.length} frames)`));
      console.log(color.gray("-".repeat(60)));
      for (const frame of frames) {
        console.log(renderFrameSummary(frame));
      }
      console.log(color.gray("-".repeat(60)));
      console.log("");
      console.log(color.yellow("Final lineage tree:"));
      const tree = await renderAsciiTreeAtTime(true);
      console.log(tree);
      return;
    }

    if (action === "diff") {
      if (!args.target || !args.second) {
        console.error(color.red("Usage: chimpoe experiment diff <run-A> <run-B>"));
        process.exit(2);
      }
      const result = await diffRuns(args.target, args.second);
      if (!result) {
        console.error(color.red("One or both runs not found."));
        process.exit(1);
      }
      console.log(color.cyan(result.runA.name) + "  vs  " + color.cyan(result.runB.name));
      console.log(color.gray("-".repeat(60)));
      console.log(
        `${"field".padEnd(22)}${result.runA.name.slice(0, 12).padStart(14)}${result.runB.name.slice(0, 12).padStart(14)}${"delta".padStart(10)}`,
      );
      for (const f of result.fields) {
        const deltaNum = Number.parseInt(f.delta, 10);
        const deltaStr =
          deltaNum > 0 ? color.green(f.delta) : deltaNum < 0 ? color.red(f.delta) : f.delta;
        console.log(
          `${f.name.padEnd(22)}${String(f.a).padStart(14)}${String(f.b).padStart(14)}${deltaStr.padStart(20)}`,
        );
      }
      return;
    }

    console.error(
      color.red(
        `Unknown action "${action}". Use presets, run, status, list, export, replay, diff.`,
      ),
    );
    process.exit(2);
  },
});

function parseDuration(s: string): number {
  const m = s.match(/^(\d+)(s|m|h|d)$/);
  if (!m) throw new Error(`Invalid duration: ${s}`);
  const n = Number.parseInt(m[1]!, 10);
  switch (m[2]) {
    case "s":
      return n * 1000;
    case "m":
      return n * 60 * 1000;
    case "h":
      return n * 60 * 60 * 1000;
    case "d":
      return n * 24 * 60 * 60 * 1000;
  }
  throw new Error(`Invalid duration unit: ${m[2]}`);
}

function formatDurationLabel(ms: number): string {
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m`;
  return `${Math.floor(ms / 3_600_000)}h`;
}
