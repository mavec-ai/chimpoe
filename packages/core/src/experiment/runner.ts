import type { ExperimentConfig } from "./config.ts";
import {
  appendEvent,
  createRun,
  getEvents,
  getRun,
  listRuns,
  markRunEnded,
  markRunStarted,
  resetSeq,
  type ExperimentEvent,
  type ExperimentRun,
} from "./store.ts";
import { getAgent, listAgents, registerAgent } from "../state/agents.ts";
import { fundAgent } from "../economy/index.ts";
import { initAgentWorkspace } from "../workspace/index.ts";
import { sendMessage } from "../messaging/index.ts";
import { getLineageSummary } from "../lineage/index.ts";
import type { Provider } from "@chimpoe/types";

export interface RunOptions {
  config: ExperimentConfig;
  dryRun?: boolean;
  taskIntervalMs?: number;
  onEvent?: (event: ExperimentEvent) => void;
  signal?: AbortSignal;
}

export interface RunResult {
  run: ExperimentRun;
  agentIds: string[];
  totalEvents: number;
  durationMs: number;
}

export async function startRun(options: RunOptions): Promise<RunResult> {
  const { config, dryRun = false } = options;
  const run = await createRun({
    name: config.name,
    preset: config.preset,
    config: config as unknown as Record<string, unknown>,
  });

  if (dryRun) {
    return { run, agentIds: [], totalEvents: 0, durationMs: 0 };
  }

  await markRunStarted(run.id);
  const startMs = Date.now();
  await appendEvent({
    runId: run.id,
    kind: "run_started",
    payload: { name: config.name, preset: config.preset ?? null, config },
  });

  const agentIds: string[] = [];
  for (const spec of config.agents) {
    const agent = await registerAgent({
      name: spec.name,
      provider: (spec.provider ?? "openai") as Provider,
      modelId: spec.model ?? "gpt-5-mini",
      genesisPrompt: spec.genesis,
    });
    await initAgentWorkspace({
      agentId: agent.id,
      agentName: agent.name,
      genesisPrompt: agent.genesisPrompt,
      createdAt: agent.createdAt,
    });
    if (spec.endowment && spec.endowment > 0) {
      await fundAgent(agent.id, spec.endowment);
    }
    agentIds.push(agent.id);
    await appendEvent({
      runId: run.id,
      kind: "agent_spawned",
      agentId: agent.id,
      payload: {
        name: agent.name,
        generation: agent.generation,
        endowment: spec.endowment ?? 0,
        role: spec.role ?? "root",
      },
    });
    if (spec.endowment && spec.endowment > 0) {
      await appendEvent({
        runId: run.id,
        kind: "agent_funded",
        agentId: agent.id,
        payload: { amount: spec.endowment },
      });
    }
  }

  const { spawnAgent, stopAgent } = await import("../process/index.ts");
  const { getChimpoeHome } = await import("@chimpoe/types");
  const chimpoeHome = getChimpoeHome();
  for (const id of agentIds) {
    try {
      await spawnAgent({ agentId: id, chimpoeHome });
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      await appendEvent({
        runId: run.id,
        kind: "note",
        agentId: id,
        payload: { note: `daemon start failed: ${reason}` },
      });
    }
  }

  const totalEvents = await pollAndDispatch({
    runId: run.id,
    config,
    agentIds,
    onEvent: options.onEvent,
    signal: options.signal,
    taskIntervalMs: options.taskIntervalMs,
  });

  for (const id of agentIds) {
    try {
      await stopAgent(id);
    } catch {
      // best effort
    }
  }

  const durationMs = Date.now() - startMs;
  const summary = await getLineageSummary();
  await appendEvent({
    runId: run.id,
    kind: "run_ended",
    payload: { durationMs, ...summary },
  });
  await markRunEnded(run.id, "completed", { durationMs, ...summary });
  await resetSeq(run.id);

  const finalRun = await getRun(run.id);
  return {
    run: finalRun ?? run,
    agentIds,
    totalEvents,
    durationMs,
  };
}

async function pollAndDispatch(args: {
  runId: string;
  config: ExperimentConfig;
  agentIds: string[];
  onEvent?: (e: ExperimentEvent) => void;
  signal?: AbortSignal;
  taskIntervalMs?: number;
}): Promise<number> {
  const { runId, config, agentIds, onEvent, signal } = args;
  const start = Date.now();
  const durationMs = config.durationMs ?? 60 * 60 * 1000;
  const intervalMs = args.taskIntervalMs ?? 5_000;
  const tasks = config.taskPool ?? [];
  let taskIdx = 0;

  while (Date.now() - start < durationMs) {
    if (signal?.aborted) break;

    if (tasks.length > 0 && taskIdx < tasks.length) {
      const task = tasks[taskIdx]!;
      taskIdx++;
      const target = task.assignTo
        ? await resolveTargetAgentByName(task.assignTo, agentIds)
        : pickAny(agentIds);
      if (target) {
        await sendMessage({
          fromAgentId: "user",
          toAgentId: target,
          content: task.prompt,
          type: "task",
          metadata: task.reward ? { reward: task.reward, runId } : { runId },
        });
        await appendEvent({
          runId,
          kind: "task_assigned",
          agentId: target,
          payload: { prompt: task.prompt, reward: task.reward ?? 0 },
        });
        if (task.delayMs) await Bun.sleep(task.delayMs);
      }
    }

    await captureStateEvents(runId, agentIds, onEvent);
    await Bun.sleep(intervalMs);
  }

  const events = await getEvents(runId, { limit: 10_000 });
  return events.length;
}

async function captureStateEvents(
  runId: string,
  agentIds: string[],
  onEvent?: (e: ExperimentEvent) => void,
): Promise<void> {
  for (const id of agentIds) {
    const agent = await getAgent(id);
    if (!agent) continue;
    if (agent.status === "dead") {
      const events = await getEvents(runId, { kind: "agent_died", limit: 1 });
      if (events.length === 0) {
        const ev = await appendEvent({
          runId,
          kind: "agent_died",
          agentId: id,
          payload: { name: agent.name, finalTier: agent.tier, budget: agent.budgetTokens },
        });
        onEvent?.(ev);
      }
    }
  }
}

function resolveTargetAgent(nameOrId: string, agentIds: string[]): string | null {
  if (agentIds.includes(nameOrId)) return nameOrId;
  for (const id of agentIds) {
    if (id.startsWith(nameOrId)) return id;
  }
  return null;
}

async function resolveTargetAgentByName(
  nameOrId: string,
  agentIds: string[],
): Promise<string | null> {
  const direct = resolveTargetAgent(nameOrId, agentIds);
  if (direct) return direct;
  const agents = await listAgents();
  const match = agents.find(
    (a) => a.name.toLowerCase() === nameOrId.toLowerCase() && agentIds.includes(a.id),
  );
  return match?.id ?? null;
}

function pickAny(agentIds: string[]): string | null {
  if (agentIds.length === 0) return null;
  const idx = Math.floor(Math.random() * agentIds.length);
  return agentIds[idx]!;
}

export interface RunStatus {
  run: ExperimentRun;
  recentEvents: ExperimentEvent[];
  agentCount: number;
}

export async function getRunStatus(runId: string): Promise<RunStatus | null> {
  let run = await getRun(runId);
  if (!run) {
    const all = await listRuns();
    run = all.find((r) => r.id.startsWith(runId)) ?? null;
  }
  if (!run) return null;
  const recent = await getEvents(run.id, { limit: 20 });
  const allAgents = await listAgents();
  return {
    run,
    recentEvents: recent,
    agentCount: allAgents.length,
  };
}
