import { getEvents, getRun, type ExperimentEvent, type ExperimentRun } from "./store.ts";
import { listAgents } from "../state/agents.ts";
import { getLineageSummary } from "../lineage/index.ts";
import { countFossils } from "../fossils/index.ts";

export interface RunMetrics {
  runId: string;
  runName: string;
  status: string;
  startedAt: number | null;
  endedAt: number | null;
  durationMs: number;
  totalEvents: number;
  eventsByKind: Record<string, number>;
  totalAgents: number;
  livingAgents: number;
  deadAgents: number;
  generations: number;
  roots: number;
  fossilsDistilled: number;
  tasksAssigned: number;
  tasksCompleted: number;
  messagesSent: number;
  childrenSpawned: number;
  notes: string[];
}

export async function computeMetrics(runId: string): Promise<RunMetrics | null> {
  let run = await getRun(runId);
  if (!run) {
    const { listRuns } = await import("./store.ts");
    const all = await listRuns();
    run = all.find((r) => r.id.startsWith(runId)) ?? null;
  }
  if (!run) return null;
  const events = await getEvents(run.id, { limit: 100_000 });
  const eventsByKind: Record<string, number> = {};
  for (const e of events) {
    eventsByKind[e.kind] = (eventsByKind[e.kind] ?? 0) + 1;
  }

  const summary = await getLineageSummary();
  const fossils = await countFossils();
  const allAgents = await listAgents();

  const durationMs = run.endedAt && run.startedAt ? run.endedAt - run.startedAt : 0;
  const notes: string[] = [];
  if (eventsByKind["tier_transition"] && eventsByKind["tier_transition"] > 5) {
    notes.push("Multiple tier transitions — economy pressure observed.");
  }
  if ((eventsByKind["child_spawned"] ?? 0) > 0 && (eventsByKind["fossil_distilled"] ?? 0) > 0) {
    notes.push("Reproduction AND fossilization — cross-generational transfer happened.");
  }
  if (eventsByKind["agent_died"] && allAgents.length === 0) {
    notes.push("All agents died during the run — possible ecological collapse.");
  }
  return {
    runId: run.id,
    runName: run.name,
    status: run.status,
    startedAt: run.startedAt,
    endedAt: run.endedAt,
    durationMs,
    totalEvents: events.length,
    eventsByKind,
    totalAgents: summary.totalAgents,
    livingAgents: summary.living,
    deadAgents: summary.dead,
    generations: summary.generations,
    roots: summary.roots,
    fossilsDistilled: fossils,
    tasksAssigned: eventsByKind["task_assigned"] ?? 0,
    tasksCompleted: eventsByKind["task_completed"] ?? 0,
    messagesSent: eventsByKind["message_sent"] ?? 0,
    childrenSpawned: eventsByKind["child_spawned"] ?? 0,
    notes,
  };
}

export interface DiffResult {
  runA: { id: string; name: string };
  runB: { id: string; name: string };
  fields: Array<{ name: string; a: string | number; b: string | number; delta: string }>;
}

export async function diffRuns(runAId: string, runBId: string): Promise<DiffResult | null> {
  const [a, b] = await Promise.all([computeMetrics(runAId), computeMetrics(runBId)]);
  if (!a || !b) return null;
  const fields: DiffResult["fields"] = [];
  const numericFields: Array<keyof RunMetrics> = [
    "totalEvents",
    "totalAgents",
    "livingAgents",
    "deadAgents",
    "generations",
    "roots",
    "fossilsDistilled",
    "tasksAssigned",
    "tasksCompleted",
    "messagesSent",
    "childrenSpawned",
    "durationMs",
  ];
  for (const key of numericFields) {
    const va = a[key] as number;
    const vb = b[key] as number;
    const delta = vb - va;
    const sign = delta >= 0 ? "+" : "";
    fields.push({
      name: key,
      a: va,
      b: vb,
      delta: `${sign}${delta}`,
    });
  }
  return {
    runA: { id: a.runId, name: a.runName },
    runB: { id: b.runId, name: b.runName },
    fields,
  };
}

export function formatMetrics(m: RunMetrics): string {
  const out: string[] = [];
  const when =
    m.startedAt && m.endedAt
      ? `${new Date(m.startedAt).toISOString().slice(0, 19)} → ${new Date(m.endedAt).toISOString().slice(0, 19)}`
      : "(not started)";
  out.push(`run: ${m.runName} (${m.runId.slice(0, 8)})`);
  out.push(`status: ${m.status}`);
  out.push(`when: ${when}`);
  out.push(`duration: ${(m.durationMs / 1000).toFixed(0)}s`);
  out.push("");
  out.push("Population:");
  out.push(`  total agents:    ${m.totalAgents}`);
  out.push(`  living:          ${m.livingAgents}`);
  out.push(`  dead:            ${m.deadAgents}`);
  out.push(`  generations:     ${m.generations}`);
  out.push(`  roots:           ${m.roots}`);
  out.push("");
  out.push("Activity:");
  out.push(`  events total:    ${m.totalEvents}`);
  out.push(`  tasks assigned:  ${m.tasksAssigned}`);
  out.push(`  tasks completed: ${m.tasksCompleted}`);
  out.push(`  messages sent:   ${m.messagesSent}`);
  out.push(`  children born:   ${m.childrenSpawned}`);
  out.push(`  fossils:         ${m.fossilsDistilled}`);
  if (m.notes.length > 0) {
    out.push("");
    out.push("Notes:");
    for (const n of m.notes) out.push(`  • ${n}`);
  }
  return out.join("\n");
}

export function exportEventsJsonl(events: ExperimentEvent[], run: ExperimentRun): string {
  const lines = events.map((e) =>
    JSON.stringify({
      run_id: run.id,
      run_name: run.name,
      seq: e.seq,
      kind: e.kind,
      agent_id: e.agentId,
      at: new Date(e.createdAt).toISOString(),
      ...e.payload,
    }),
  );
  return lines.join("\n");
}
