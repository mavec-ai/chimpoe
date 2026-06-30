import { getEvents, type ExperimentEvent } from "./store.ts";
import { getChildren, getRoots } from "../lineage/index.ts";
import type { AgentConfig } from "@chimpoe/types";

export interface ReplayFrame {
  at: number;
  seq: number;
  eventsSinceLastFrame: ExperimentEvent[];
}

export async function getReplayFrames(
  runId: string,
  options: { bucketMs?: number; maxFrames?: number } = {},
): Promise<ReplayFrame[]> {
  const events = await getEvents(runId, { limit: 100_000 });
  if (events.length === 0) return [];
  const bucketMs = options.bucketMs ?? 1000;
  const frames: ReplayFrame[] = [];
  let bucketStart = events[0]!.createdAt;
  let current: ExperimentEvent[] = [];
  for (const e of events) {
    if (e.createdAt - bucketStart >= bucketMs) {
      if (current.length > 0) {
        frames.push({
          at: bucketStart,
          seq: current[current.length - 1]!.seq,
          eventsSinceLastFrame: current,
        });
      }
      bucketStart = e.createdAt;
      current = [];
      if (options.maxFrames && frames.length >= options.maxFrames) break;
    }
    current.push(e);
  }
  if (current.length > 0) {
    frames.push({
      at: bucketStart,
      seq: current[current.length - 1]!.seq,
      eventsSinceLastFrame: current,
    });
  }
  return frames;
}

export function renderFrameSummary(frame: ReplayFrame): string {
  const ts = new Date(frame.at).toISOString().slice(11, 19);
  const byKind: Record<string, number> = {};
  for (const e of frame.eventsSinceLastFrame) {
    byKind[e.kind] = (byKind[e.kind] ?? 0) + 1;
  }
  const parts = Object.entries(byKind).map(([k, v]) => `${k}=${v}`);
  return `[${ts}] seq=${frame.seq}  ${parts.join("  ")}`;
}

export interface TimelineSnapshot {
  populationByStatus: Record<string, number>;
  newSpawns: string[];
  newDeaths: string[];
  messagesCount: number;
}

export async function computeTimelineSnapshot(
  events: ExperimentEvent[],
): Promise<TimelineSnapshot> {
  const snap: TimelineSnapshot = {
    populationByStatus: {},
    newSpawns: [],
    newDeaths: [],
    messagesCount: 0,
  };
  for (const e of events) {
    if (e.kind === "agent_spawned") {
      snap.newSpawns.push((e.payload?.name as string) ?? e.agentId ?? "?");
    }
    if (e.kind === "agent_died") {
      snap.newDeaths.push((e.payload?.name as string) ?? e.agentId ?? "?");
    }
    if (e.kind === "message_sent") snap.messagesCount++;
  }
  return snap;
}

export async function renderAsciiTreeAtTime(runEnd: boolean): Promise<string> {
  if (!runEnd) return "(tree snapshot at run end)";
  const roots = await getRoots();
  const lines: string[] = [];
  for (const root of roots) {
    await walkAndRender(root, "", true, true, lines);
  }
  return lines.join("\n");
}

async function walkAndRender(
  agent: AgentConfig,
  prefix: string,
  isLast: boolean,
  isRoot: boolean,
  lines: string[],
): Promise<void> {
  const branch = isRoot ? "" : isLast ? "└─ " : "├─ ";
  const status = agent.status === "dead" ? "✝" : agent.status === "running" ? "●" : "○";
  lines.push(`${prefix}${branch}${status} ${agent.name} (g${agent.generation})`);
  const children = await getChildren(agent.id);
  if (children.length === 0) return;
  const newPrefix = isRoot ? "" : prefix + (isLast ? "   " : "│  ");
  for (let i = 0; i < children.length; i++) {
    await walkAndRender(children[i]!, newPrefix, i === children.length - 1, false, lines);
  }
}
