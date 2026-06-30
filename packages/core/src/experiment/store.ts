import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";

export type RunStatus = "pending" | "running" | "completed" | "failed" | "aborted";
export type EventKind =
  | "run_started"
  | "run_ended"
  | "agent_spawned"
  | "agent_died"
  | "agent_funded"
  | "message_sent"
  | "task_assigned"
  | "task_completed"
  | "task_failed"
  | "tier_transition"
  | "reputation_event"
  | "child_spawned"
  | "fossil_distilled"
  | "skill_installed"
  | "heartbeat_event"
  | "note";

export interface ExperimentRun {
  id: string;
  name: string;
  preset: string | null;
  configJson: string;
  status: RunStatus;
  startedAt: number | null;
  endedAt: number | null;
  createdAt: number;
  summaryJson: string | null;
}

export interface ExperimentEvent {
  id: string;
  runId: string;
  seq: number;
  kind: EventKind;
  agentId: string | null;
  payload: Record<string, unknown> | null;
  createdAt: number;
}

interface RunRow {
  id: string;
  name: string;
  preset: string | null;
  config_json: string;
  status: string;
  started_at: number | null;
  ended_at: number | null;
  created_at: number;
  summary_json: string | null;
}

interface EventRow {
  id: string;
  run_id: string;
  seq: number;
  kind: string;
  agent_id: string | null;
  payload_json: string | null;
  created_at: number;
}

function rowToRun(row: RunRow): ExperimentRun {
  return {
    id: row.id,
    name: row.name,
    preset: row.preset,
    configJson: row.config_json,
    status: row.status as RunStatus,
    startedAt: row.started_at,
    endedAt: row.ended_at,
    createdAt: row.created_at,
    summaryJson: row.summary_json,
  };
}

function rowToEvent(row: EventRow): ExperimentEvent {
  return {
    id: row.id,
    runId: row.run_id,
    seq: row.seq,
    kind: row.kind as EventKind,
    agentId: row.agent_id,
    payload: row.payload_json ? (JSON.parse(row.payload_json) as Record<string, unknown>) : null,
    createdAt: row.created_at,
  };
}

export interface CreateRunInput {
  name: string;
  preset?: string;
  config: Record<string, unknown>;
}

export async function createRun(input: CreateRunInput): Promise<ExperimentRun> {
  const id = randomUUID();
  const now = Date.now();
  await withDb((db) => {
    db.prepare(
      `INSERT INTO experiment_runs (id, name, preset, config_json, status, started_at, ended_at, created_at, summary_json)
       VALUES (?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)`,
    ).run(id, input.name, input.preset ?? null, JSON.stringify(input.config, null, 2), now);
  });
  return {
    id,
    name: input.name,
    preset: input.preset ?? null,
    configJson: JSON.stringify(input.config, null, 2),
    status: "pending",
    startedAt: null,
    endedAt: null,
    createdAt: now,
    summaryJson: null,
  };
}

export async function markRunStarted(runId: string): Promise<void> {
  await withDb((db) => {
    db.prepare("UPDATE experiment_runs SET status = 'running', started_at = ? WHERE id = ?").run(
      Date.now(),
      runId,
    );
  });
}

export async function markRunEnded(
  runId: string,
  status: RunStatus,
  summary?: Record<string, unknown>,
): Promise<void> {
  await withDb((db) => {
    db.prepare(
      "UPDATE experiment_runs SET status = ?, ended_at = ?, summary_json = ? WHERE id = ?",
    ).run(status, Date.now(), summary ? JSON.stringify(summary) : null, runId);
  });
}

export async function getRun(runId: string): Promise<ExperimentRun | null> {
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM experiment_runs WHERE id = ?").get(runId) as
      | RunRow
      | undefined;
    return row ? rowToRun(row) : null;
  });
}

export async function listRuns(limit = 50): Promise<ExperimentRun[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM experiment_runs ORDER BY created_at DESC LIMIT ?")
      .all(limit) as RunRow[];
    return rows.map(rowToRun);
  });
}

const seqCache = new Map<string, number>();

export async function appendEvent(input: {
  runId: string;
  kind: EventKind;
  agentId?: string | null;
  payload?: Record<string, unknown>;
}): Promise<ExperimentEvent> {
  const id = randomUUID();
  const now = Date.now();
  const seq = seqCache.get(input.runId) ?? 0;
  const nextSeq = seq + 1;
  seqCache.set(input.runId, nextSeq);
  await withDb((db) => {
    db.prepare(
      `INSERT INTO experiment_events (id, run_id, seq, kind, agent_id, payload_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id,
      input.runId,
      nextSeq,
      input.kind,
      input.agentId ?? null,
      input.payload ? JSON.stringify(input.payload) : null,
      now,
    );
  });
  return {
    id,
    runId: input.runId,
    seq: nextSeq,
    kind: input.kind,
    agentId: input.agentId ?? null,
    payload: input.payload ?? null,
    createdAt: now,
  };
}

export async function getEvents(
  runId: string,
  options: { kind?: EventKind; limit?: number; since?: number } = {},
): Promise<ExperimentEvent[]> {
  return withDb((db) => {
    let sql = "SELECT * FROM experiment_events WHERE run_id = ?";
    const params: (string | number)[] = [runId];
    if (options.kind) {
      sql += " AND kind = ?";
      params.push(options.kind);
    }
    if (options.since !== undefined) {
      sql += " AND seq > ?";
      params.push(options.since);
    }
    sql += " ORDER BY seq ASC";
    if (options.limit) {
      sql += " LIMIT ?";
      params.push(options.limit);
    }
    const rows = db.prepare(sql).all(...params) as EventRow[];
    return rows.map(rowToEvent);
  });
}

export async function countEvents(runId: string, kind?: EventKind): Promise<number> {
  return withDb((db) => {
    if (kind) {
      const row = db
        .prepare("SELECT COUNT(*) as n FROM experiment_events WHERE run_id = ? AND kind = ?")
        .get(runId, kind) as { n: number };
      return row.n;
    }
    const row = db
      .prepare("SELECT COUNT(*) as n FROM experiment_events WHERE run_id = ?")
      .get(runId) as { n: number };
    return row.n;
  });
}

export async function deleteRun(runId: string): Promise<void> {
  await withDb((db) => {
    db.prepare("DELETE FROM experiment_runs WHERE id = ?").run(runId);
  });
  seqCache.delete(runId);
}

export async function resetSeq(runId: string): Promise<void> {
  const maxSeq = await withDb((db) => {
    const row = db
      .prepare("SELECT MAX(seq) as m FROM experiment_events WHERE run_id = ?")
      .get(runId) as { m: number | null };
    return row.m ?? 0;
  });
  seqCache.set(runId, maxSeq);
}
