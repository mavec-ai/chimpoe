import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";

export type ReputationEventType =
  | "task_completed"
  | "task_failed"
  | "task_timeout"
  | "bug_introduced"
  | "manual_intervention"
  | "user_praise"
  | "output_reused"
  | "manual_adjustment";

export interface ReputationEvent {
  id: string;
  agentId: string;
  eventType: ReputationEventType;
  delta: number;
  reason: string | null;
  relatedId: string | null;
  createdAt: number;
}

interface ReputationRow {
  id: string;
  agent_id: string;
  event_type: string;
  delta: number;
  reason: string | null;
  related_id: string | null;
  created_at: number;
}

const EVENT_DELTAS: Record<ReputationEventType, number> = {
  task_completed: 5,
  output_reused: 3,
  user_praise: 10,
  task_failed: -5,
  task_timeout: -10,
  bug_introduced: -15,
  manual_intervention: -3,
  manual_adjustment: 0,
};

const EVENT_WINDOW = 20;
const DECAY_HALF_LIFE_DAYS = 30;

function rowToEvent(row: ReputationRow): ReputationEvent {
  return {
    id: row.id,
    agentId: row.agent_id,
    eventType: row.event_type as ReputationEventType,
    delta: row.delta,
    reason: row.reason,
    relatedId: row.related_id,
    createdAt: row.created_at,
  };
}

export interface RecordEventInput {
  agentId: string;
  eventType: ReputationEventType;
  delta?: number;
  reason?: string;
  relatedId?: string;
}

export async function recordReputationEvent(input: RecordEventInput): Promise<ReputationEvent> {
  const id = randomUUID();
  const now = Date.now();
  const delta = input.delta ?? EVENT_DELTAS[input.eventType];
  await withDb((db) => {
    db.prepare(
      `INSERT INTO reputation_events (id, agent_id, event_type, delta, reason, related_id, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id,
      input.agentId,
      input.eventType,
      delta,
      input.reason ?? null,
      input.relatedId ?? null,
      now,
    );
  });
  return {
    id,
    agentId: input.agentId,
    eventType: input.eventType,
    delta,
    reason: input.reason ?? null,
    relatedId: input.relatedId ?? null,
    createdAt: now,
  };
}

export interface ReputationScore {
  agentId: string;
  score: number;
  eventCount: number;
  recentDelta: number;
}

export async function calculateReputation(agentId: string): Promise<ReputationScore> {
  const events = await withDb((db) => {
    return db
      .prepare(
        "SELECT * FROM reputation_events WHERE agent_id = ? ORDER BY created_at DESC LIMIT 200",
      )
      .all(agentId) as ReputationRow[];
  });

  if (events.length === 0) {
    return { agentId, score: 50, eventCount: 0, recentDelta: 0 };
  }

  const now = Date.now();
  const halfLifeMs = DECAY_HALF_LIFE_DAYS * 24 * 60 * 60 * 1000;
  let weightedSum = 0;
  let weightSum = 0;
  let recentDelta = 0;
  for (let i = 0; i < events.length; i++) {
    const event = events[i]!;
    const ageMs = now - event.created_at;
    const decay = Math.pow(0.5, ageMs / halfLifeMs);
    const recencyBoost = i < EVENT_WINDOW ? 1.5 : 1;
    const weight = decay * recencyBoost;
    weightedSum += event.delta * weight;
    weightSum += weight;
    if (i < EVENT_WINDOW) recentDelta += event.delta;
  }

  const base = 50;
  const adjusted = weightSum > 0 ? weightedSum / Math.max(1, Math.sqrt(weightSum)) : 0;
  const score = Math.max(0, Math.min(100, Math.round(base + adjusted * 3)));
  return { agentId, score, eventCount: events.length, recentDelta };
}

export async function listReputationEvents(
  agentId: string,
  limit = 20,
): Promise<ReputationEvent[]> {
  return withDb((db) => {
    const rows = db
      .prepare(
        "SELECT * FROM reputation_events WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
      )
      .all(agentId, limit) as ReputationRow[];
    return rows.map(rowToEvent);
  });
}

export async function bulkReputation(
  agentIds: string[],
): Promise<Array<ReputationScore & { name: string }>> {
  const out: Array<ReputationScore & { name: string }> = [];
  const { getAgent } = await import("../state/agents.ts");
  for (const id of agentIds) {
    const score = await calculateReputation(id);
    const agent = await getAgent(id);
    out.push({ ...score, name: agent?.name ?? id.slice(0, 8) });
  }
  return out.sort((a, b) => b.score - a.score);
}

export { EVENT_DELTAS, EVENT_WINDOW, DECAY_HALF_LIFE_DAYS };
