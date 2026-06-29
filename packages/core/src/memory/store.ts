import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";

export type MemoryType = "working" | "episodic" | "semantic" | "procedural";

export interface MemoryRecord {
  id: string;
  agentId: string;
  type: MemoryType;
  content: string;
  tags: string[];
  importance: number;
  createdAt: number;
  lastRecalledAt: number | null;
  recallCount: number;
}

interface MemoryRow {
  id: string;
  agent_id: string;
  type: string;
  content: string;
  tags_json: string | null;
  importance: number;
  created_at: number;
  last_recalled_at: number | null;
  recall_count: number;
}

function rowToRecord(row: MemoryRow): MemoryRecord {
  return {
    id: row.id,
    agentId: row.agent_id,
    type: row.type as MemoryType,
    content: row.content,
    tags: row.tags_json ? (JSON.parse(row.tags_json) as string[]) : [],
    importance: row.importance,
    createdAt: row.created_at,
    lastRecalledAt: row.last_recalled_at,
    recallCount: row.recall_count,
  };
}

export interface WriteMemoryInput {
  agentId: string;
  type: MemoryType;
  content: string;
  tags?: string[];
  importance?: number;
}

export async function writeMemory(input: WriteMemoryInput): Promise<MemoryRecord> {
  const id = randomUUID();
  const now = Date.now();
  const tags = input.tags ?? [];
  await withDb((db) => {
    db.prepare(
      `INSERT INTO memories (id, agent_id, type, content, tags_json, importance, created_at, last_recalled_at, recall_count)
       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0)`,
    ).run(
      id,
      input.agentId,
      input.type,
      input.content,
      tags.length > 0 ? JSON.stringify(tags) : null,
      Math.max(0, Math.min(100, input.importance ?? 50)),
      now,
    );
  });
  return {
    id,
    agentId: input.agentId,
    type: input.type,
    content: input.content,
    tags,
    importance: input.importance ?? 50,
    createdAt: now,
    lastRecalledAt: null,
    recallCount: 0,
  };
}

export interface SearchMemoryOptions {
  type?: MemoryType;
  tags?: string[];
  limit?: number;
}

export async function searchMemory(
  agentId: string,
  query: string,
  options: SearchMemoryOptions = {},
): Promise<MemoryRecord[]> {
  const limit = Math.min(options.limit ?? 10, 50);
  const keywords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length >= 2);

  if (keywords.length === 0) {
    return listMemories(agentId, { type: options.type, limit });
  }

  return withDb((db) => {
    let sql = "SELECT * FROM memories WHERE agent_id = ?";
    const params: (string | number)[] = [agentId];
    if (options.type) {
      sql += " AND type = ?";
      params.push(options.type);
    }
    sql += ` AND (${keywords.map(() => "LOWER(content) LIKE ?").join(" OR ")})`;
    for (const kw of keywords) params.push(`%${kw}%`);
    sql += " ORDER BY importance DESC, created_at DESC LIMIT ?";
    params.push(limit);
    const rows = db.prepare(sql).all(...params) as MemoryRow[];
    const records = rows.map(rowToRecord);
    bumpRecall(records.map((r) => r.id));
    return records;
  });
}

export interface ListMemoriesOptions {
  type?: MemoryType;
  tags?: string[];
  limit?: number;
  orderBy?: "recent" | "important" | "recalled";
}

export async function listMemories(
  agentId: string,
  options: ListMemoriesOptions = {},
): Promise<MemoryRecord[]> {
  const limit = Math.min(options.limit ?? 50, 200);
  const order =
    options.orderBy === "important"
      ? "importance DESC, created_at DESC"
      : options.orderBy === "recalled"
        ? "recall_count DESC, importance DESC"
        : "created_at DESC";
  return withDb((db) => {
    let sql = "SELECT * FROM memories WHERE agent_id = ?";
    const params: (string | number)[] = [agentId];
    if (options.type) {
      sql += " AND type = ?";
      params.push(options.type);
    }
    sql += ` ORDER BY ${order} LIMIT ?`;
    params.push(limit);
    const rows = db.prepare(sql).all(...params) as MemoryRow[];
    return rows.map(rowToRecord);
  });
}

export async function getMemory(id: string): Promise<MemoryRecord | null> {
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM memories WHERE id = ?").get(id) as MemoryRow | null;
    if (!row) return null;
    const rec = rowToRecord(row);
    bumpRecall([rec.id]);
    return rec;
  });
}

export async function deleteMemory(id: string): Promise<void> {
  await withDb((db) => {
    db.prepare("DELETE FROM memories WHERE id = ?").run(id);
  });
}

export async function countMemories(agentId: string, type?: MemoryType): Promise<number> {
  return withDb((db) => {
    if (type) {
      const row = db
        .prepare("SELECT COUNT(*) as n FROM memories WHERE agent_id = ? AND type = ?")
        .get(agentId, type) as { n: number };
      return row.n;
    }
    const row = db
      .prepare("SELECT COUNT(*) as n FROM memories WHERE agent_id = ?")
      .get(agentId) as { n: number };
    return row.n;
  });
}

function bumpRecall(ids: string[]): void {
  if (ids.length === 0) return;
  void withDb((db) => {
    const stmt = db.prepare(
      "UPDATE memories SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id = ?",
    );
    const now = Date.now();
    for (const id of ids) stmt.run(now, id);
  });
}

export function formatMemoryForPrompt(m: MemoryRecord): string {
  const tags = m.tags.length > 0 ? ` [${m.tags.join(", ")}]` : "";
  const date = new Date(m.createdAt).toISOString().slice(0, 10);
  return `- (${m.type}/${date}${tags}, imp=${m.importance}) ${m.content}`;
}
