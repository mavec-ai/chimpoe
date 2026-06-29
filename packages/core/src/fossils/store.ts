import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { getFossilsDir } from "@chimpoe/types";
import { withDb } from "../state/db.ts";

export interface Fossil {
  id: string;
  agentId: string;
  agentName: string;
  generation: number;
  content: string;
  lineagePath: string | null;
  keywords: string[];
  createdAt: number;
}

interface FossilRow {
  id: string;
  agent_id: string;
  agent_name: string;
  generation: number;
  content: string;
  lineage_path: string | null;
  keywords_json: string | null;
  created_at: number;
}

function rowToFossil(row: FossilRow): Fossil {
  return {
    id: row.id,
    agentId: row.agent_id,
    agentName: row.agent_name,
    generation: row.generation,
    content: row.content,
    lineagePath: row.lineage_path,
    keywords: row.keywords_json ? (JSON.parse(row.keywords_json) as string[]) : [],
    createdAt: row.created_at,
  };
}

export interface SaveFossilInput {
  agentId: string;
  agentName: string;
  generation: number;
  content: string;
  lineagePath?: string[];
  keywords?: string[];
}

export async function saveFossil(input: SaveFossilInput): Promise<Fossil> {
  const id = randomUUID();
  const now = Date.now();
  const lineagePath = input.lineagePath ? input.lineagePath.join("→") : null;
  const keywords = input.keywords ?? [];

  await withDb((db) => {
    db.prepare(
      `INSERT INTO fossils (id, agent_id, agent_name, generation, content, lineage_path, keywords_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(agent_id) DO UPDATE SET
         content = excluded.content,
         keywords_json = excluded.keywords_json,
         lineage_path = excluded.lineage_path`,
    ).run(
      id,
      input.agentId,
      input.agentName,
      input.generation,
      input.content,
      lineagePath,
      keywords.length > 0 ? JSON.stringify(keywords) : null,
      now,
    );
  });

  try {
    const fossilsDir = getFossilsDir();
    await mkdir(fossilsDir, { recursive: true });
    await Bun.write(join(fossilsDir, `${input.agentId}.md`), input.content);
  } catch {
    // file persistence is best-effort; DB is canonical
  }

  return {
    id,
    agentId: input.agentId,
    agentName: input.agentName,
    generation: input.generation,
    content: input.content,
    lineagePath,
    keywords,
    createdAt: now,
  };
}

export async function getFossilByAgent(agentId: string): Promise<Fossil | null> {
  return withDb((db) => {
    const row = db
      .prepare("SELECT * FROM fossils WHERE agent_id = ?")
      .get(agentId) as FossilRow | null;
    return row ? rowToFossil(row) : null;
  });
}

export async function getFossil(id: string): Promise<Fossil | null> {
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM fossils WHERE id = ?").get(id) as FossilRow | null;
    return row ? rowToFossil(row) : null;
  });
}

export async function listFossils(limit = 50): Promise<Fossil[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM fossils ORDER BY created_at DESC LIMIT ?")
      .all(limit) as FossilRow[];
    return rows.map(rowToFossil);
  });
}

export interface SearchFossilsOptions {
  keywords?: string[];
  minGeneration?: number;
  maxGeneration?: number;
  agentIds?: string[];
  limit?: number;
}

export async function searchFossils(
  query: string,
  options: SearchFossilsOptions = {},
): Promise<Fossil[]> {
  const limit = Math.min(options.limit ?? 10, 50);
  const queryKeywords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length >= 3);

  if (queryKeywords.length === 0) return listFossils(limit);

  return withDb((db) => {
    let sql = "SELECT * FROM fossils WHERE ";
    const params: (string | number)[] = [];
    const clauses: string[] = [];
    for (const kw of queryKeywords) {
      clauses.push("(LOWER(content) LIKE ?)");
      params.push(`%${kw}%`);
    }
    sql += clauses.join(" OR ");
    if (options.agentIds && options.agentIds.length > 0) {
      sql += ` AND agent_id IN (${options.agentIds.map(() => "?").join(",")})`;
      params.push(...options.agentIds);
    }
    if (options.minGeneration !== undefined) {
      sql += " AND generation >= ?";
      params.push(options.minGeneration);
    }
    if (options.maxGeneration !== undefined) {
      sql += " AND generation <= ?";
      params.push(options.maxGeneration);
    }
    sql += " ORDER BY generation DESC, created_at DESC LIMIT ?";
    params.push(limit);
    const rows = db.prepare(sql).all(...params) as FossilRow[];
    return rows.map(rowToFossil);
  });
}

export async function countFossils(): Promise<number> {
  return withDb((db) => {
    const row = db.prepare("SELECT COUNT(*) as n FROM fossils").get() as { n: number };
    return row.n;
  });
}

export async function deleteFossil(id: string): Promise<void> {
  await withDb((db) => {
    db.prepare("DELETE FROM fossils WHERE id = ?").run(id);
  });
}
