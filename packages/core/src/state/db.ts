import { Database } from "bun:sqlite";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { SCHEMA_SQL, SCHEMA_VERSION } from "./schema.ts";

let sharedDb: Database | null = null;

export interface OpenDbOptions {
  readonly?: boolean;
}

const USER_PSEUDO_AGENT = {
  id: "user",
  name: "user",
  provider: "n/a",
  model_id: "n/a",
  genesis_prompt: "Pseudo-agent representing the human user. Seeded automatically.",
  parent_id: null,
  generation: 0,
  budget_tokens: 0,
  tier: "thriving",
  status: "running",
};

export async function openSharedDb(): Promise<Database> {
  if (sharedDb) return sharedDb;
  const { getSharedStateDbPath } = await import("@chimpoe/types");
  const path = getSharedStateDbPath();
  await mkdir(dirname(path), { recursive: true });
  sharedDb = new Database(path, { create: true });
  sharedDb.exec("PRAGMA journal_mode = WAL;");
  sharedDb.exec("PRAGMA foreign_keys = ON;");
  sharedDb.exec(SCHEMA_SQL);
  sharedDb
    .prepare("INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)")
    .run("schema_version", String(SCHEMA_VERSION));

  const now = Date.now();
  sharedDb
    .prepare(
      `INSERT OR IGNORE INTO agents (id, name, provider, model_id, genesis_prompt, parent_id, generation, budget_tokens, tier, status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, NULL, 0, 0, ?, ?, ?, ?)`,
    )
    .run(
      USER_PSEUDO_AGENT.id,
      USER_PSEUDO_AGENT.name,
      USER_PSEUDO_AGENT.provider,
      USER_PSEUDO_AGENT.model_id,
      USER_PSEUDO_AGENT.genesis_prompt,
      USER_PSEUDO_AGENT.tier,
      USER_PSEUDO_AGENT.status,
      now,
      now,
    );

  return sharedDb;
}

export function closeSharedDb(): void {
  if (sharedDb) {
    sharedDb.close();
    sharedDb = null;
  }
}

export async function withDb<T>(fn: (db: Database) => T): Promise<T> {
  const db = await openSharedDb();
  return fn(db);
}
