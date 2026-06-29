import { Database } from "bun:sqlite";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { SCHEMA_SQL, SCHEMA_VERSION } from "./schema.ts";

let sharedDb: Database | null = null;

export interface OpenDbOptions {
  readonly?: boolean;
}

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
