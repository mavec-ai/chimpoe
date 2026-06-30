import { randomUUID } from "node:crypto";
import { mkdir, readdir, rm, writeFile, stat, readFile } from "node:fs/promises";
import { join, dirname, basename } from "node:path";
import { getAgentHome } from "@chimpoe/types";
import { withDb } from "../state/db.ts";

export type ModificationKind =
  | "install_package"
  | "remove_package"
  | "write_file"
  | "edit_file"
  | "delete_file"
  | "skill_install"
  | "skill_remove"
  | "skill_toggle";

export interface ModificationRecord {
  id: string;
  agentId: string;
  kind: ModificationKind;
  target: string;
  details: Record<string, unknown> | null;
  status: string;
  createdAt: number;
}

interface ModificationRow {
  id: string;
  agent_id: string;
  kind: string;
  target: string;
  details_json: string | null;
  status: string;
  created_at: number;
}

async function logModification(input: {
  agentId: string;
  kind: ModificationKind;
  target: string;
  details?: Record<string, unknown>;
  status?: string;
}): Promise<ModificationRecord> {
  const id = randomUUID();
  const now = Date.now();
  await withDb((db) => {
    db.prepare(
      `INSERT INTO modifications (id, agent_id, kind, target, details_json, status, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id,
      input.agentId,
      input.kind,
      input.target,
      input.details ? JSON.stringify(input.details) : null,
      input.status ?? "applied",
      now,
    );
  });
  return {
    id,
    agentId: input.agentId,
    kind: input.kind,
    target: input.target,
    details: input.details ?? null,
    status: input.status ?? "applied",
    createdAt: now,
  };
}

export async function listModifications(
  agentId: string,
  limit = 20,
): Promise<ModificationRecord[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM modifications WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?")
      .all(agentId, limit) as ModificationRow[];
    return rows.map((r) => ({
      id: r.id,
      agentId: r.agent_id,
      kind: r.kind as ModificationKind,
      target: r.target,
      details: r.details_json ? (JSON.parse(r.details_json) as Record<string, unknown>) : null,
      status: r.status,
      createdAt: r.created_at,
    }));
  });
}

const PROTECTED_FILES = new Set([
  "genesis.md",
  "constitution.md",
  "config.json",
  ".pid",
  "state.db",
]);

function isProtected(relPath: string): boolean {
  const base = basename(relPath);
  if (PROTECTED_FILES.has(base)) return true;
  if (relPath.startsWith("state.db")) return true;
  return false;
}

export interface InstallPackageResult {
  ok: boolean;
  package: string;
  message: string;
  durationMs: number;
}

export async function installPackage(
  agentId: string,
  packageName: string,
  options: { dev?: boolean } = {},
): Promise<InstallPackageResult> {
  const start = Date.now();
  if (!/^[a-zA-Z0-9@/_@.-]+$/.test(packageName) || packageName.length > 200) {
    return { ok: false, package: packageName, message: "Invalid package name.", durationMs: 0 };
  }
  const cwd = getAgentHome(agentId);
  const args = ["add"];
  if (options.dev) args.push("-d");
  args.push(packageName);
  const proc = Bun.spawn({
    cmd: ["bun", ...args],
    cwd,
    stdout: "pipe",
    stderr: "pipe",
  });
  const exitCode = await proc.exited;
  const durationMs = Date.now() - start;
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    await logModification({
      agentId,
      kind: "install_package",
      target: packageName,
      details: { dev: options.dev, exitCode, stderr: stderr.slice(0, 500) },
      status: "failed",
    });
    return {
      ok: false,
      package: packageName,
      message: `bun add exit ${exitCode}: ${stderr.slice(0, 200)}`,
      durationMs,
    };
  }
  await logModification({
    agentId,
    kind: "install_package",
    target: packageName,
    details: { dev: options.dev, durationMs },
  });
  return {
    ok: true,
    package: packageName,
    message: `Installed ${packageName}${options.dev ? " (dev)" : ""} in ${durationMs}ms.`,
    durationMs,
  };
}

export interface WriteFileResult {
  ok: boolean;
  path: string;
  bytes: number;
  message: string;
}

export async function writeWorkspaceFile(
  agentId: string,
  relPath: string,
  content: string,
): Promise<WriteFileResult> {
  if (isProtected(relPath)) {
    return {
      ok: false,
      path: relPath,
      bytes: 0,
      message: `Cannot write to protected file: ${relPath}`,
    };
  }
  if (relPath.includes("..") || relPath.startsWith("/")) {
    return {
      ok: false,
      path: relPath,
      bytes: 0,
      message: "Path must be relative and stay within workspace.",
    };
  }
  const fullPath = join(getAgentHome(agentId), relPath);
  await mkdir(dirname(fullPath), { recursive: true });
  await writeFile(fullPath, content, { mode: 0o644 });
  await logModification({
    agentId,
    kind: "write_file",
    target: relPath,
    details: { bytes: content.length },
  });
  return {
    ok: true,
    path: relPath,
    bytes: content.length,
    message: `Wrote ${content.length} bytes to ${relPath}.`,
  };
}

export interface WorkspaceEntry {
  name: string;
  kind: "file" | "dir";
  size: number;
  path: string;
}

export async function listWorkspaceFiles(agentId: string, subdir = "."): Promise<WorkspaceEntry[]> {
  if (subdir.includes("..")) return [];
  const fullDir = join(getAgentHome(agentId), subdir);
  try {
    const entries = await readdir(fullDir, { withFileTypes: true });
    const out: WorkspaceEntry[] = [];
    for (const entry of entries) {
      const fullPath = join(fullDir, entry.name);
      let size = 0;
      if (entry.isFile()) {
        try {
          size = (await stat(fullPath)).size;
        } catch {
          // unreadable
        }
      }
      out.push({
        name: entry.name,
        kind: entry.isFile() ? "file" : "dir",
        size,
        path: subdir === "." ? entry.name : `${subdir}/${entry.name}`,
      });
    }
    return out.sort((a, b) => {
      if (a.kind === "dir" && b.kind !== "dir") return -1;
      if (a.kind !== "dir" && b.kind === "dir") return 1;
      return a.name.localeCompare(b.name);
    });
  } catch {
    return [];
  }
}

export async function readWorkspaceFile(
  agentId: string,
  relPath: string,
): Promise<{ ok: boolean; content: string; message: string }> {
  if (relPath.includes("..")) {
    return { ok: false, content: "", message: "Path must stay within workspace." };
  }
  const fullPath = join(getAgentHome(agentId), relPath);
  try {
    const content = await readFile(fullPath, "utf8");
    return { ok: true, content, message: `Read ${content.length} bytes.` };
  } catch {
    return { ok: false, content: "", message: `File not found: ${relPath}` };
  }
}

export async function deleteWorkspaceFile(
  agentId: string,
  relPath: string,
): Promise<{ ok: boolean; message: string }> {
  if (isProtected(relPath)) {
    return { ok: false, message: `Cannot delete protected file: ${relPath}` };
  }
  if (relPath.includes("..")) {
    return { ok: false, message: "Path must stay within workspace." };
  }
  const fullPath = join(getAgentHome(agentId), relPath);
  try {
    await rm(fullPath, { force: true });
    await logModification({
      agentId,
      kind: "delete_file",
      target: relPath,
    });
    return { ok: true, message: `Deleted ${relPath}.` };
  } catch {
    return { ok: false, message: `Could not delete ${relPath}.` };
  }
}
