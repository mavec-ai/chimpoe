import { exists, readFile, unlink, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import process from "node:process";
import { getAgentHome } from "@chimpoe/types";
import { getAgent } from "../state/agents.ts";
import { updateAgentStatus } from "../state/agents.ts";

const RUNNER_PATH = new URL("../../../cli/src/agent-runner.ts", import.meta.url).pathname;

export interface SpawnAgentOptions {
  agentId: string;
  chimpoeHome?: string;
  env?: Record<string, string>;
}

export interface SpawnedProcess {
  pid: number;
  agentId: string;
  startedAt: number;
}

function pidPath(agentId: string): string {
  return join(getAgentHome(agentId), ".pid");
}

function logPath(agentId: string): string {
  return join(getAgentHome(agentId), "agent.log");
}

async function isProcessAlive(pid: number): Promise<boolean> {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err !== null && (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

export async function spawnAgent(options: SpawnAgentOptions): Promise<SpawnedProcess> {
  const agent = await getAgent(options.agentId);
  if (!agent) {
    throw new Error(`Agent ${options.agentId} not found`);
  }

  const existing = await readPidFile(options.agentId);
  if (existing && (await isProcessAlive(existing))) {
    throw new Error(
      `Agent ${agent.name} (${agent.id.slice(0, 8)}) is already running (pid ${existing}).`,
    );
  }

  await mkdir(getAgentHome(options.agentId), { recursive: true });
  const logFile = Bun.file(logPath(options.agentId));

  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (typeof v === "string") env[k] = v;
  }
  if (options.chimpoeHome) env.CHIMPOE_HOME = options.chimpoeHome;
  for (const [k, v] of Object.entries(options.env ?? {})) env[k] = v;

  const proc = Bun.spawn({
    cmd: ["bun", RUNNER_PATH, options.agentId],
    cwd: getAgentHome(options.agentId),
    env,
    stdout: logFile,
    stderr: logFile,
    stdin: "ignore",
    detached: true,
  });
  proc.unref?.();

  const pid = proc.pid!;
  await writePidFile(options.agentId, pid);
  await updateAgentStatus(options.agentId, "running");

  return { pid, agentId: options.agentId, startedAt: Date.now() };
}

export async function stopAgent(
  agentId: string,
  options: { force?: boolean; timeoutMs?: number } = {},
): Promise<{ stopped: boolean; pid: number | null; reason: string }> {
  const pid = await readPidFile(agentId);
  if (!pid) {
    return { stopped: false, pid: null, reason: "No PID file — agent not running." };
  }
  if (!(await isProcessAlive(pid))) {
    await unlink(pidPath(agentId)).catch(() => {});
    await updateAgentStatus(agentId, "idle");
    return { stopped: false, pid, reason: "Process already dead." };
  }

  const signal = options.force ? "SIGKILL" : "SIGTERM";
  try {
    process.kill(pid, signal);
  } catch {
    // best effort
  }

  const timeoutMs = options.timeoutMs ?? 5000;
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (!(await isProcessAlive(pid))) break;
    await Bun.sleep(100);
  }

  if (await isProcessAlive(pid)) {
    return {
      stopped: false,
      pid,
      reason: `Process did not exit within ${timeoutMs}ms after ${signal}. Try --force.`,
    };
  }

  await unlink(pidPath(agentId)).catch(() => {});
  await updateAgentStatus(agentId, "idle");
  return { stopped: true, pid, reason: `Stopped (signal ${signal}).` };
}

export interface ProcessInfo {
  agentId: string;
  pid: number;
  startedAt: number;
  alive: boolean;
}

export async function getProcessInfo(agentId: string): Promise<ProcessInfo | null> {
  const pid = await readPidFile(agentId);
  if (!pid) return null;
  const alive = await isProcessAlive(pid);
  return { agentId, pid, startedAt: 0, alive };
}

export async function listProcesses(): Promise<ProcessInfo[]> {
  const { listAgents } = await import("../state/agents.ts");
  const agents = await listAgents();
  const infos: ProcessInfo[] = [];
  for (const a of agents) {
    const info = await getProcessInfo(a.id);
    if (info) infos.push(info);
  }
  return infos;
}

async function readPidFile(agentId: string): Promise<number | null> {
  const path = pidPath(agentId);
  if (!(await exists(path))) return null;
  try {
    const text = await readFile(path, "utf8");
    const pid = Number.parseInt(text.trim(), 10);
    return Number.isFinite(pid) ? pid : null;
  } catch {
    return null;
  }
}

async function writePidFile(agentId: string, pid: number): Promise<void> {
  await writeFile(pidPath(agentId), String(pid) + "\n", { mode: 0o644 });
}
