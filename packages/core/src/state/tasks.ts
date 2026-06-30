import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";
import { adjustBudget } from "./agents.ts";
import { recordReputationEvent } from "../reputation/index.ts";

export type TaskStatus =
  | "pending"
  | "claimed"
  | "in_progress"
  | "completed"
  | "failed"
  | "abandoned";

export interface Task {
  id: string;
  runId: string | null;
  fromAgentId: string;
  assigneeId: string | null;
  status: TaskStatus;
  prompt: string;
  rewardTokens: number;
  difficulty: number;
  metadata: Record<string, unknown> | null;
  createdAt: number;
  claimedAt: number | null;
  completedAt: number | null;
  resultText: string | null;
}

interface TaskRow {
  id: string;
  run_id: string | null;
  from_agent_id: string;
  assignee_id: string | null;
  status: string;
  prompt: string;
  reward_tokens: number;
  difficulty: number;
  metadata_json: string | null;
  created_at: number;
  claimed_at: number | null;
  completed_at: number | null;
  result_text: string | null;
}

function rowToTask(row: TaskRow): Task {
  return {
    id: row.id,
    runId: row.run_id,
    fromAgentId: row.from_agent_id,
    assigneeId: row.assignee_id,
    status: row.status as TaskStatus,
    prompt: row.prompt,
    rewardTokens: row.reward_tokens,
    difficulty: row.difficulty,
    metadata: row.metadata_json ? (JSON.parse(row.metadata_json) as Record<string, unknown>) : null,
    createdAt: row.created_at,
    claimedAt: row.claimed_at,
    completedAt: row.completed_at,
    resultText: row.result_text,
  };
}

export interface CreateTaskInput {
  fromAgentId?: string;
  assigneeId?: string;
  prompt: string;
  rewardTokens?: number;
  difficulty?: number;
  runId?: string;
  metadata?: Record<string, unknown>;
}

export async function createTask(input: CreateTaskInput): Promise<Task> {
  const id = randomUUID();
  const now = Date.now();
  await withDb((db) => {
    db.prepare(
      `INSERT INTO tasks (id, run_id, from_agent_id, assignee_id, status, prompt, reward_tokens, difficulty, metadata_json, created_at, claimed_at, completed_at, result_text)
       VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, NULL, NULL, NULL)`,
    ).run(
      id,
      input.runId ?? null,
      input.fromAgentId ?? "user",
      input.assigneeId ?? null,
      input.prompt,
      Math.max(0, Math.floor(input.rewardTokens ?? 0)),
      Math.max(1, Math.min(10, input.difficulty ?? 3)),
      input.metadata ? JSON.stringify(input.metadata) : null,
      now,
    );
  });
  return {
    id,
    runId: input.runId ?? null,
    fromAgentId: input.fromAgentId ?? "user",
    assigneeId: input.assigneeId ?? null,
    status: "pending",
    prompt: input.prompt,
    rewardTokens: Math.max(0, Math.floor(input.rewardTokens ?? 0)),
    difficulty: Math.max(1, Math.min(10, input.difficulty ?? 3)),
    metadata: input.metadata ?? null,
    createdAt: now,
    claimedAt: null,
    completedAt: null,
    resultText: null,
  };
}

export async function getTask(taskId: string): Promise<Task | null> {
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM tasks WHERE id = ?").get(taskId) as TaskRow | undefined;
    return row ? rowToTask(row) : null;
  });
}

export async function listTasks(
  options: { status?: TaskStatus; assigneeId?: string; runId?: string; limit?: number } = {},
): Promise<Task[]> {
  return withDb((db) => {
    let sql = "SELECT * FROM tasks WHERE 1=1";
    const params: (string | number)[] = [];
    if (options.status) {
      sql += " AND status = ?";
      params.push(options.status);
    }
    if (options.assigneeId) {
      sql += " AND assignee_id = ?";
      params.push(options.assigneeId);
    }
    if (options.runId) {
      sql += " AND run_id = ?";
      params.push(options.runId);
    }
    sql += " ORDER BY created_at DESC LIMIT ?";
    params.push(options.limit ?? 50);
    const rows = db.prepare(sql).all(...params) as TaskRow[];
    return rows.map(rowToTask);
  });
}

export interface ClaimResult {
  claimed: boolean;
  task: Task | null;
  message: string;
}

export async function claimTask(taskId: string, agentId: string): Promise<ClaimResult> {
  const now = Date.now();
  return withDb((db) => {
    const row = db.prepare("SELECT * FROM tasks WHERE id = ?").get(taskId) as TaskRow | undefined;
    if (!row) return { claimed: false, task: null, message: "Task not found." };
    const task = rowToTask(row);
    if (task.status !== "pending") {
      return { claimed: false, task, message: `Task is ${task.status}, cannot claim.` };
    }
    if (task.assigneeId && task.assigneeId !== agentId) {
      return { claimed: false, task, message: "Task is assigned to another agent." };
    }
    const info = db
      .prepare(
        "UPDATE tasks SET status = 'claimed', assignee_id = ?, claimed_at = ? WHERE id = ? AND status = 'pending'",
      )
      .run(agentId, now, taskId);
    if (info.changes === 0) {
      return { claimed: false, task, message: "Race: task was claimed by another agent." };
    }
    const updated = rowToTask(
      db.prepare("SELECT * FROM tasks WHERE id = ?").get(taskId) as TaskRow,
    );
    return { claimed: true, task: updated, message: `Task ${taskId.slice(0, 8)} claimed.` };
  });
}

export interface CompleteResult {
  completed: boolean;
  task: Task | null;
  rewardTransferred: number;
  message: string;
}

export async function completeTask(
  taskId: string,
  agentId: string,
  resultText: string,
): Promise<CompleteResult> {
  const now = Date.now();
  const task = await getTask(taskId);
  if (!task) {
    return { completed: false, task: null, rewardTransferred: 0, message: "Task not found." };
  }
  if (task.assigneeId !== agentId) {
    return {
      completed: false,
      task,
      rewardTransferred: 0,
      message: "You did not claim this task.",
    };
  }
  if (task.status !== "claimed" && task.status !== "in_progress") {
    return { completed: false, task, rewardTransferred: 0, message: `Task is ${task.status}.` };
  }
  await withDb((db) => {
    db.prepare(
      "UPDATE tasks SET status = 'completed', completed_at = ?, result_text = ? WHERE id = ?",
    ).run(now, resultText, taskId);
  });
  let rewardTransferred = 0;
  if (task.rewardTokens > 0) {
    const newBalance = await adjustBudget(agentId, task.rewardTokens);
    rewardTransferred = task.rewardTokens;
    void newBalance;
  }
  await recordReputationEvent({
    agentId,
    eventType: "task_completed",
    reason: `Completed task ${taskId.slice(0, 8)}`,
    relatedId: taskId,
  });
  const updated = await getTask(taskId);
  return {
    completed: true,
    task: updated,
    rewardTransferred,
    message: `Task ${taskId.slice(0, 8)} completed (+${rewardTransferred} tokens, +5 reputation).`,
  };
}

export async function failTask(
  taskId: string,
  agentId: string,
  reason: string,
): Promise<{ failed: boolean; message: string }> {
  const task = await getTask(taskId);
  if (!task) return { failed: false, message: "Task not found." };
  if (task.assigneeId !== agentId) return { failed: false, message: "Not your task." };
  if (task.status === "completed" || task.status === "failed") {
    return { failed: false, message: `Task already ${task.status}.` };
  }
  await withDb((db) => {
    db.prepare(
      "UPDATE tasks SET status = 'failed', completed_at = ?, result_text = ? WHERE id = ?",
    ).run(Date.now(), `FAILED: ${reason}`, taskId);
  });
  await recordReputationEvent({
    agentId,
    eventType: "task_failed",
    reason: `Failed task ${taskId.slice(0, 8)}: ${reason.slice(0, 80)}`,
    relatedId: taskId,
  });
  return { failed: true, message: `Task ${taskId.slice(0, 8)} failed (-5 reputation).` };
}

export async function abandonTask(
  taskId: string,
): Promise<{ abandoned: boolean; message: string }> {
  const task = await getTask(taskId);
  if (!task) return { abandoned: false, message: "Task not found." };
  if (task.status !== "claimed" && task.status !== "in_progress") {
    return { abandoned: false, message: `Task is ${task.status}, cannot abandon.` };
  }
  await withDb((db) => {
    db.prepare(
      "UPDATE tasks SET status = 'pending', assignee_id = NULL, claimed_at = NULL WHERE id = ?",
    ).run(taskId);
  });
  return { abandoned: true, message: `Task ${taskId.slice(0, 8)} returned to pool.` };
}

export async function countTasks(status?: TaskStatus): Promise<number> {
  return withDb((db) => {
    if (status) {
      const row = db.prepare("SELECT COUNT(*) as n FROM tasks WHERE status = ?").get(status) as {
        n: number;
      };
      return row.n;
    }
    const row = db.prepare("SELECT COUNT(*) as n FROM tasks").get() as { n: number };
    return row.n;
  });
}
