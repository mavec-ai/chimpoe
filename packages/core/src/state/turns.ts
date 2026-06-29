import { randomUUID } from "node:crypto";
import type { ToolCallRecord, TurnRecord, TurnStatus } from "@chimpoe/types";
import { withDb } from "./db.ts";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface TurnRow {
  id: string;
  agent_id: string;
  session_id: string;
  user_message: string;
  assistant_text: string | null;
  tool_calls_json: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  duration_ms: number | null;
  status: string;
  created_at: number;
}

function rowToRecord(row: TurnRow): TurnRecord {
  return {
    id: row.id,
    agentId: row.agent_id,
    sessionId: row.session_id,
    userMessage: row.user_message,
    assistantText: row.assistant_text,
    toolCalls: row.tool_calls_json ? (JSON.parse(row.tool_calls_json) as ToolCallRecord[]) : [],
    inputTokens: row.input_tokens,
    outputTokens: row.output_tokens,
    durationMs: row.duration_ms,
    status: row.status as TurnStatus,
    createdAt: row.created_at,
  };
}

export interface RecordTurnInput {
  agentId: string;
  sessionId: string;
  userMessage: string;
  assistantText: string | null;
  toolCalls?: ToolCallRecord[];
  inputTokens?: number | null;
  outputTokens?: number | null;
  durationMs?: number | null;
  status?: TurnStatus;
}

export async function recordTurn(input: RecordTurnInput): Promise<TurnRecord> {
  const now = Date.now();
  const id = randomUUID();
  const toolCallsJson =
    input.toolCalls && input.toolCalls.length > 0 ? JSON.stringify(input.toolCalls) : null;

  await withDb((db) => {
    db.prepare(
      `INSERT INTO turns (id, agent_id, session_id, user_message, assistant_text, tool_calls_json, input_tokens, output_tokens, duration_ms, status, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      id,
      input.agentId,
      input.sessionId,
      input.userMessage,
      input.assistantText,
      toolCallsJson,
      input.inputTokens ?? null,
      input.outputTokens ?? null,
      input.durationMs ?? null,
      input.status ?? "complete",
      now,
    );
  });

  return {
    id,
    agentId: input.agentId,
    sessionId: input.sessionId,
    userMessage: input.userMessage,
    assistantText: input.assistantText,
    toolCalls: input.toolCalls ?? [],
    inputTokens: input.inputTokens ?? null,
    outputTokens: input.outputTokens ?? null,
    durationMs: input.durationMs ?? null,
    status: input.status ?? "complete",
    createdAt: now,
  };
}

export async function listTurns(agentId: string, limit = 50): Promise<TurnRecord[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM turns WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?")
      .all(agentId, limit) as TurnRow[];
    return rows.reverse().map(rowToRecord);
  });
}

export async function listTurnsBySession(sessionId: string, limit = 50): Promise<TurnRecord[]> {
  return withDb((db) => {
    const rows = db
      .prepare("SELECT * FROM turns WHERE session_id = ? ORDER BY created_at DESC LIMIT ?")
      .all(sessionId, limit) as TurnRow[];
    return rows.reverse().map(rowToRecord);
  });
}

export async function countTurns(agentId: string): Promise<number> {
  return withDb((db) => {
    const row = db.prepare("SELECT COUNT(*) as n FROM turns WHERE agent_id = ?").get(agentId) as {
      n: number;
    };
    return row.n;
  });
}

export async function getRecentHistory(agentId: string, limit = 10): Promise<ChatMessage[]> {
  const turns = await listTurns(agentId, limit);
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    if (!turn.assistantText || turn.status !== "complete") continue;
    messages.push({ role: "user", content: turn.userMessage });
    messages.push({ role: "assistant", content: turn.assistantText });
  }
  return messages;
}
