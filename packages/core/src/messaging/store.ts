import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";

export type MessageType = "text" | "task" | "result" | "system";

export interface AgentMessage {
  id: string;
  fromAgentId: string;
  toAgentId: string;
  type: MessageType;
  content: string;
  inReplyTo: string | null;
  metadata: Record<string, unknown> | null;
  createdAt: number;
  readAt: number | null;
}

interface MessageRow {
  id: string;
  from_agent_id: string;
  to_agent_id: string;
  type: string;
  content: string;
  in_reply_to: string | null;
  metadata_json: string | null;
  created_at: number;
  read_at: number | null;
}

function rowToMessage(row: MessageRow): AgentMessage {
  return {
    id: row.id,
    fromAgentId: row.from_agent_id,
    toAgentId: row.to_agent_id,
    type: row.type as MessageType,
    content: row.content,
    inReplyTo: row.in_reply_to,
    metadata: row.metadata_json ? (JSON.parse(row.metadata_json) as Record<string, unknown>) : null,
    createdAt: row.created_at,
    readAt: row.read_at,
  };
}

export interface SendMessageInput {
  fromAgentId: string;
  toAgentId: string;
  content: string;
  type?: MessageType;
  inReplyTo?: string | null;
  metadata?: Record<string, unknown> | null;
}

export async function sendMessage(input: SendMessageInput): Promise<AgentMessage> {
  const id = randomUUID();
  const now = Date.now();
  await withDb((db) => {
    db.prepare(
      `INSERT INTO messages (id, from_agent_id, to_agent_id, type, content, in_reply_to, metadata_json, created_at, read_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)`,
    ).run(
      id,
      input.fromAgentId,
      input.toAgentId,
      input.type ?? "text",
      input.content,
      input.inReplyTo ?? null,
      input.metadata ? JSON.stringify(input.metadata) : null,
      now,
    );
  });
  return {
    id,
    fromAgentId: input.fromAgentId,
    toAgentId: input.toAgentId,
    type: input.type ?? "text",
    content: input.content,
    inReplyTo: input.inReplyTo ?? null,
    metadata: input.metadata ?? null,
    createdAt: now,
    readAt: null,
  };
}

export interface CheckInboxOptions {
  unreadOnly?: boolean;
  limit?: number;
}

export async function checkInbox(
  agentId: string,
  options: CheckInboxOptions = {},
): Promise<AgentMessage[]> {
  const limit = Math.min(options.limit ?? 50, 200);
  return withDb((db) => {
    let sql = "SELECT * FROM messages WHERE to_agent_id = ?";
    if (options.unreadOnly) sql += " AND read_at IS NULL";
    sql += " ORDER BY created_at DESC LIMIT ?";
    const rows = db.prepare(sql).all(agentId, limit) as MessageRow[];
    return rows.reverse().map(rowToMessage);
  });
}

export async function markRead(messageId: string): Promise<void> {
  await withDb((db) => {
    db.prepare("UPDATE messages SET read_at = ? WHERE id = ? AND read_at IS NULL").run(
      Date.now(),
      messageId,
    );
  });
}

export async function markAllRead(agentId: string): Promise<number> {
  return withDb((db) => {
    const info = db
      .prepare("UPDATE messages SET read_at = ? WHERE to_agent_id = ? AND read_at IS NULL")
      .run(Date.now(), agentId);
    return info.changes;
  });
}

export async function countUnread(agentId: string): Promise<number> {
  return withDb((db) => {
    const row = db
      .prepare("SELECT COUNT(*) as n FROM messages WHERE to_agent_id = ? AND read_at IS NULL")
      .get(agentId) as { n: number };
    return row.n;
  });
}

export interface Conversation {
  partnerId: string;
  lastMessage: AgentMessage;
  unreadCount: number;
}

export async function listConversations(agentId: string): Promise<Conversation[]> {
  return withDb((db) => {
    const rows = db
      .prepare(
        `SELECT m.*, (
           SELECT COUNT(*) FROM messages mm
           WHERE mm.from_agent_id = m.other_id AND mm.to_agent_id = ? AND mm.read_at IS NULL
         ) as unread_count
         FROM (
           SELECT *,
             CASE WHEN from_agent_id = ? THEN to_agent_id ELSE from_agent_id END AS other_id,
             ROW_NUMBER() OVER (
               PARTITION BY CASE WHEN from_agent_id = ? THEN to_agent_id ELSE from_agent_id END
               ORDER BY created_at DESC
             ) AS rn
           FROM messages
           WHERE from_agent_id = ? OR to_agent_id = ?
         ) m
         WHERE m.rn = 1
         ORDER BY m.created_at DESC`,
      )
      .all(agentId, agentId, agentId, agentId, agentId) as (MessageRow & {
      unread_count: number;
    })[];
    return rows.map((row) => ({
      partnerId: row.from_agent_id === agentId ? row.to_agent_id : row.from_agent_id,
      lastMessage: rowToMessage(row),
      unreadCount: row.unread_count,
    }));
  });
}
