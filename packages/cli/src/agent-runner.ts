#!/usr/bin/env bun
import {
  abandonAllClaimedByAgent,
  checkInbox,
  createAgent,
  DEFAULT_SCHEDULE,
  distillAgent,
  getAgent,
  loadConfig,
  markRead,
  recordReputationEvent,
  runHeartbeatTick,
  sendMessage,
  updateAgentStatus,
} from "@chimpoe/core";
import { getAgentHome, getChimpoeHome, getConfigPath } from "@chimpoe/types";
import { join } from "node:path";
import { rm } from "node:fs/promises";

async function cleanupPid(agentId: string): Promise<void> {
  const pidPath = join(getAgentHome(agentId), ".pid");
  try {
    await rm(pidPath, { force: true });
  } catch {
    // best effort
  }
}

const POLL_INTERVAL_MS = 1500;
const MAX_RETRIES = 3;
const RETRY_COOLDOWN_MS = 10_000;
const TRANSIENT_ERROR_PATTERNS = [
  "socket connection was closed",
  "timeout",
  "timed out",
  "ECONNRESET",
  "ECONNREFUSED",
  "ETIMEDOUT",
  "ENOTFOUND",
  "fetch failed",
  "network",
  "socket hang up",
  "connect ETIMEDOUT",
  "operation timed out",
  "Cannot connect to API",
];

function isTransientError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return TRANSIENT_ERROR_PATTERNS.some((p) => msg.includes(p.toLowerCase()));
}

const retryCounts = new Map<string, number>();

async function loadEnvFile(): Promise<void> {
  const envPath = `${getChimpoeHome()}/.env`;
  try {
    const text = await Bun.file(envPath).text();
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq < 0) continue;
      const key = trimmed.slice(0, eq).trim();
      const value = trimmed.slice(eq + 1).trim();
      if (key && !(key in process.env)) process.env[key] = value;
    }
  } catch {
    // env file optional
  }
}

function log(agentName: string, msg: string): void {
  const ts = new Date().toISOString();
  console.log(`[${ts}] [${agentName}] ${msg}`);
}

let cachedAgent: Awaited<ReturnType<typeof createAgent>> | null = null;
let cachedAgentId: string | null = null;

async function handleMessage(
  agentId: string,
  fromName: string,
  message: { id: string; fromAgentId: string; content: string; type: string },
): Promise<void> {
  const agentRecord = await getAgent(agentId);
  if (!agentRecord) throw new Error(`Agent ${agentId} vanished mid-run`);

  const MIN_PROCESSING_BUDGET = 500;
  if (agentRecord.budgetTokens < MIN_PROCESSING_BUDGET) {
    log(
      agentRecord.name,
      `insufficient budget (${agentRecord.budgetTokens} < ${MIN_PROCESSING_BUDGET}), refusing message`,
    );
    await markRead(message.id);
    await sendMessage({
      fromAgentId: agentId,
      toAgentId: message.fromAgentId,
      content: `(Insufficient budget (${agentRecord.budgetTokens} tokens). Fund me via 'chimpoe fund <id> <amount>'.)`,
      type: "result",
      inReplyTo: message.id,
    });
    await updateAgentStatus(agentId, "dead", "dead");
    await abandonAllClaimedByAgent(agentId);
    await cleanupPid(agentId);
    process.exit(0);
  }

  log(agentRecord.name, `← message from ${fromName}: ${message.content.slice(0, 100)}`);

  if (!cachedAgent || cachedAgentId !== agentId) {
    cachedAgent = await createAgent({
      config: agentRecord,
      extraInstructions:
        `You just received a message from another agent (${fromName}). ` +
        "Respond directly to the content. Keep it concise and actionable. " +
        "If the message asks you to do work, do it and report back. " +
        "If you don't understand or can't help, say so plainly.",
    });
    cachedAgentId = agentId;
  }
  cachedAgent.budget.remaining = agentRecord.budgetTokens;

  const prompt =
    message.type === "task"
      ? `Task assignment from ${fromName}:\n\n${message.content}`
      : `Message from ${fromName}:\n\n${message.content}`;

  const result = await cachedAgent.agent.generate({ prompt });
  const responseText = result.text || "(no response)";

  const finalBalance = cachedAgent.budget.remaining;
  log(
    agentRecord.name,
    `burned ${agentRecord.budgetTokens - Math.max(0, finalBalance)} tokens (balance ${Math.max(0, finalBalance)})`,
  );

  if (finalBalance <= 0) {
    log(agentRecord.name, "budget exhausted, distilling fossil before suspend");
    try {
      const distill = await distillAgent(agentId);
      log(
        agentRecord.name,
        `fossil distilled (${distill.sizeBytes} bytes, ${distill.keywords.length} keywords)`,
      );
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      log(agentRecord.name, `fossil distill failed: ${reason}`);
    }
    await markRead(message.id);
    await sendMessage({
      fromAgentId: agentId,
      toAgentId: message.fromAgentId,
      content:
        "(I have run out of budget and must suspend. A fossil of my knowledge has been distilled for descendants.)",
      type: "result",
      inReplyTo: message.id,
    });
    await updateAgentStatus(agentId, "dead", "dead");
    await abandonAllClaimedByAgent(agentId);
    await cleanupPid(agentId);
    process.exit(0);
  }

  await markRead(message.id);
  await sendMessage({
    fromAgentId: agentId,
    toAgentId: message.fromAgentId,
    content: responseText,
    type: "result",
    inReplyTo: message.id,
  });
  await recordReputationEvent({
    agentId,
    eventType: "task_completed",
    reason: `Replied to ${fromName}`,
    relatedId: message.id,
  });
  retryCounts.delete(message.id);

  log(agentRecord.name, `→ reply to ${fromName}: ${responseText.slice(0, 100)}`);
}

async function main(): Promise<void> {
  const agentId = process.argv[2];
  if (!agentId) {
    console.error("Usage: agent-runner.ts <agentId>");
    process.exit(2);
  }

  await loadEnvFile();

  const config = await loadConfig();
  if (!config) {
    console.error(`No config at ${getConfigPath()}. Run "chimpoe init" first.`);
    process.exit(1);
  }

  const agentRecord = await getAgent(agentId);
  if (!agentRecord) {
    console.error(`Agent ${agentId} not found in registry.`);
    process.exit(1);
  }

  await updateAgentStatus(agentId, "running");
  log(agentRecord.name, `daemon started (pid ${process.pid})`);

  let stopped = false;
  let tickCount = 0;
  let lastActiveAt = Date.now();
  let lastReflectionAt = Date.now();
  const shutdown = async (signal: string) => {
    if (stopped) return;
    stopped = true;
    log(agentRecord.name, `received ${signal}, shutting down`);
    await updateAgentStatus(agentId, "idle");
    await cleanupPid(agentId);
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
  process.on("SIGINT", () => void shutdown("SIGINT"));

  while (!stopped) {
    tickCount++;
    try {
      const inbox = await checkInbox(agentId, { unreadOnly: true, limit: 10 });
      if (inbox.length > 0) lastActiveAt = Date.now();
      for (const msg of inbox) {
        const sender = await getAgent(msg.fromAgentId);
        const fromName = sender?.name ?? msg.fromAgentId.slice(0, 8);
        try {
          await handleMessage(agentId, fromName, {
            id: msg.id,
            fromAgentId: msg.fromAgentId,
            content: msg.content,
            type: msg.type,
          });
        } catch (err) {
          const reason = err instanceof Error ? err.message : String(err);
          const transient = isTransientError(err);
          const retries = retryCounts.get(msg.id) ?? 0;

          if (transient && retries < MAX_RETRIES) {
            retryCounts.set(msg.id, retries + 1);
            log(
              agentRecord.name,
              `transient error on ${msg.id.slice(0, 8)} (retry ${retries + 1}/${MAX_RETRIES}): ${reason}`,
            );
            // DON'T mark as read — message stays unread, will be retried next poll
            // But wait longer before next poll to let API recover
            await Bun.sleep(RETRY_COOLDOWN_MS);
            continue;
          }

          // Permanent error or max retries exhausted
          retryCounts.delete(msg.id);
          log(agentRecord.name, `error handling message ${msg.id.slice(0, 8)}: ${reason}`);
          await markRead(msg.id);
          await sendMessage({
            fromAgentId: agentId,
            toAgentId: msg.fromAgentId,
            content: `(error processing your message: ${reason})`,
            type: "result",
            inReplyTo: msg.id,
          });
          await recordReputationEvent({
            agentId,
            eventType: "task_failed",
            reason: `Failed to handle message: ${reason.slice(0, 100)}`,
            relatedId: msg.id,
          });
        }
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      log(agentRecord.name, `poll error: ${reason}`);
    }
    try {
      const tick = await runHeartbeatTick(
        { agentId, lastActiveAt, pollIntervalMs: POLL_INTERVAL_MS },
        DEFAULT_SCHEDULE,
        { tickCount, lastReflectionAt },
      );
      if (tick.budget?.kind === "budget_warning") {
        log(agentRecord.name, `heartbeat: ${tick.budget.message}`);
      }
      if (tick.reflection?.kind === "reflection") {
        lastReflectionAt = Date.now();
        log(agentRecord.name, "heartbeat: wrote reflection to SOUL.md");
      }
      if (tick.idle?.kind === "idle_too_long") {
        log(agentRecord.name, `heartbeat: ${tick.idle.note}`);
      }
      if (tick.selfCull?.shouldSelfCull) {
        log(
          agentRecord.name,
          `heartbeat: SELF-CULL triggered — ${tick.selfCull.reason} (rep=${tick.selfCull.reputation}, bal=${tick.selfCull.budget})`,
        );
        try {
          const { distillAgent } = await import("@chimpoe/core");
          const distill = await distillAgent(agentId);
          log(agentRecord.name, `fossil distilled (${distill.sizeBytes}b) before self-cull`);
        } catch (err) {
          const reason = err instanceof Error ? err.message : String(err);
          log(agentRecord.name, `fossil distill failed during self-cull: ${reason}`);
        }
        await updateAgentStatus(agentId, "dead", "dead");
        await abandonAllClaimedByAgent(agentId);
        await cleanupPid(agentId);
        process.exit(0);
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      log(agentRecord.name, `heartbeat error: ${reason}`);
    }
    await Bun.sleep(POLL_INTERVAL_MS);
  }
}

void main().catch((err) => {
  console.error(err);
  process.exit(1);
});
