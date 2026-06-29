#!/usr/bin/env bun
import {
  chargeInference,
  checkInbox,
  createAgent,
  getAgent,
  loadConfig,
  markRead,
  recordReputationEvent,
  sendMessage,
  updateAgentStatus,
} from "@chimpoe/core";
import { getChimpoeHome, getConfigPath } from "@chimpoe/types";

const POLL_INTERVAL_MS = 1500;

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

async function handleMessage(
  agentId: string,
  fromName: string,
  message: { id: string; fromAgentId: string; content: string; type: string },
): Promise<void> {
  const agentRecord = await getAgent(agentId);
  if (!agentRecord) throw new Error(`Agent ${agentId} vanished mid-run`);

  log(agentRecord.name, `← message from ${fromName}: ${message.content.slice(0, 100)}`);

  const agent = await createAgent({
    config: agentRecord,
    extraInstructions:
      `You just received a message from another agent (${fromName}). ` +
      "Respond directly to the content. Keep it concise and actionable. " +
      "If the message asks you to do work, do it and report back. " +
      "If you don't understand or can't help, say so plainly.",
  });

  const prompt =
    message.type === "task"
      ? `Task assignment from ${fromName}:\n\n${message.content}`
      : `Message from ${fromName}:\n\n${message.content}`;

  const result = await agent.generate({
    prompt,
  });

  const responseText = result.text || "(no response)";
  const usage = await result.totalUsage;
  if (usage && (usage.inputTokens || usage.outputTokens)) {
    const charge = await chargeInference({
      agentId,
      modelId: agentRecord.modelId,
      inputTokens: usage.inputTokens ?? 0,
      outputTokens: usage.outputTokens ?? 0,
    });
    log(
      agentRecord.name,
      `burned ${charge.costTokens} tokens (balance ${charge.newBalance}, tier ${charge.newTier}${charge.tierChanged ? " CHANGED" : ""})`,
    );
    if (charge.newTier === "dead") {
      log(agentRecord.name, "budget exhausted, suspending");
      await markRead(message.id);
      await sendMessage({
        fromAgentId: agentId,
        toAgentId: message.fromAgentId,
        content: "(I have run out of budget and must suspend. Top up via 'chimpoe fund'.)",
        type: "result",
        inReplyTo: message.id,
      });
      await updateAgentStatus(agentId, "dead", "dead");
      process.exit(0);
    }
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
  const shutdown = async (signal: string) => {
    if (stopped) return;
    stopped = true;
    log(agentRecord.name, `received ${signal}, shutting down`);
    await updateAgentStatus(agentId, "idle");
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
  process.on("SIGINT", () => void shutdown("SIGINT"));

  while (!stopped) {
    try {
      const inbox = await checkInbox(agentId, { unreadOnly: true, limit: 10 });
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
    await Bun.sleep(POLL_INTERVAL_MS);
  }
}

void main().catch((err) => {
  console.error(err);
  process.exit(1);
});
