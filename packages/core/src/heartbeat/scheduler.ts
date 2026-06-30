import { getAgent } from "../state/agents.ts";
import { calculateTier, getBudgetSnapshot } from "../economy/index.ts";
import { calculateReputation } from "../reputation/index.ts";
import { appendReflection } from "../soul/index.ts";
import { sendMessage } from "../messaging/index.ts";

export type HeartbeatResult =
  | { kind: "no-op"; reason: string }
  | { kind: "budget_warning"; message: string; sentTo: string | null }
  | { kind: "reflection"; note: string }
  | { kind: "idle_too_long"; note: string };

export interface HeartbeatContext {
  agentId: string;
  lastActiveAt: number;
  pollIntervalMs: number;
}

export interface HeartbeatSchedule {
  budgetCheckEveryTicks: number;
  reflectEveryMs: number;
  idleWarnAfterMs: number;
}

export const DEFAULT_SCHEDULE: HeartbeatSchedule = {
  budgetCheckEveryTicks: 8,
  reflectEveryMs: 6 * 60 * 60 * 1000,
  idleWarnAfterMs: 30 * 60 * 1000,
};

export async function runBudgetCheck(
  ctx: HeartbeatContext,
  notifyAgentId?: string,
): Promise<HeartbeatResult> {
  const agent = await getAgent(ctx.agentId);
  if (!agent) return { kind: "no-op", reason: "agent not found" };
  if (agent.status === "dead") return { kind: "no-op", reason: "agent is dead" };
  const snapshot = await getBudgetSnapshot(ctx.agentId);
  if (!snapshot) return { kind: "no-op", reason: "no budget snapshot" };
  const currentTier = calculateTier(snapshot.balance);

  if (currentTier === "conservation" || currentTier === "dormant") {
    const msg =
      currentTier === "dormant"
        ? `Critical: ${agent.name} has ${snapshot.balance} tokens (dormant). Will suspend on next message.`
        : `Warning: ${agent.name} has ${snapshot.balance} tokens (conservation). Seek work or top up.`;
    if (notifyAgentId) {
      await sendMessage({
        fromAgentId: ctx.agentId,
        toAgentId: notifyAgentId,
        content: msg,
        type: "system",
      }).catch(() => {});
    }
    return { kind: "budget_warning", message: msg, sentTo: notifyAgentId ?? null };
  }
  return { kind: "no-op", reason: `tier ${currentTier}, balance ${snapshot.balance}` };
}

export async function runReflection(ctx: HeartbeatContext): Promise<HeartbeatResult> {
  const agent = await getAgent(ctx.agentId);
  if (!agent) return { kind: "no-op", reason: "agent not found" };
  const rep = await calculateReputation(ctx.agentId);
  const idleMs = Date.now() - ctx.lastActiveAt;
  const idleLabel =
    idleMs < 60_000 ? `${Math.floor(idleMs / 1000)}s` : `${Math.floor(idleMs / 60_000)}m`;
  const note =
    `Heartbeat reflection — status ${agent.status}, tier ${agent.tier}, ` +
    `reputation ${rep.score}/100, idle ${idleLabel}. ` +
    `No new work; keeping context warm.`;
  await appendReflection(ctx.agentId, note).catch(() => {});
  return { kind: "reflection", note };
}

export async function runIdleCheck(ctx: HeartbeatContext): Promise<HeartbeatResult> {
  const idleMs = Date.now() - ctx.lastActiveAt;
  if (idleMs < DEFAULT_SCHEDULE.idleWarnAfterMs) {
    return { kind: "no-op", reason: `idle ${Math.floor(idleMs / 1000)}s < threshold` };
  }
  const minutes = Math.floor(idleMs / 60_000);
  return {
    kind: "idle_too_long",
    note: `Idle for ${minutes} minutes. Consider checking inbox or seeking work via message_agent.`,
  };
}

export interface HeartbeatTick {
  budget?: HeartbeatResult;
  reflection?: HeartbeatResult;
  idle?: HeartbeatResult;
}

export async function runHeartbeatTick(
  ctx: HeartbeatContext,
  schedule: HeartbeatSchedule = DEFAULT_SCHEDULE,
  options: { tickCount: number; lastReflectionAt: number; notifyAgentId?: string },
): Promise<HeartbeatTick> {
  const out: HeartbeatTick = {};
  if (options.tickCount > 0 && options.tickCount % schedule.budgetCheckEveryTicks === 0) {
    out.budget = await runBudgetCheck(ctx, options.notifyAgentId);
  }
  if (Date.now() - options.lastReflectionAt >= schedule.reflectEveryMs) {
    out.reflection = await runReflection(ctx);
  }
  out.idle = await runIdleCheck(ctx);
  return out;
}
