import { randomUUID } from "node:crypto";
import { withDb } from "../state/db.ts";
import { getAgent, listAgents, updateAgentStatus } from "../state/agents.ts";
import { calculateReputation } from "../reputation/index.ts";
import { distillAgent } from "../fossils/index.ts";

export interface CullThresholds {
  reputationFloor: number;
  sustainedHours: number;
  bottomPercentFloor: number;
}

export const DEFAULT_THRESHOLDS: CullThresholds = {
  reputationFloor: 25,
  sustainedHours: 24,
  bottomPercentFloor: 10,
};

export async function protectAgent(agentId: string, reason: string): Promise<void> {
  await withDb((db) => {
    db.prepare(
      "INSERT OR REPLACE INTO protected_agents (agent_id, reason, protected_at) VALUES (?, ?, ?)",
    ).run(agentId, reason, Date.now());
  });
}

export async function unprotectAgent(agentId: string): Promise<void> {
  await withDb((db) => {
    db.prepare("DELETE FROM protected_agents WHERE agent_id = ?").run(agentId);
  });
}

export async function isProtected(agentId: string): Promise<boolean> {
  return withDb((db) => {
    const row = db.prepare("SELECT 1 FROM protected_agents WHERE agent_id = ?").get(agentId) as
      | { 1: number }
      | undefined;
    return !!row;
  });
}

export async function listProtected(): Promise<
  Array<{ agentId: string; reason: string; protectedAt: number }>
> {
  return withDb((db) => {
    const rows = db
      .prepare(
        "SELECT agent_id, reason, protected_at FROM protected_agents ORDER BY protected_at DESC",
      )
      .all() as Array<{ agent_id: string; reason: string; protected_at: number }>;
    return rows.map((r) => ({
      agentId: r.agent_id,
      reason: r.reason,
      protectedAt: r.protected_at,
    }));
  });
}

export interface CullCandidate {
  agentId: string;
  name: string;
  reason: string;
  reputation: number;
  budget: number;
  status: string;
  protected: boolean;
}

export interface CullScanResult {
  candidates: CullCandidate[];
  scanned: number;
  culled: CullCandidate[];
  failed: Array<{ agentId: string; reason: string }>;
}

export async function scanCullCandidates(
  thresholds: CullThresholds = DEFAULT_THRESHOLDS,
  options: { bottomPercent?: number } = {},
): Promise<CullCandidate[]> {
  const agents = await listAgents();
  const protectedIds = new Set((await listProtected()).map((p) => p.agentId));
  const candidates: CullCandidate[] = [];

  const scored = await Promise.all(
    agents
      .filter((a) => a.status !== "dead")
      .map(async (a) => ({
        agent: a,
        reputation: (await calculateReputation(a.id)).score,
        protected: protectedIds.has(a.id),
      })),
  );

  for (const s of scored) {
    if (s.protected) continue;
    const reason = evaluateReason(s.agent.tier, s.reputation, s.agent.budgetTokens, thresholds);
    if (reason) {
      candidates.push({
        agentId: s.agent.id,
        name: s.agent.name,
        reason,
        reputation: s.reputation,
        budget: s.agent.budgetTokens,
        status: s.agent.status,
        protected: false,
      });
    }
  }

  candidates.sort((a, b) => a.reputation - b.reputation);

  if (options.bottomPercent) {
    const cutoff = Math.floor((options.bottomPercent / 100) * scored.length);
    const bottomIds = new Set(
      [...scored]
        .sort((a, b) => a.reputation - b.reputation)
        .slice(0, cutoff)
        .filter((s) => !s.protected)
        .map((s) => s.agent.id),
    );
    for (const c of candidates) {
      void c;
    }
    for (const s of scored) {
      if (bottomIds.has(s.agent.id) && !candidates.find((c) => c.agentId === s.agent.id)) {
        candidates.push({
          agentId: s.agent.id,
          name: s.agent.name,
          reason: `Bottom ${options.bottomPercent}% by reputation`,
          reputation: s.reputation,
          budget: s.agent.budgetTokens,
          status: s.agent.status,
          protected: false,
        });
      }
    }
  }

  return candidates;
}

function evaluateReason(
  tier: string,
  reputation: number,
  budget: number,
  thresholds: CullThresholds,
): string | null {
  if (tier === "dead") return null;
  if (reputation < thresholds.reputationFloor && budget < 1000) {
    return `Low reputation (${reputation} < ${thresholds.reputationFloor}) and low budget (${budget} < 1000)`;
  }
  if (reputation < 10) {
    return `Reputation critically low (${reputation} < 10)`;
  }
  return null;
}

export async function executeCull(
  candidates: CullCandidate[],
  options: { dryRun?: boolean; distill?: boolean } = {},
): Promise<CullScanResult> {
  const culled: CullCandidate[] = [];
  const failed: Array<{ agentId: string; reason: string }> = [];

  for (const candidate of candidates) {
    if (await isProtected(candidate.agentId)) {
      failed.push({ agentId: candidate.agentId, reason: "Agent is protected." });
      continue;
    }
    if (options.dryRun) {
      culled.push(candidate);
      continue;
    }
    try {
      if (options.distill !== false) {
        try {
          await distillAgent(candidate.agentId);
        } catch {
          // best-effort
        }
      }
      await updateAgentStatus(candidate.agentId, "dead", "dead");
      culled.push(candidate);
    } catch (err) {
      failed.push({
        agentId: candidate.agentId,
        reason: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return {
    candidates,
    scanned: candidates.length,
    culled,
    failed,
  };
}

export interface SelfCullDecision {
  shouldSelfCull: boolean;
  reason: string;
  reputation: number;
  tier: string;
  budget: number;
}

export async function checkSelfCull(
  agentId: string,
  thresholds: CullThresholds = DEFAULT_THRESHOLDS,
): Promise<SelfCullDecision> {
  const agent = await getAgent(agentId);
  if (!agent) {
    return {
      shouldSelfCull: true,
      reason: "Agent not found in registry",
      reputation: 0,
      tier: "dead",
      budget: 0,
    };
  }
  if (agent.status === "dead") {
    return {
      shouldSelfCull: false,
      reason: "Already dead",
      reputation: 0,
      tier: agent.tier,
      budget: agent.budgetTokens,
    };
  }
  if (await isProtected(agentId)) {
    return {
      shouldSelfCull: false,
      reason: "Protected — exempt from cull",
      reputation: 0,
      tier: agent.tier,
      budget: agent.budgetTokens,
    };
  }
  const reputation = (await calculateReputation(agentId)).score;
  const reason = evaluateReason(agent.tier, reputation, agent.budgetTokens, thresholds);
  return {
    shouldSelfCull: !!reason,
    reason: reason ?? "ok",
    reputation,
    tier: agent.tier,
    budget: agent.budgetTokens,
  };
}

void randomUUID;
