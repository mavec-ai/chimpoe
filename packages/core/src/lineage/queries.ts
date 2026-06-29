import type { AgentConfig } from "@chimpoe/types";
import { getAgent, listAgents } from "../state/agents.ts";
import { withDb } from "../state/db.ts";

export interface LineageNode {
  agent: AgentConfig;
  parent: LineageNode | null;
  children: LineageNode[];
}

export async function getParent(agentId: string): Promise<AgentConfig | null> {
  const agent = await getAgent(agentId);
  if (!agent || !agent.parentId) return null;
  if (agent.parentId === "user") return null;
  return getAgent(agent.parentId);
}

export async function getChildren(agentId: string): Promise<AgentConfig[]> {
  const all = await listAgents();
  return all.filter((a) => a.parentId === agentId).sort((a, b) => a.createdAt - b.createdAt);
}

export async function getAncestors(agentId: string): Promise<AgentConfig[]> {
  const chain: AgentConfig[] = [];
  let current: AgentConfig | null = await getAgent(agentId);
  while (current && current.parentId && current.parentId !== "user") {
    const parent: AgentConfig | null = await getAgent(current.parentId);
    if (!parent) break;
    chain.unshift(parent);
    current = parent;
  }
  return chain;
}

export async function getDescendants(agentId: string): Promise<AgentConfig[]> {
  const out: AgentConfig[] = [];
  const queue: string[] = [agentId];
  while (queue.length > 0) {
    const next = queue.shift()!;
    const children = await getChildren(next);
    for (const child of children) {
      out.push(child);
      queue.push(child.id);
    }
  }
  return out.sort((a, b) => a.createdAt - b.createdAt);
}

export async function getRoots(): Promise<AgentConfig[]> {
  const all = await listAgents();
  return all.filter((a) => !a.parentId || a.parentId === "user");
}

export async function getSiblings(agentId: string): Promise<AgentConfig[]> {
  const agent = await getAgent(agentId);
  if (!agent || !agent.parentId) return [];
  const siblings = await getChildren(agent.parentId);
  return siblings.filter((a) => a.id !== agentId);
}

export async function getLineageTree(rootId: string): Promise<LineageNode | null> {
  const root = await getAgent(rootId);
  if (!root) return null;
  const children = await getChildren(rootId);
  const childNodes: LineageNode[] = [];
  for (const child of children) {
    const sub = await getLineageTree(child.id);
    if (sub) childNodes.push(sub);
  }
  return { agent: root, parent: null, children: childNodes };
}

export interface LineageSummary {
  totalAgents: number;
  roots: number;
  generations: number;
  living: number;
  dead: number;
}

export async function getLineageSummary(): Promise<LineageSummary> {
  const all = await listAgents();
  const roots = all.filter((a) => !a.parentId || a.parentId === "user").length;
  const generations = all.reduce((max, a) => Math.max(max, a.generation), 0);
  const living = all.filter((a) => a.status !== "dead").length;
  const dead = all.filter((a) => a.status === "dead").length;
  return {
    totalAgents: all.length,
    roots,
    generations: generations + 1,
    living,
    dead,
  };
}

export interface BudgetRow {
  agentId: string;
  balance: number;
}

export async function getBudget(agentId: string): Promise<number> {
  return withDb((db) => {
    const row = db.prepare("SELECT budget_tokens FROM agents WHERE id = ?").get(agentId) as
      | { budget_tokens: number }
      | undefined;
    return row?.budget_tokens ?? 0;
  });
}
