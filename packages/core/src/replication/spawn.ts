import { initAgentWorkspace } from "../workspace/index.ts";
import { registerAgent, transferBudget, type CreateAgentInput } from "../state/index.ts";
import { getAgent } from "../state/agents.ts";
import type { AgentConfig, Provider } from "@chimpoe/types";

export interface SpawnChildInput {
  parentId: string;
  name: string;
  genesisPrompt: string;
  provider?: Provider;
  modelId?: string;
  endowmentTokens?: number;
}

export interface SpawnChildResult {
  child: AgentConfig;
  endowmentTransferred: number;
  parentBalanceAfter: number;
}

export async function spawnChild(input: SpawnChildInput): Promise<SpawnChildResult> {
  const parent = await getAgent(input.parentId);
  if (!parent) {
    throw new Error(`Parent agent ${input.parentId} not found`);
  }
  if (parent.status === "dead") {
    throw new Error(`Parent ${parent.name} is dead and cannot reproduce`);
  }

  const provider = input.provider ?? parent.provider;
  const modelId = input.modelId ?? parent.modelId;
  const endowment = Math.max(0, Math.floor(input.endowmentTokens ?? 0));

  const childInput: CreateAgentInput = {
    name: input.name,
    provider,
    modelId,
    genesisPrompt: input.genesisPrompt,
    parentId: parent.id,
    generation: parent.generation + 1,
  };
  const child = await registerAgent(childInput);

  await initAgentWorkspace({
    agentId: child.id,
    agentName: child.name,
    genesisPrompt: child.genesisPrompt,
    createdAt: child.createdAt,
  });

  let parentBalanceAfter = 0;
  if (endowment > 0) {
    const transfer = await transferBudget(parent.id, child.id, endowment).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Endowment transfer failed: ${msg}`);
    });
    parentBalanceAfter = transfer.fromBalance;
  }

  return {
    child,
    endowmentTransferred: endowment,
    parentBalanceAfter,
  };
}
