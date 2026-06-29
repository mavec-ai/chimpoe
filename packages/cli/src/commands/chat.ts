import { defineCommand } from "citty";
import color from "picocolors";
import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import {
  chargeInference,
  createAgent,
  getRecentHistory,
  loadConfig,
  recordTurn,
  updateAgentStatus,
  type ChatMessage,
  listAgents,
} from "@chimpoe/core";
import { type AgentConfig, type Provider, getConfigPath, getChimpoeHome } from "@chimpoe/types";
import { randomUUID } from "node:crypto";
import { resolveAgent } from "../utils/resolve.ts";

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

async function persistentRepl(agentConfig: AgentConfig): Promise<void> {
  const sessionId = randomUUID();
  const agent = await createAgent({ config: agentConfig });

  await updateAgentStatus(agentConfig.id, "running");

  console.log(
    color.green("•") +
      ` chatting with ${color.cyan(agentConfig.name)} (${agentConfig.provider}/${agentConfig.modelId})`,
  );
  console.log(color.dim(`  session ${sessionId.slice(0, 8)}  ·  Ctrl+C or /exit to leave`));

  const history: ChatMessage[] = await getRecentHistory(agentConfig.id, 10);
  if (history.length > 0) {
    console.log(color.dim(`  resumed ${history.length / 2} prior turn(s) from last session`));
  }
  console.log("");

  const rl = readline.createInterface({ input: stdin, output: stdout, prompt: color.cyan("> ") });
  rl.prompt();

  const onSigint = () => {
    console.log("");
    rl.close();
  };
  process.on("SIGINT", onSigint);

  try {
    for await (const raw of rl) {
      const input = raw.trim();
      if (!input) {
        rl.prompt();
        continue;
      }
      if (input === "/exit" || input === "/quit") {
        break;
      }

      const start = Date.now();
      let assistantText = "";
      process.stdout.write(color.green("•") + " ");
      try {
        const messages = [...history, { role: "user" as const, content: input }];
        const result = await agent.stream({ messages });
        for await (const chunk of result.textStream) {
          process.stdout.write(chunk);
          assistantText += chunk;
        }
        const usage = await result.totalUsage;
        let chargeInfo: { costTokens: number; newBalance: number; newTier: string } | null = null;
        if (usage && (usage.inputTokens || usage.outputTokens)) {
          const charge = await chargeInference({
            agentId: agentConfig.id,
            modelId: agentConfig.modelId,
            inputTokens: usage.inputTokens ?? 0,
            outputTokens: usage.outputTokens ?? 0,
          });
          chargeInfo = charge;
        }
        const balanceStr = chargeInfo
          ? color.dim(` bal=${chargeInfo.newBalance} tier=${chargeInfo.newTier}`)
          : "";
        const usageStr =
          usage && (usage.inputTokens || usage.outputTokens)
            ? color.dim(
                `  [${usage.inputTokens ?? 0}↑ ${usage.outputTokens ?? 0}↓ ${Date.now() - start}ms${balanceStr}]`,
              )
            : color.dim(`  [${Date.now() - start}ms]`);
        process.stdout.write("\n" + usageStr + "\n\n");

        history.push({ role: "user", content: input });
        history.push({ role: "assistant", content: assistantText });

        await recordTurn({
          agentId: agentConfig.id,
          sessionId,
          userMessage: input,
          assistantText,
          inputTokens: usage?.inputTokens ?? null,
          outputTokens: usage?.outputTokens ?? null,
          durationMs: Date.now() - start,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        process.stdout.write(color.red(`\n  error: ${message}\n\n`));
        await recordTurn({
          agentId: agentConfig.id,
          sessionId,
          userMessage: input,
          assistantText: null,
          status: "failed",
          durationMs: Date.now() - start,
        });
      }
      rl.prompt();
    }
  } finally {
    process.off("SIGINT", onSigint);
    await updateAgentStatus(agentConfig.id, "idle");
    console.log(color.dim(`session ${sessionId.slice(0, 8)} ended.`));
  }
}

async function ephemeralTui(agentConfig: AgentConfig): Promise<void> {
  const { runAgentTUI } = await import("@ai-sdk/tui");
  const agent = await createAgent({ config: agentConfig });
  await runAgentTUI({
    title: `chimpoe (${color.cyan(agentConfig.modelId)})`,
    agent: agent as never,
  });
}

export default defineCommand({
  meta: {
    name: "chat",
    description: "Chat with an agent (persistent by id, or ephemeral)",
  },
  args: {
    agentId: {
      type: "positional",
      required: false,
      description: "Agent id (or short prefix). Omit for ephemeral session.",
    },
    genesis: {
      type: "string",
      description: "Genesis prompt (ephemeral mode only)",
    },
    model: {
      type: "string",
      description: "Override model",
    },
    provider: {
      type: "string",
      description: "Override provider",
    },
  },
  run: async ({ args }) => {
    await loadEnvFile();

    if (args.agentId) {
      const idQuery = args.agentId;
      const agents = await listAgents();
      const match = resolveAgent(agents, idQuery);
      if (!match) {
        console.error(color.red(`No agent matching "${idQuery}".`));
        console.error(color.dim(`Run ${color.cyan("chimpoe list")} to see agents.`));
        process.exit(1);
      }
      await persistentRepl(match);
      return;
    }

    const config = await loadConfig();
    if (!config) {
      console.error(color.red("No chimpoe config found."));
      console.error(`Run ${color.cyan("chimpoe init")} first.`);
      console.error(color.dim(`Expected config at ${getConfigPath()}`));
      process.exit(1);
    }

    const provider = (args.provider || config.defaultProvider) as Provider;
    const modelId = args.model || config.defaultModel;
    const genesisPrompt =
      args.genesis ||
      "You are chimpoe, a helpful local-first AI agent. Be concise and direct. Use tools when they help.";

    const ephemeralConfig: AgentConfig = {
      id: "ephemeral",
      name: "ephemeral",
      provider,
      modelId,
      genesisPrompt,
      parentId: null,
      generation: 0,
      budgetTokens: 0,
      tier: "normal",
      status: "running",
      createdAt: Date.now(),
    };
    await ephemeralTui(ephemeralConfig);
  },
});
