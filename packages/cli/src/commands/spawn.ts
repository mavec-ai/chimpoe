import { defineCommand } from "citty";
import * as p from "@clack/prompts";
import color from "picocolors";
import { initAgentWorkspace, loadConfig, registerAgent } from "@chimpoe/core";
import { type Provider } from "@chimpoe/types";

interface ProviderOption {
  value: Provider;
  label: string;
  hint: string;
  defaultModel: string;
}

const PROVIDER_OPTIONS: ProviderOption[] = [
  { value: "openai", label: "OpenAI", hint: "gpt-5, gpt-5-mini, ...", defaultModel: "gpt-5-mini" },
  {
    value: "anthropic",
    label: "Anthropic",
    hint: "claude-opus-4, claude-haiku-4, ...",
    defaultModel: "claude-haiku-4-5",
  },
  { value: "ollama", label: "Ollama (local)", hint: "free, no API key", defaultModel: "llama3.2" },
];

export default defineCommand({
  meta: {
    name: "spawn",
    description: "Create a new persistent agent with its own workspace",
  },
  args: {
    name: {
      type: "string",
      description: "Human-readable agent name",
    },
    genesis: {
      type: "string",
      description: "Genesis prompt (the agent's DNA)",
    },
    provider: {
      type: "string",
      description: "LLM provider (openai, anthropic, ollama)",
    },
    model: {
      type: "string",
      description: "Model id (e.g. gpt-5-mini)",
    },
    parent: {
      type: "string",
      description: "Parent agent id (for reproduction)",
    },
  },
  run: async ({ args }) => {
    const config = await loadConfig();
    if (!config) {
      console.error(
        color.red("No chimpoe config. Run ") + color.cyan("chimpoe init") + color.red(" first."),
      );
      process.exit(1);
    }

    p.intro(color.bgCyan(color.black(" chimpoe spawn ")));

    const name =
      args.name ??
      (await p.text({
        message: "Agent name",
        placeholder: "e.g. coder, devops, qa",
        validate: (v) => (v.trim().length === 0 ? "Name is required." : undefined),
      }));
    if (p.isCancel(name)) {
      p.cancel("Cancelled.");
      process.exit(0);
    }

    const providerValue = (args.provider ?? config.defaultProvider) as Provider;
    if (!args.provider) {
      const pick = await p.select({
        message: "Provider",
        initialValue: providerValue,
        options: PROVIDER_OPTIONS.map((o) => ({ value: o.value, label: o.label, hint: o.hint })),
      });
      if (p.isCancel(pick)) {
        p.cancel("Cancelled.");
        process.exit(0);
      }
    }
    const providerMeta = PROVIDER_OPTIONS.find((x) => x.value === providerValue);

    const modelId =
      args.model ??
      (await p.text({
        message: "Model",
        initialValue: providerMeta?.defaultModel ?? config.defaultModel,
      }));
    if (p.isCancel(modelId)) {
      p.cancel("Cancelled.");
      process.exit(0);
    }

    const genesis =
      args.genesis ??
      (await p.text({
        message: "Genesis prompt (the agent's purpose/identity)",
        placeholder: "You are ...",
        validate: (v) => (v.trim().length === 0 ? "Genesis prompt is required." : undefined),
      }));
    if (p.isCancel(genesis)) {
      p.cancel("Cancelled.");
      process.exit(0);
    }

    const agent = await registerAgent({
      name: String(name),
      provider: providerValue,
      modelId: String(modelId),
      genesisPrompt: String(genesis),
      parentId: args.parent ?? null,
      generation: args.parent ? 1 : 0,
    });

    await initAgentWorkspace({
      agentId: agent.id,
      genesisPrompt: agent.genesisPrompt,
    });

    p.outro(color.green(`Agent spawned.`));
    console.log("");
    console.log(`  id         ${color.cyan(agent.id)}`);
    console.log(`  name       ${agent.name}`);
    console.log(`  provider   ${agent.provider} (${agent.modelId})`);
    console.log(`  generation ${agent.generation}`);
    console.log(`  workspace  ${color.dim(`~/.chimpoe/agents/${agent.id}/`)}`);
    console.log("");
    console.log(`Next: ${color.cyan(`chimpoe chat ${agent.id}`)}`);
  },
});
