import { defineCommand } from "citty";
import * as p from "@clack/prompts";
import color from "picocolors";
import { ensureHome, makeDefaultConfig, saveConfig, loadConfig } from "@chimpoe/core";
import { type ChimpoeConfig, type Provider, CHIMPOE_VERSION, getChimpoeHome } from "@chimpoe/types";

const PROVIDERS: Array<{ value: Provider; label: string; hint: string; defaultModel: string }> = [
  {
    value: "openai",
    label: "OpenAI",
    hint: "gpt-5, gpt-5-mini, gpt-4.1, o-series",
    defaultModel: "gpt-5-mini",
  },
  {
    value: "anthropic",
    label: "Anthropic",
    hint: "claude-opus-4, claude-haiku-4",
    defaultModel: "claude-haiku-4-5",
  },
  {
    value: "glm",
    label: "GLM (Z.AI regular)",
    hint: "glm-4.6, glm-5.2 — pay per token, needs ZAI_API_KEY",
    defaultModel: "glm-4.6",
  },
  {
    value: "glm-coding",
    label: "GLM Coding Plan (Z.AI)",
    hint: "subscription endpoint — needs ZAI_API_KEY (coding plan key)",
    defaultModel: "glm-4.6",
  },
  {
    value: "ollama",
    label: "Ollama (local)",
    hint: "free, no API key needed",
    defaultModel: "llama3.2",
  },
];

const PROVIDER_ENV_KEY: Record<Provider, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
  xai: "XAI_API_KEY",
  groq: "GROQ_API_KEY",
  glm: "ZAI_API_KEY",
  "glm-coding": "ZAI_API_KEY",
  ollama: "",
};

export default defineCommand({
  meta: {
    name: "init",
    description: "Initialize chimpoe at ~/.chimpoe (or $CHIMPOE_HOME)",
  },
  args: {
    force: {
      type: "boolean",
      description: "Overwrite existing config",
    },
  },
  run: async ({ args }) => {
    const home = getChimpoeHome();

    p.intro(color.bgCyan(color.black(` chimpoe v${CHIMPOE_VERSION} `)));

    const existing = await loadConfig();
    if (existing && !args.force) {
      const overwrite = await p.confirm({
        message: `Config already exists at ${color.cyan(home)}. Overwrite?`,
        initialValue: false,
      });
      if (p.isCancel(overwrite) || !overwrite) {
        p.cancel("Init cancelled.");
        process.exit(0);
      }
    }

    const providerChoice = await p.select({
      message: "Pick your default LLM provider",
      initialValue: "openai" as Provider,
      options: PROVIDERS.map((p) => ({ value: p.value, label: p.label, hint: p.hint })),
    });
    if (p.isCancel(providerChoice)) {
      p.cancel("Init cancelled.");
      process.exit(0);
    }
    const provider = providerChoice as Provider;
    const providerMeta = PROVIDERS.find((x) => x.value === provider)!;

    let apiKey: string | undefined;
    if (provider !== "ollama") {
      const envKey = PROVIDER_ENV_KEY[provider];
      const existingKey = process.env[envKey];
      const keyInput = await p.text({
        message: `Paste your ${envKey}${existingKey ? color.dim(` (detected in env, leave blank to use it)`) : ""}:`,
        placeholder: existingKey ? color.dim("<using env var>") : "sk-...",
        validate: (val) => {
          if (!val && !existingKey) return `${envKey} is required for ${providerMeta.label}.`;
          return undefined;
        },
      });
      if (p.isCancel(keyInput)) {
        p.cancel("Init cancelled.");
        process.exit(0);
      }
      apiKey = keyInput || existingKey;
    }

    const modelInput = await p.text({
      message: "Default model",
      initialValue: providerMeta.defaultModel,
      validate: (val) => (val.trim().length === 0 ? "Model is required." : undefined),
    });
    if (p.isCancel(modelInput)) {
      p.cancel("Init cancelled.");
      process.exit(0);
    }
    const defaultModel = modelInput as string;

    await ensureHome();

    if (apiKey && provider !== "ollama") {
      const envKey = PROVIDER_ENV_KEY[provider];
      const envPath = `${home}/.env`;
      const envContent = `${envKey}=${apiKey}\n`;
      await Bun.write(envPath, envContent);
    }

    const config: ChimpoeConfig = makeDefaultConfig({
      defaultProvider: provider,
      defaultModel,
    });
    await saveConfig(config);

    p.outro(color.green(`Done. Config saved to ${color.cyan(home + "/chimpoe.json")}`));
    console.log("");
    console.log(`Next: ${color.cyan("chimpoe chat")} to start your first session.`);
  },
});
