import { defineCommand } from "citty";
import color from "picocolors";
import { type Provider, getChimpoeHome, getConfigPath, getSharedStateDbPath } from "@chimpoe/types";
import { exists } from "node:fs/promises";

export default defineCommand({
  meta: {
    name: "doctor",
    description: "Diagnose chimpoe setup — home, config, db, env",
  },
  args: {},
  run: async () => {
    const home = getChimpoeHome();
    const configPath = getConfigPath();
    const dbPath = getSharedStateDbPath();
    console.log(color.cyan("chimpoe doctor"));
    console.log(color.gray("-".repeat(60)));

    console.log(`home:           ${home}`);
    console.log(`config:         ${configPath}`);
    console.log(`state db:       ${dbPath}`);
    console.log("");

    const homeExists = await exists(home);
    printCheck("home directory exists", homeExists, home);

    const configExists = await exists(configPath);
    printCheck("config file exists", configExists, configPath);
    if (configExists) {
      try {
        const cfg = await Bun.file(configPath).json();
        console.log(
          color.dim(`  defaultProvider: ${cfg.defaultProvider}, defaultModel: ${cfg.defaultModel}`),
        );
      } catch {
        console.log(color.red("  config invalid JSON"));
      }
    }

    const dbExists = await exists(dbPath);
    printCheck("state db exists", dbExists, dbPath);

    console.log("");
    console.log(color.yellow("Environment:"));
    const envChecks: Array<{ key: string; label: string }> = [
      { key: "OPENAI_API_KEY", label: "OpenAI" },
      { key: "ANTHROPIC_API_KEY", label: "Anthropic" },
      { key: "ZAI_API_KEY", label: "GLM (Z.AI)" },
      { key: "GOOGLE_API_KEY", label: "Google" },
      { key: "XAI_API_KEY", label: "xAI" },
      { key: "GROQ_API_KEY", label: "Groq" },
      { key: "OPENROUTER_API_KEY", label: "OpenRouter" },
    ];
    for (const env of envChecks) {
      const value = process.env[env.key];
      if (value && value.length > 0) {
        console.log(
          `  ${color.green("✓")} ${env.label.padEnd(12)} ${color.dim(`(${env.key} set, ${value.length} chars)`)}`,
        );
      } else {
        console.log(`  ${color.gray("○")} ${env.label.padEnd(12)} ${color.dim("(not set)")}`);
      }
    }
    const ollama = process.env.OLLAMA_BASE_URL;
    if (ollama) {
      console.log(`  ${color.green("✓")} Ollama        ${color.dim(`(${ollama})`)}`);
    }

    console.log("");
    const envHome = process.env.CHIMPOE_HOME;
    if (envHome) {
      console.log(color.yellow(`CHIMPOE_HOME override: ${envHome}`));
    }

    let problems = 0;
    if (!homeExists) problems++;
    if (!configExists) problems++;
    if (!dbExists && configExists) problems++;
    const anyProvider = envChecks.some((e) => process.env[e.key]);
    if (!anyProvider && !ollama) problems++;

    console.log("");
    if (problems === 0) {
      console.log(color.green("●") + " all checks passed");
    } else {
      console.log(
        color.yellow("○") + ` ${problems} issue(s) found. Run "chimpoe init" if config is missing.`,
      );
    }
  },
});

function printCheck(label: string, ok: boolean, path: string): void {
  const mark = ok ? color.green("✓") : color.red("✗");
  console.log(`  ${mark} ${label.padEnd(22)} ${color.dim(path)}`);
}

void ({} as Provider);
