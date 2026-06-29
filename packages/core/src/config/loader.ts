import { exists, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { type ChimpoeConfig, CHIMPOE_VERSION, getConfigPath, getChimpoeHome } from "@chimpoe/types";

export type LoadedConfig = ChimpoeConfig;

export async function ensureHome(): Promise<void> {
  const home = getChimpoeHome();
  await mkdir(home, { recursive: true });
}

export async function loadConfig(): Promise<LoadedConfig | null> {
  const path = getConfigPath();
  try {
    const file = Bun.file(path);
    if (!(await file.exists())) return null;
    return (await file.json()) as LoadedConfig;
  } catch {
    return null;
  }
}

export async function saveConfig(config: LoadedConfig): Promise<void> {
  const path = getConfigPath();
  await mkdir(dirname(path), { recursive: true });
  await Bun.write(path, JSON.stringify(config, null, 2) + "\n");
}

export async function configExists(): Promise<boolean> {
  const path = getConfigPath();
  return await exists(path);
}

export function makeDefaultConfig(overrides: Partial<LoadedConfig> = {}): LoadedConfig {
  return {
    version: CHIMPOE_VERSION,
    defaultProvider: "openai",
    defaultModel: "gpt-5-mini",
    createdAt: Date.now(),
    ...overrides,
  };
}
