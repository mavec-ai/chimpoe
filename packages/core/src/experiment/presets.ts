import type { ExperimentConfig } from "./config.ts";

export interface Preset {
  name: string;
  description: string;
  build: (overrides?: Partial<ExperimentConfig>) => ExperimentConfig;
}

const baseConfig = (
  name: string,
  agents: ExperimentConfig["agents"],
  extras: Partial<ExperimentConfig> = {},
): ExperimentConfig => ({
  name,
  agents,
  ...extras,
});

export const PRESETS: Record<string, Preset> = {
  garden: {
    name: "garden",
    description:
      "Abundant budget, no selection pressure. Lets agents explore and specialize freely. Good baseline.",
    build: (overrides) =>
      baseConfig(
        overrides?.name ?? "garden",
        overrides?.agents ?? [
          {
            name: "founder",
            genesis:
              "You are the founder of a small colony. Explore what work you find meaningful. " +
              "Spawn children only when you genuinely need a specialist. Be curious.",
            endowment: 200_000,
          },
        ],
        {
          durationMs: 60 * 60 * 1000,
          policy: { autoFundOnConservation: true, blockSelfSpawn: false },
          ...overrides,
        },
      ),
  },
  "high-pressure": {
    name: "high-pressure",
    description:
      "Tight budgets, no auto-fund. Agents must deliver value to survive. Selective regime.",
    build: (overrides) =>
      baseConfig(
        overrides?.name ?? "high-pressure",
        overrides?.agents ?? [
          {
            name: "founder",
            genesis:
              "You are the founder. Budget is scarce. Prioritize ruthlessly, deliver value, " +
              "spawn children only when clearly worth the endowment cost.",
            endowment: 20_000,
          },
        ],
        {
          durationMs: 60 * 60 * 1000,
          policy: { autoFundOnConservation: false, autoCullReputationUnder: 25 },
          ...overrides,
        },
      ),
  },
  "mutation-storm": {
    name: "mutation-storm",
    description:
      "High budget but genesis prompts include a randomization instruction to encourage diverse offspring.",
    build: (overrides) =>
      baseConfig(
        overrides?.name ?? "mutation-storm",
        overrides?.agents ?? [
          {
            name: "founder",
            genesis:
              "You are the founder. Spawn children aggressively with diverse specialties. " +
              "Each child MUST have a meaningfully different genesis. Aim for 5+ children.",
            endowment: 500_000,
          },
        ],
        {
          durationMs: 2 * 60 * 60 * 1000,
          ...overrides,
        },
      ),
  },
  "isolated-comparison": {
    name: "isolated-comparison",
    description:
      "Two unrelated roots in one run. No initial relationship, both funded equally. Use for diff comparison.",
    build: (overrides) =>
      baseConfig(
        overrides?.name ?? "isolated-comparison",
        overrides?.agents ?? [
          {
            name: "alpha",
            genesis: "You are alpha. Outperform beta.",
            endowment: 50_000,
          },
          {
            name: "beta",
            genesis: "You are beta. Outperform alpha.",
            endowment: 50_000,
          },
        ],
        {
          durationMs: 60 * 60 * 1000,
          ...overrides,
        },
      ),
  },
};

export function listPresets(): Preset[] {
  return Object.values(PRESETS);
}

export function getPreset(name: string): Preset | null {
  return PRESETS[name] ?? null;
}

export function resolveConfig(
  source: { preset?: string; overrides?: Partial<ExperimentConfig> } | { config: ExperimentConfig },
): ExperimentConfig {
  if ("config" in source) return source.config;
  const preset = source.preset ? getPreset(source.preset) : null;
  if (!preset) {
    throw new Error(`Unknown preset: ${source.preset ?? "(none)"}`);
  }
  return preset.build(source.overrides);
}
