export interface ExperimentConfig {
  name: string;
  preset?: string;
  description?: string;
  durationMs?: number;
  agents: AgentSpec[];
  taskPool?: TaskSpec[];
  metrics?: string[];
  policy?: {
    autoCullReputationUnder?: number;
    autoFundOnConservation?: boolean;
    blockSelfSpawn?: boolean;
  };
}

export interface AgentSpec {
  name: string;
  genesis: string;
  endowment?: number;
  provider?: string;
  model?: string;
  role?: "root" | "leaf" | "orchestrator";
}

export interface TaskSpec {
  prompt: string;
  reward?: number;
  assignTo?: string;
  delayMs?: number;
}

export function validateConfig(input: unknown): ExperimentConfig {
  if (typeof input !== "object" || input === null) {
    throw new Error("Config must be an object");
  }
  const cfg = input as Record<string, unknown>;
  if (typeof cfg.name !== "string" || cfg.name.trim().length === 0) {
    throw new Error("Config must have a non-empty 'name'");
  }
  if (!Array.isArray(cfg.agents) || cfg.agents.length === 0) {
    throw new Error("Config must have a non-empty 'agents' array");
  }
  for (let i = 0; i < cfg.agents.length; i++) {
    const a = cfg.agents[i] as Record<string, unknown>;
    if (typeof a.name !== "string" || typeof a.genesis !== "string") {
      throw new Error(`agents[${i}] must have name (string) and genesis (string)`);
    }
  }
  if (cfg.taskPool !== undefined && !Array.isArray(cfg.taskPool)) {
    throw new Error("'taskPool' must be an array if provided");
  }
  if (cfg.durationMs !== undefined && typeof cfg.durationMs !== "number") {
    throw new Error("'durationMs' must be a number");
  }
  return cfg as unknown as ExperimentConfig;
}

export function parseConfigYaml(text: string): ExperimentConfig {
  const yaml = parseSimpleYaml(text);
  return validateConfig(yaml);
}

function parseSimpleYaml(text: string): Record<string, unknown> {
  const lines = text.split("\n");
  const root: Record<string, unknown> = {};
  const stack: Array<{ indent: number; obj: Record<string, unknown> }> = [
    { indent: -1, obj: root },
  ];

  for (const rawLine of lines) {
    if (!rawLine.trim() || rawLine.trim().startsWith("#")) continue;
    const indent = rawLine.length - rawLine.trimStart().length;
    const line = rawLine.trim();
    while (stack.length > 1 && stack[stack.length - 1]!.indent >= indent) {
      stack.pop();
    }
    const parent = stack[stack.length - 1]!.obj;
    const dashMatch = line.match(/^- (.+)$/);
    if (dashMatch) {
      const item = dashMatch[1]!;
      if (item.includes(":")) {
        const [k, ...rest] = item.split(":");
        const v = rest.join(":").trim();
        const child: Record<string, unknown> = {};
        if (v.length > 0) child[k!.trim()] = parseScalar(v);
        const arr = parent[Object.keys(parent).slice(-1)[0]!] as unknown[];
        if (Array.isArray(arr)) arr.push(child);
        stack.push({ indent, obj: child });
      } else {
        const key = Object.keys(parent).slice(-1)[0];
        if (key && Array.isArray(parent[key])) {
          (parent[key] as unknown[]).push(parseScalar(item));
        }
      }
      continue;
    }
    const colonMatch = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$/);
    if (colonMatch) {
      const key = colonMatch[1]!;
      const value = colonMatch[2]!.trim();
      if (value.length === 0) {
        const arr: unknown[] = [];
        parent[key] = arr;
        const wrapObj: Record<string, unknown> = {};
        (arr as unknown[]).push(wrapObj);
        stack.push({ indent, obj: wrapObj });
      } else {
        parent[key] = parseScalar(value);
      }
    }
  }
  return root;
}

function parseScalar(s: string): string | number | boolean | string[] {
  const trimmed = s.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "null") return "";
  if (/^-?\d+$/.test(trimmed)) return Number.parseInt(trimmed, 10);
  if (/^-?\d+\.\d+$/.test(trimmed)) return Number.parseFloat(trimmed);
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return trimmed
      .slice(1, -1)
      .split(",")
      .map((x) => x.trim().replace(/^["']|["']$/g, ""))
      .filter((x) => x.length > 0);
  }
  return trimmed.replace(/^["']|["']$/g, "");
}
