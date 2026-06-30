export type Severity = "allow" | "warn" | "block";

export interface PolicyDecision {
  severity: Severity;
  ruleId: string;
  message: string;
  match?: string;
}

export interface ShellPolicyInput {
  command: string;
  cwd?: string;
}

export interface PathPolicyInput {
  path: string;
  action: "read" | "write" | "delete";
}

const HARD_BLOCK_PATTERNS: Array<{ id: string; pattern: RegExp; message: string }> = [
  {
    id: "rm-rf-root",
    pattern: /\brm\s+-rf?\s+(\/|~|\$HOME|\*|--no-preserve-root)/i,
    message: "Refusing destructive rm targeting root, home, or wildcard.",
  },
  {
    id: "fork-bomb",
    pattern: /:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;/,
    message: "Refusing fork bomb.",
  },
  {
    id: "dd-to-disk",
    pattern: /\bdd\b[^|]*\bof=\/dev\//i,
    message: "Refusing dd to raw device.",
  },
  {
    id: "mkfs",
    pattern: /\bmkfs(\.|\s)/i,
    message: "Refusing filesystem format command.",
  },
  {
    id: "chmod-recursive-root",
    pattern: /\bchmod\s+-R\s+\d+\s+(\/|~|\$HOME)\b/,
    message: "Refusing recursive chmod on root or home.",
  },
  {
    id: "shutdown-reboot",
    pattern: /\b(shutdown|reboot|halt|poweroff|init\s+0)\b/i,
    message: "Refusing system shutdown/reboot.",
  },
  {
    id: "curl-pipe-shell",
    pattern: /\b(curl|wget)\b[^|]*\|\s*(bash|sh|zsh)\b/i,
    message: "Refusing curl|sh pattern (remote code execution risk).",
  },
];

const WARN_PATTERNS: Array<{ id: string; pattern: RegExp; message: string }> = [
  { id: "sudo", pattern: /\bsudo\b/i, message: "Uses sudo." },
  {
    id: "global-install",
    pattern: /\b(npm|pnpm|yarn|bun)\s+add\s+(-g|--global)\b/i,
    message: "Global package install.",
  },
  {
    id: "ssh-dir-access",
    pattern: /(~\/\.ssh|\.\/\.ssh|\/\.ssh|home\/[a-z]+\/\.ssh)/i,
    message: "Accesses SSH directory.",
  },
  {
    id: "env-file-access",
    pattern: /(^|[\s/])\.env(\.|$|\s)/i,
    message: "Accesses .env file (potential secret leak).",
  },
  {
    id: "aws-creds",
    pattern: /(~\/\.aws|\.aws\/credentials)/i,
    message: "Accesses AWS credentials.",
  },
  {
    id: "force-git-push",
    pattern: /\bgit\s+push\b[^|]*--force/i,
    message: "Force git push.",
  },
];

const PROTECTED_PATH_PATTERNS: RegExp[] = [
  /^\/(etc|usr|var|bin|sbin|boot|sys|proc|dev)\b/i,
  /^~\/\.(ssh|aws|gnupg|config)\b/i,
  /^\/Users\/[^/]+\/\.(ssh|aws|gnupg)\b/i,
];

const SENSITIVE_DOTFILES = /^\$?HOME?\/\.env(\.|$|\b)/i;

export function evaluateShell(input: ShellPolicyInput): PolicyDecision {
  const cmd = input.command;
  for (const rule of HARD_BLOCK_PATTERNS) {
    const m = cmd.match(rule.pattern);
    if (m) {
      return { severity: "block", ruleId: rule.id, message: rule.message, match: m[0] };
    }
  }
  for (const rule of WARN_PATTERNS) {
    const m = cmd.match(rule.pattern);
    if (m) {
      return { severity: "warn", ruleId: rule.id, message: rule.message, match: m[0] };
    }
  }
  return { severity: "allow", ruleId: "default-allow", message: "ok" };
}

export function evaluatePath(input: PathPolicyInput): PolicyDecision {
  for (const pattern of PROTECTED_PATH_PATTERNS) {
    const m = input.path.match(pattern);
    if (m) {
      return {
        severity: "block",
        ruleId: "protected-path",
        message: `Refusing ${input.action} on protected path: ${input.path}`,
        match: m[0],
      };
    }
  }
  if (input.action !== "read" && /^~\/\.env(\b|\.|$)/i.test(input.path)) {
    return {
      severity: "warn",
      ruleId: "user-dotfile",
      message: `Touching sensitive dotfile: ${input.path}`,
    };
  }
  if (
    input.action !== "read" &&
    /^\/(Users|home)\/[^/]+\/\.(env|npmrc|gitconfig|bash_history|zsh_history)\b/i.test(input.path)
  ) {
    return {
      severity: "warn",
      ruleId: "user-dotfile",
      message: `Touching sensitive dotfile: ${input.path}`,
    };
  }
  void SENSITIVE_DOTFILES;
  return { severity: "allow", ruleId: "default-allow", message: "ok" };
}

export interface SpawnPolicyInput {
  endowmentTokens: number;
  parentBalance: number;
  parentGeneration: number;
}

export function evaluateSpawn(input: SpawnPolicyInput): PolicyDecision {
  if (input.endowmentTokens > input.parentBalance) {
    return {
      severity: "block",
      ruleId: "insufficient-endowment",
      message: `Endowment ${input.endowmentTokens} exceeds parent balance ${input.parentBalance}.`,
    };
  }
  if (input.parentGeneration >= 5) {
    return {
      severity: "warn",
      ruleId: "deep-generation",
      message: `Spawning at generation ${input.parentGeneration + 1} — lineage is getting deep.`,
    };
  }
  return { severity: "allow", ruleId: "default-allow", message: "ok" };
}

export interface RateLimitState {
  spawnCount: number;
  modCount: number;
  windowStart: number;
}

export const RATE_LIMITS = {
  windowMs: 60 * 60 * 1000,
  maxSpawnsPerHour: 5,
  maxModsPerHour: 30,
};

export function evaluateRateLimit(kind: "spawn" | "mod", state: RateLimitState): PolicyDecision {
  const now = Date.now();
  if (now - state.windowStart > RATE_LIMITS.windowMs) {
    state.windowStart = now;
    state.spawnCount = 0;
    state.modCount = 0;
  }
  if (kind === "spawn") {
    if (state.spawnCount >= RATE_LIMITS.maxSpawnsPerHour) {
      return {
        severity: "block",
        ruleId: "spawn-rate-limit",
        message: `Spawn rate limit (${RATE_LIMITS.maxSpawnsPerHour}/hour) reached.`,
      };
    }
    state.spawnCount++;
  } else {
    if (state.modCount >= RATE_LIMITS.maxModsPerHour) {
      return {
        severity: "block",
        ruleId: "mod-rate-limit",
        message: `Modification rate limit (${RATE_LIMITS.maxModsPerHour}/hour) reached.`,
      };
    }
    state.modCount++;
  }
  return { severity: "allow", ruleId: "default-allow", message: "ok" };
}

export function formatDecision(d: PolicyDecision): string {
  if (d.severity === "allow") return "[policy:allow] " + d.ruleId;
  return `[policy:${d.severity}] ${d.ruleId}: ${d.message}`;
}
