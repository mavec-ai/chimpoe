export interface RunShellOptions {
  cwd?: string;
  timeoutMs?: number;
  maxOutputBytes?: number;
  env?: Record<string, string>;
  signal?: AbortSignal;
}

export interface ShellResult {
  command: string;
  exitCode: number;
  stdout: string;
  stderr: string;
  durationMs: number;
  truncated: boolean;
  timedOut: boolean;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_OUTPUT = 1_048_576; // 1 MiB

export async function runShell(
  command: string,
  options: RunShellOptions = {},
): Promise<ShellResult> {
  const start = Date.now();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxOutput = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  let stdout = "";
  let stderr = "";
  let truncated = false;
  let timedOut = false;

  try {
    const proc = Bun.spawn({
      cmd: ["bash", "-c", command],
      cwd: options.cwd ?? process.cwd(),
      env: { ...process.env, ...options.env },
      stdout: "pipe",
      stderr: "pipe",
      stdin: "ignore",
      signal: controller.signal,
    });

    const stdoutBytes = await new Response(proc.stdout).arrayBuffer();
    const stderrBytes = await new Response(proc.stderr).arrayBuffer();

    stdout = Buffer.from(stdoutBytes).toString("utf8");
    stderr = Buffer.from(stderrBytes).toString("utf8");

    if (stdout.length > maxOutput) {
      stdout = stdout.slice(0, maxOutput);
      truncated = true;
    }
    if (stderr.length > maxOutput) {
      stderr = stderr.slice(0, maxOutput);
      truncated = true;
    }

    let exitCode = await proc.exited;
    if (controller.signal.aborted) {
      timedOut = true;
      exitCode = 124;
    }

    return {
      command,
      exitCode,
      stdout,
      stderr,
      durationMs: Date.now() - start,
      truncated,
      timedOut,
    };
  } catch (err) {
    if (controller.signal.aborted) timedOut = true;
    const message = err instanceof Error ? err.message : String(err);
    return {
      command,
      exitCode: -1,
      stdout,
      stderr: stderr + (stderr ? "\n" : "") + message,
      durationMs: Date.now() - start,
      truncated,
      timedOut,
    };
  } finally {
    clearTimeout(timeout);
  }
}

export function formatShellResult(r: ShellResult): string {
  const parts: string[] = [
    `$ ${r.command}`,
    `exit=${r.exitCode} duration=${r.durationMs}ms${r.timedOut ? " (TIMEOUT)" : ""}${r.truncated ? " (OUTPUT TRUNCATED)" : ""}`,
  ];
  if (r.stdout) parts.push("--- stdout ---\n" + r.stdout);
  if (r.stderr) parts.push("--- stderr ---\n" + r.stderr);
  return parts.join("\n");
}
