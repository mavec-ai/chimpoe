import { mkdir, readdir, stat } from "node:fs/promises";
import { dirname } from "node:path";

export interface ReadFileOptions {
  maxBytes?: number;
  offset?: number;
}

export async function readFile(path: string, options: ReadFileOptions = {}): Promise<string> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    throw new Error(`File not found: ${path}`);
  }
  const maxBytes = options.maxBytes ?? 1_048_576;
  const size = file.size ?? 0;
  let text = await file.text();
  if (options.offset) text = text.slice(options.offset);
  if (text.length > maxBytes) {
    text = text.slice(0, maxBytes) + `\n\n... (${size} bytes total, truncated at ${maxBytes})`;
  }
  return text;
}

export async function writeFile(path: string, content: string): Promise<{ bytes: number }> {
  await mkdir(dirname(path), { recursive: true });
  const bytes = await Bun.write(path, content);
  return { bytes };
}

export interface EditFileOptions {
  all?: boolean;
}

export interface EditFileResult {
  path: string;
  applied: boolean;
  replacements: number;
  message: string;
}

export async function editFile(
  path: string,
  oldString: string,
  newString: string,
  options: EditFileOptions = {},
): Promise<EditFileResult> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    throw new Error(`File not found: ${path}`);
  }
  const content = await file.text();

  if (oldString === newString) {
    return {
      path,
      applied: false,
      replacements: 0,
      message: "oldString and newString are identical. No change.",
    };
  }

  const occurrences = countOccurrences(content, oldString);
  if (occurrences === 0) {
    return {
      path,
      applied: false,
      replacements: 0,
      message: `oldString not found in ${path}. Match whitespace and indentation exactly.`,
    };
  }
  if (occurrences > 1 && !options.all) {
    return {
      path,
      applied: false,
      replacements: occurrences,
      message: `oldString found ${occurrences} times in ${path}. Pass all=true to replace all, or include more context.`,
    };
  }

  const updated = options.all
    ? content.split(oldString).join(newString)
    : content.replace(oldString, newString);
  await Bun.write(path, updated);

  return {
    path,
    applied: true,
    replacements: options.all ? occurrences : 1,
    message: `Replaced ${options.all ? occurrences : 1} occurrence(s) in ${path}.`,
  };
}

export interface ListFilesEntry {
  name: string;
  kind: "file" | "dir" | "other";
  size: number;
}

export interface ListFilesResult {
  path: string;
  entries: ListFilesEntry[];
}

export async function listFiles(path: string): Promise<ListFilesResult> {
  const s = await stat(path).catch(() => null);
  if (!s) throw new Error(`Path not found: ${path}`);

  if (!s.isDirectory()) {
    return { path, entries: [{ name: path, kind: "file", size: s.size }] };
  }

  const items = await readdir(path, { withFileTypes: true });
  const entries: ListFilesEntry[] = [];
  for (const item of items) {
    let kind: ListFilesEntry["kind"] = "other";
    let size = 0;
    if (item.isFile()) {
      kind = "file";
      const childStat = await stat(`${path}/${item.name}`).catch(() => null);
      size = childStat?.size ?? 0;
    } else if (item.isDirectory()) {
      kind = "dir";
    }
    entries.push({ name: item.name, kind, size });
  }
  entries.sort((a, b) => {
    if (a.kind === "dir" && b.kind !== "dir") return -1;
    if (a.kind !== "dir" && b.kind === "dir") return 1;
    return a.name.localeCompare(b.name);
  });
  return { path, entries };
}

function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let i = 0;
  while ((i = haystack.indexOf(needle, i)) !== -1) {
    count++;
    i += needle.length;
  }
  return count;
}
