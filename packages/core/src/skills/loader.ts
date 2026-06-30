import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { getAgentHome } from "@chimpoe/types";

export interface Skill {
  name: string;
  description: string;
  version: string;
  tags: string[];
  enabled: boolean;
  source: "built-in" | "user";
  filePath: string;
  body: string;
  sizeBytes: number;
}

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;

function parseFrontmatter(text: string): {
  fields: Record<string, string | string[] | boolean>;
  body: string;
} {
  const match = text.match(FRONTMATTER_RE);
  if (!match) return { fields: {}, body: text };
  const raw = match[1]!;
  const body = match[2] ?? "";
  const fields: Record<string, string | string[] | boolean> = {};
  for (const line of raw.split("\n")) {
    const m = line.match(/^([a-zA-Z_]+):\s*(.*)$/);
    if (!m) continue;
    const key = m[1]!.trim();
    let value: string | string[] | boolean = m[2]!.trim();
    if (value === "true") value = true;
    else if (value === "false") value = false;
    else if (value.startsWith("[") && value.endsWith("]")) {
      value = value
        .slice(1, -1)
        .split(",")
        .map((s) => s.trim().replace(/^["']|["']$/g, ""))
        .filter((s) => s.length > 0);
    } else {
      value = value.replace(/^["']|["']$/g, "");
    }
    fields[key] = value;
  }
  return { fields, body };
}

export function renderSkill(skill: Skill): string {
  return `### Skill: ${skill.name}\n\n${skill.body.trim()}`;
}

export async function listSkills(agentId: string): Promise<Skill[]> {
  const skillsDir = join(getAgentHome(agentId), "skills");
  let entries: string[] = [];
  try {
    entries = await readdir(skillsDir);
  } catch {
    return [];
  }
  const skills: Skill[] = [];
  for (const entry of entries) {
    if (!entry.endsWith(".md") && !entry.endsWith(".mdx")) continue;
    const filePath = join(skillsDir, entry);
    try {
      const text = await readFile(filePath, "utf8");
      const { fields, body } = parseFrontmatter(text);
      const s = await stat(filePath);
      skills.push({
        name: (fields.name as string) ?? entry.replace(/\.mdx?$/, ""),
        description: (fields.description as string) ?? "",
        version: (fields.version as string) ?? "0.0.0",
        tags: Array.isArray(fields.tags) ? fields.tags : [],
        enabled: fields.enabled !== false,
        source: "user",
        filePath,
        body,
        sizeBytes: s.size,
      });
    } catch {
      // skip unreadable
    }
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

export async function getActiveSkills(agentId: string): Promise<Skill[]> {
  const all = await listSkills(agentId);
  return all.filter((s) => s.enabled);
}

export interface InstallSkillInput {
  agentId: string;
  name: string;
  body: string;
  description?: string;
  version?: string;
  tags?: string[];
  enabled?: boolean;
}

export async function installSkill(input: InstallSkillInput): Promise<Skill> {
  const skillsDir = join(getAgentHome(input.agentId), "skills");
  await mkdir(skillsDir, { recursive: true });
  const fileName = input.name.replace(/[^a-z0-9-]/gi, "-").toLowerCase() + ".md";
  const filePath = join(skillsDir, fileName);

  const frontmatter: string[] = ["---"];
  frontmatter.push(`name: ${input.name}`);
  if (input.description) frontmatter.push(`description: ${input.description}`);
  frontmatter.push(`version: ${input.version ?? "0.0.0"}`);
  if (input.tags && input.tags.length > 0) {
    frontmatter.push(`tags: [${input.tags.join(", ")}]`);
  }
  frontmatter.push(`enabled: ${input.enabled ?? true}`);
  frontmatter.push("---");
  const content = frontmatter.join("\n") + "\n\n" + input.body.trim() + "\n";
  await writeFile(filePath, content, { mode: 0o644 });

  const s = await stat(filePath);
  return {
    name: input.name,
    description: input.description ?? "",
    version: input.version ?? "0.0.0",
    tags: input.tags ?? [],
    enabled: input.enabled ?? true,
    source: "user",
    filePath,
    body: input.body,
    sizeBytes: s.size,
  };
}

export async function removeSkill(
  agentId: string,
  name: string,
): Promise<{ removed: boolean; path: string | null }> {
  const skills = await listSkills(agentId);
  const target = skills.find((s) => s.name === name || s.filePath.endsWith(name + ".md"));
  if (!target) return { removed: false, path: null };
  await rm(target.filePath, { force: true });
  return { removed: true, path: target.filePath };
}

export async function setSkillEnabled(
  agentId: string,
  name: string,
  enabled: boolean,
): Promise<{ updated: boolean; message: string }> {
  const skills = await listSkills(agentId);
  const target = skills.find((s) => s.name === name);
  if (!target) {
    return { updated: false, message: `Skill "${name}" not found.` };
  }
  if (target.enabled === enabled) {
    return {
      updated: false,
      message: `Skill "${name}" already ${enabled ? "enabled" : "disabled"}.`,
    };
  }
  const text = await readFile(target.filePath, "utf8");
  const match = text.match(FRONTMATTER_RE);
  if (!match) {
    return { updated: false, message: `Skill "${name}" has no frontmatter to edit.` };
  }
  const raw = match[1]!;
  const body = match[2] ?? "";
  const lines = raw.split("\n").map((line) => {
    const m = line.match(/^enabled:\s*(.*)$/);
    if (m) return `enabled: ${enabled}`;
    return line;
  });
  const updated = `---\n${lines.join("\n")}\n---\n${body}`;
  await writeFile(target.filePath, updated, { mode: 0o644 });
  return { updated: true, message: `Skill "${name}" ${enabled ? "enabled" : "disabled"}.` };
}
