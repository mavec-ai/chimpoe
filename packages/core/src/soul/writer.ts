import { exists } from "node:fs/promises";
import { getSoulPath } from "./paths.ts";

export async function readSoul(agentId: string): Promise<string> {
  const path = getSoulPath(agentId);
  const file = Bun.file(path);
  if (!(await file.exists())) return "";
  return file.text();
}

export async function soulExists(agentId: string): Promise<boolean> {
  return await exists(getSoulPath(agentId));
}

export interface SoulSection {
  heading: string;
  body: string;
  startOffset: number;
}

export function parseSoulSections(content: string): SoulSection[] {
  const lines = content.split("\n");
  const sections: SoulSection[] = [];
  let current: SoulSection | null = null;
  let offset = 0;
  for (const line of lines) {
    const headingMatch = line.match(/^#\s+(.+)$/);
    if (headingMatch) {
      if (current) sections.push(current);
      current = {
        heading: headingMatch[1]!.trim(),
        body: "",
        startOffset: offset,
      };
    } else if (current) {
      current.body += (current.body ? "\n" : "") + line;
    }
    offset += line.length + 1;
  }
  if (current) sections.push(current);
  return sections;
}

export interface UpdateSectionResult {
  applied: boolean;
  message: string;
}

export async function updateSoulSection(
  agentId: string,
  heading: string,
  newBody: string,
): Promise<UpdateSectionResult> {
  const content = await readSoul(agentId);
  if (!content) {
    return { applied: false, message: `SOUL.md not found for agent ${agentId}.` };
  }

  const lines = content.split("\n");
  const headingLower = heading.toLowerCase();
  const headingPrefix = "# ";

  let sectionStart = -1;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (
      line.startsWith(headingPrefix) &&
      line.slice(headingPrefix.length).trim().toLowerCase() === headingLower
    ) {
      sectionStart = i;
      break;
    }
  }

  if (sectionStart < 0) {
    const updated = content.trimEnd() + `\n\n# ${heading}\n\n${newBody}\n`;
    await Bun.write(getSoulPath(agentId), updated);
    return { applied: true, message: `Created new section "# ${heading}".` };
  }

  let sectionEnd = lines.length;
  for (let i = sectionStart + 1; i < lines.length; i++) {
    if (lines[i]!.startsWith(headingPrefix)) {
      sectionEnd = i;
      break;
    }
  }

  const newSection = [`# ${heading}`, "", newBody, ""];
  const next = [...lines.slice(0, sectionStart), ...newSection, ...lines.slice(sectionEnd)];
  await Bun.write(getSoulPath(agentId), next.join("\n"));
  return { applied: true, message: `Updated section "# ${heading}".` };
}

export async function appendReflection(agentId: string, note: string): Promise<void> {
  const lines = (await readSoul(agentId)).split("\n");
  const marker = "# Reflections";
  const idx = lines.findIndex((l) => l.trim() === marker);
  if (idx < 0) {
    const stamp = new Date().toISOString().slice(0, 10);
    const entry = `\n\n${marker}\n\n## ${stamp}\n\n${note}\n`;
    await Bun.write(getSoulPath(agentId), (await readSoul(agentId)).trimEnd() + entry);
    return;
  }
  let end = lines.length;
  for (let i = idx + 1; i < lines.length; i++) {
    if (lines[i]!.startsWith("# ")) {
      end = i;
      break;
    }
  }
  const stamp = new Date().toISOString().slice(0, 10);
  const existingBody = lines
    .slice(idx + 1, end)
    .join("\n")
    .trim();
  const isFirst = existingBody.length === 0 || existingBody.startsWith("(append");
  const newBody = isFirst
    ? `## ${stamp}\n\n${note}\n`
    : `${existingBody}\n\n## ${stamp}\n\n${note}\n`;
  const next = [...lines.slice(0, idx + 1), "", newBody, ...lines.slice(end)];
  await Bun.write(getSoulPath(agentId), next.join("\n").replace(/\n{3,}/g, "\n\n") + "\n");
}
