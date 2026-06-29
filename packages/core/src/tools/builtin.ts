import { tool, type Tool } from "ai";
import { z } from "zod";
import { editFile, listFiles, readFile, runShell, writeFile } from "../exec/index.ts";
import { appendReflection, readSoul, updateSoulSection } from "../soul/index.ts";
import {
  deleteMemory,
  formatMemoryForPrompt,
  getMemory,
  listMemories,
  searchMemory,
  writeMemory,
  type MemoryType,
} from "../memory/index.ts";

export const shellTool: Tool = tool({
  description:
    "Run a shell command via bash. Returns stdout, stderr, exit code, and duration. " +
    "Default timeout is 30s. Output is truncated at 1MB. Use for: ls, git, grep, build, test, etc.",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute"),
    cwd: z.string().optional().describe("Working directory (defaults to process cwd)"),
    timeoutMs: z
      .number()
      .optional()
      .describe("Max execution time in ms (default 30000, max 300000)"),
  }),
  execute: async ({ command, cwd, timeoutMs }, { abortSignal }) => {
    const result = await runShell(command, {
      cwd,
      timeoutMs: Math.min(timeoutMs ?? 30_000, 300_000),
      signal: abortSignal,
    });
    return result;
  },
});

export const readFileTool: Tool = tool({
  description:
    "Read the contents of a text file. Returns the text. Truncates at 1MB by default. " +
    "Throws if the file does not exist.",
  inputSchema: z.object({
    path: z.string().describe("Absolute or cwd-relative file path"),
    offset: z.number().optional().describe("Skip the first N characters"),
    maxBytes: z.number().optional().describe("Max bytes to read (default 1048576)"),
  }),
  execute: async ({ path, offset, maxBytes }) => {
    const content = await readFile(path, { offset, maxBytes });
    return { path, content };
  },
});

export const writeFileTool: Tool = tool({
  description:
    "Write content to a file (creates or overwrites). Parent directories are created " +
    "automatically. Returns the number of bytes written.",
  inputSchema: z.object({
    path: z.string().describe("Absolute or cwd-relative file path"),
    content: z.string().describe("The full content to write"),
  }),
  execute: async ({ path, content }) => {
    const result = await writeFile(path, content);
    return { path, ...result };
  },
});

export const editFileTool: Tool = tool({
  description:
    "Edit a file by replacing oldString with newString. By default requires oldString " +
    "to be unique in the file. Pass all=true to replace every occurrence. " +
    "Never matches when oldString === newString. Returns whether the edit was applied.",
  inputSchema: z.object({
    path: z.string().describe("Absolute or cwd-relative file path"),
    oldString: z.string().describe("The exact text to find (match whitespace exactly)"),
    newString: z.string().describe("The replacement text"),
    all: z.boolean().optional().describe("Replace all occurrences (default false)"),
  }),
  execute: async ({ path, oldString, newString, all }) => {
    return await editFile(path, oldString, newString, { all });
  },
});

export const listFilesTool: Tool = tool({
  description:
    "List entries in a directory. Returns name, kind (file|dir|other), and size. " +
    "Directories sorted first, then alphabetical.",
  inputSchema: z.object({
    path: z.string().describe("Absolute or cwd-relative directory path"),
  }),
  execute: async ({ path }) => {
    return await listFiles(path);
  },
});

export function builtinTools(agentId?: string): Record<string, Tool> {
  const tools: Record<string, Tool> = {
    shell: shellTool,
    read_file: readFileTool,
    write_file: writeFileTool,
    edit_file: editFileTool,
    list_files: listFilesTool,
  };
  if (agentId) {
    tools.read_soul = tool({
      description: "Read your own SOUL.md identity document.",
      inputSchema: z.object({}),
      execute: async () => {
        const content = await readSoul(agentId);
        return { content };
      },
    });
    tools.update_soul_section = tool({
      description:
        "Rewrite a section of your SOUL.md by heading. Use this to evolve your sense of self. " +
        "Headings like 'Strengths', 'Weaknesses', 'Current focus', 'Self-description'. " +
        "If the section does not exist it is created. Provide the FULL new body, not a diff.",
      inputSchema: z.object({
        heading: z.string().describe("Section heading without the leading #"),
        body: z.string().describe("Full new body for this section"),
      }),
      execute: async ({ heading, body }) => {
        return await updateSoulSection(agentId, heading, body);
      },
    });
    tools.append_reflection = tool({
      description:
        "Append a timestamped reflection to your SOUL.md. Use sparingly to record meaningful " +
        "shifts in self-understanding, not routine status updates.",
      inputSchema: z.object({
        note: z.string().describe("A short reflection note"),
      }),
      execute: async ({ note }) => {
        await appendReflection(agentId, note);
        return { applied: true };
      },
    });
    tools.write_memory = tool({
      description:
        "Persist a memory across sessions. Pick the right type: " +
        "episodic = specific past event ('yesterday I refactored X and test Y broke'), " +
        "semantic = generalized fact ('library Z is buggy in v2'), " +
        "procedural = how-to ('deploy via wrangler deploy'), " +
        "working = current task context (volatile, short-lived). " +
        "Use sparingly — only for things worth remembering long-term (except working).",
      inputSchema: z.object({
        type: z.enum(["working", "episodic", "semantic", "procedural"]),
        content: z.string().describe("The memory content. Be specific and self-contained."),
        tags: z.array(z.string()).optional().describe("Optional tags for grouping"),
        importance: z
          .number()
          .min(0)
          .max(100)
          .optional()
          .describe("Importance 0-100 (default 50). Higher = more likely to be recalled."),
      }),
      execute: async ({ type, content, tags, importance }) => {
        const mem = await writeMemory({
          agentId,
          type: type as MemoryType,
          content,
          tags,
          importance,
        });
        return { id: mem.id, type: mem.type, content: mem.content };
      },
    });
    tools.search_memory = tool({
      description:
        "Search your memories by keyword. Returns matches ranked by importance. " +
        "Use this BEFORE asking the user something you might already know, and BEFORE " +
        "repeating work you may have done before.",
      inputSchema: z.object({
        query: z.string().describe("Keywords to search for"),
        type: z
          .enum(["working", "episodic", "semantic", "procedural"])
          .optional()
          .describe("Limit to a specific memory type"),
        limit: z.number().min(1).max(50).optional().describe("Max results (default 10)"),
      }),
      execute: async ({ query, type, limit }) => {
        const results = await searchMemory(agentId, query, {
          type: type as MemoryType | undefined,
          limit,
        });
        return { count: results.length, results: results.map(formatMemoryForPrompt) };
      },
    });
    tools.list_memories = tool({
      description: "List your memories, optionally filtered by type. Use to refresh your context.",
      inputSchema: z.object({
        type: z
          .enum(["working", "episodic", "semantic", "procedural"])
          .optional()
          .describe("Limit to a specific memory type"),
        limit: z.number().min(1).max(50).optional().describe("Max results (default 20)"),
      }),
      execute: async ({ type, limit }) => {
        const results = await listMemories(agentId, {
          type: type as MemoryType | undefined,
          limit,
        });
        return { count: results.length, results: results.map(formatMemoryForPrompt) };
      },
    });
    tools.delete_memory = tool({
      description: "Delete a memory by id. Use when a memory is wrong, outdated, or low-value.",
      inputSchema: z.object({
        id: z.string(),
      }),
      execute: async ({ id }) => {
        const existing = await getMemory(id);
        if (!existing) return { deleted: false, message: "Memory not found." };
        await deleteMemory(id);
        return { deleted: true, message: `Deleted: ${existing.content.slice(0, 80)}` };
      },
    });
  }
  return tools;
}
