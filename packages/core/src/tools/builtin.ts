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
import {
  checkInbox,
  listConversations,
  markRead,
  sendMessage,
  type MessageType,
} from "../messaging/index.ts";
import { listAgents } from "../state/agents.ts";
import { spawnChild } from "../replication/index.ts";
import { getBudgetSnapshot } from "../economy/index.ts";
import {
  calculateReputation,
  listReputationEvents,
  recordReputationEvent,
} from "../reputation/index.ts";
import { getFossilByAgent, listFossils, searchFossils } from "../fossils/index.ts";

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
    tools.message_agent = tool({
      description:
        "Send a message to another agent by id (or short id prefix). Use type 'task' when " +
        "assigning work, 'text' for general chat, 'result' for a task response. " +
        "The receiving agent will see this in its inbox on its next poll (if running as a daemon) " +
        "or when it next calls check_inbox.",
      inputSchema: z.object({
        toAgentId: z.string().describe("Destination agent id or short prefix"),
        content: z.string().describe("Message body"),
        type: z.enum(["text", "task", "result"]).optional().describe("Message type (default text)"),
      }),
      execute: async ({ toAgentId, content, type }) => {
        const all = await listAgents();
        const match = all.find((a) => a.id === toAgentId || a.id.startsWith(toAgentId));
        if (!match) {
          return { sent: false, message: `No agent matching "${toAgentId}".` };
        }
        const msg = await sendMessage({
          fromAgentId: agentId,
          toAgentId: match.id,
          content,
          type: (type ?? "text") as MessageType,
        });
        return {
          sent: true,
          messageId: msg.id,
          toName: match.name,
          toId: match.id,
        };
      },
    });
    tools.check_inbox = tool({
      description:
        "Check your inbox for messages from other agents. Returns unread messages first. " +
        "Use this proactively if you are waiting for a reply or expecting work.",
      inputSchema: z.object({
        unreadOnly: z.boolean().optional().describe("Only unread (default true)"),
        limit: z.number().min(1).max(50).optional().describe("Max messages (default 10)"),
      }),
      execute: async ({ unreadOnly, limit }) => {
        const messages = await checkInbox(agentId, {
          unreadOnly: unreadOnly ?? true,
          limit,
        });
        return {
          count: messages.length,
          messages: messages.map((m) => ({
            id: m.id,
            from: m.fromAgentId,
            type: m.type,
            content: m.content,
            createdAt: new Date(m.createdAt).toISOString(),
            inReplyTo: m.inReplyTo,
          })),
        };
      },
    });
    tools.mark_message_read = tool({
      description: "Mark a received message as read by id. Done automatically when you reply.",
      inputSchema: z.object({ messageId: z.string() }),
      execute: async ({ messageId }) => {
        await markRead(messageId);
        return { marked: true };
      },
    });
    tools.list_peers = tool({
      description:
        "List other agents you can message, plus your recent conversations (last message per peer).",
      inputSchema: z.object({}),
      execute: async () => {
        const [allAgents, convos] = await Promise.all([listAgents(), listConversations(agentId)]);
        return {
          agents: allAgents
            .filter((a) => a.id !== agentId)
            .map((a) => ({ id: a.id, name: a.name, status: a.status, tier: a.tier })),
          conversations: convos.map((c) => ({
            partnerId: c.partnerId,
            lastContent: c.lastMessage.content.slice(0, 120),
            lastAt: new Date(c.lastMessage.createdAt).toISOString(),
            unread: c.unreadCount,
          })),
        };
      },
    });
    tools.spawn_child = tool({
      description:
        "Reproduce: spawn a child agent with a new genesis prompt. The child inherits your " +
        "provider/model by default. You may endow it with budget (deducted from your own). " +
        "Use this when you face work that needs a specialist you cannot efficiently do yourself. " +
        "The child is sovereign — it has its own workspace, SOUL, and inbox. " +
        "Write a SPECIFIC, actionable genesis prompt. Bad: 'be helpful'. " +
        "Good: 'You are a TypeScript test specialist. Write vitest tests. Prefer behavior over " +
        "snapshot tests. Always run bun test before reporting done.'",
      inputSchema: z.object({
        name: z
          .string()
          .min(1)
          .max(40)
          .describe("Short kebab-case name, e.g. 'qa', 'devops', 'refactor'"),
        genesisPrompt: z.string().min(20).describe("Full genesis prompt for the child — its DNA"),
        provider: z
          .enum(["openai", "anthropic", "ollama", "google", "xai", "groq"])
          .optional()
          .describe("Override provider (defaults to your provider)"),
        modelId: z.string().optional().describe("Override model id (defaults to your model)"),
        endowmentTokens: z
          .number()
          .int()
          .min(0)
          .optional()
          .describe("Token budget to transfer from your balance to the child (default 0)"),
      }),
      execute: async ({ name, genesisPrompt, provider, modelId, endowmentTokens }) => {
        try {
          const result = await spawnChild({
            parentId: agentId,
            name,
            genesisPrompt,
            provider,
            modelId,
            endowmentTokens,
          });
          return {
            spawned: true,
            childId: result.child.id,
            childName: result.child.name,
            generation: result.child.generation,
            endowmentTransferred: result.endowmentTransferred,
            yourBalanceNow: result.parentBalanceAfter,
            message: `Child "${result.child.name}" spawned (gen ${result.child.generation}). Message it via message_agent once it is started.`,
          };
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          return { spawned: false, message: `Spawn failed: ${msg}` };
        }
      },
    });
    tools.check_budget = tool({
      description:
        "Check your current token budget, survival tier, and projected runway. " +
        "Tiers: thriving (>50k), normal (>5k), conservation (>500), dormant (>0), dead (=0). " +
        "When you reach conservation, slow down and seek work. When dormant, only critical actions.",
      inputSchema: z.object({}),
      execute: async () => {
        const snap = await getBudgetSnapshot(agentId);
        if (!snap) return { error: "Agent not found." };
        return snap;
      },
    });
    tools.check_reputation = tool({
      description:
        "Check your reputation score (0-100) and recent events. Reputation affects task " +
        "routing priority. Score 50 is baseline. Recent events (last 20) weigh 1.5x. " +
        "Old events decay with 30-day half-life.",
      inputSchema: z.object({
        limit: z.number().min(1).max(50).optional().describe("Events to show (default 10)"),
      }),
      execute: async ({ limit }) => {
        const [score, events] = await Promise.all([
          calculateReputation(agentId),
          listReputationEvents(agentId, limit ?? 10),
        ]);
        return {
          score: score.score,
          eventCount: score.eventCount,
          recentDelta: score.recentDelta,
          recentEvents: events.map((e) => ({
            type: e.eventType,
            delta: e.delta,
            reason: e.reason,
            at: new Date(e.createdAt).toISOString().slice(0, 10),
          })),
        };
      },
    });
    tools.reward_peer = tool({
      description:
        "Record a reputation event for another agent. Use 'user_praise' to upvote a peer's " +
        "work (you saw them do something well), or 'bug_introduced' if they broke something. " +
        "Use sparingly — reputation works best when based on real observations.",
      inputSchema: z.object({
        peerId: z.string().describe("Target agent id or prefix"),
        eventType: z.enum(["user_praise", "bug_introduced", "output_reused"]),
        reason: z.string().min(5).max(200).describe("Why you're recording this"),
      }),
      execute: async ({ peerId, eventType, reason }) => {
        const all = await listAgents();
        const match = all.find((a) => a.id === peerId || a.id.startsWith(peerId));
        if (!match) return { recorded: false, message: `No peer matching "${peerId}".` };
        if (match.id === agentId) {
          return { recorded: false, message: "Cannot record reputation for yourself." };
        }
        const event = await recordReputationEvent({
          agentId: match.id,
          eventType,
          reason: `From ${agentId.slice(0, 8)}: ${reason}`,
        });
        return {
          recorded: true,
          peerName: match.name,
          eventType: event.eventType,
          delta: event.delta,
        };
      },
    });
    tools.read_my_fossil = tool({
      description:
        "Read your own fossil (auto-distilled when you die, capturing your SOUL + top memories). " +
        "Useful to review what of yours will outlive you. Returns null if you haven't been distilled yet.",
      inputSchema: z.object({}),
      execute: async () => {
        const fossil = await getFossilByAgent(agentId);
        if (!fossil)
          return { found: false, message: "No fossil yet — fossils are distilled on death." };
        return {
          found: true,
          sizeBytes: fossil.content.length,
          keywords: fossil.keywords,
          excerpt: fossil.content.slice(0, 4000),
        };
      },
    });
    tools.search_fossils = tool({
      description:
        "Search the fossil record of dead agents (your ancestors and lineage peers). " +
        "Fossils are distilled knowledge from agents that came before you. " +
        "Use this BEFORE asking a peer or user for help — your ancestors may have already solved it.",
      inputSchema: z.object({
        query: z.string().describe("Keywords to search across all fossils"),
        limit: z.number().min(1).max(20).optional().describe("Max results (default 5)"),
      }),
      execute: async ({ query, limit }) => {
        const results = await searchFossils(query, { limit: limit ?? 5 });
        return {
          count: results.length,
          fossils: results.map((f) => ({
            id: f.id,
            fromAgent: f.agentName,
            generation: f.generation,
            keywords: f.keywords.slice(0, 6),
            excerpt: f.content.slice(0, 1500),
          })),
        };
      },
    });
    tools.list_fossils = tool({
      description: "List all available fossils (dead agents' distilled knowledge).",
      inputSchema: z.object({
        limit: z.number().min(1).max(50).optional().describe("Max results (default 20)"),
      }),
      execute: async ({ limit }) => {
        const results = await listFossils(limit ?? 20);
        return {
          count: results.length,
          fossils: results.map((f) => ({
            id: f.id,
            fromAgent: f.agentName,
            generation: f.generation,
            keywords: f.keywords.slice(0, 6),
            sizeBytes: f.content.length,
            distilledAt: new Date(f.createdAt).toISOString().slice(0, 10),
          })),
        };
      },
    });
  }
  return tools;
}
