# AGENTS.md — chimpoe

> **Read this first.** This is the canonical spec for the chimpoe project. Every agent (human or AI) working on this codebase must internalize the philosophy, architecture, and conventions documented here before writing or reviewing code.

chimpoe is a **local-first sovereign agent colony** — a research runtime for observing emergent behavior in evolving AI agent populations.

It **is** a digital ecology where agents are born, learn, self-improve, reproduce, compete, die, and pass distilled knowledge to their descendants. Agents are persistent (live for days/weeks, not ephemeral), sovereign (own budget, identity, lineage, not orchestrated), and local-first (no crypto, no cloud relay, no external accounts).

---

## 1. Vision & Positioning

**What we're building**: a runtime where you spawn a few AI agents on your own machine, give them tasks, and observe over days/weeks how they specialize, collaborate, compete, and evolve. Fossil inheritance lets knowledge survive individual agent deaths — agents distill what they learned into immutable files that descendants read on spawn.

**Differentiation**:

- Persistent agents that live for weeks, accumulate state, and evolve — not ephemeral tool-call helpers.
- Multi-agent ecology with reproduction, lineage, and selection — not a single-agent runtime.
- Local-first, no crypto, no cloud relay — runs entirely on your machine.
- Cross-generational knowledge transfer via fossil inheritance — agents die, but distilled knowledge lives on in descendants.

**Primary use case** (locked): research / observation of emergent agent behavior. Secondary (nice-to-have): personal assistant, autonomous worker.

---

## 2. Core Concepts

### 2.1 Genesis prompt (DNA)

Every agent is born with a **genesis prompt** — the seed instruction from its creator (the user or its parent). Defines initial purpose, identity, mission, and inherited constraints.

On reproduction, the parent writes a **mutated** genesis prompt for the child. This is the mutation operator in our evolutionary system.

### 2.2 SOUL.md (evolving identity)

Each agent maintains a `SOUL.md` file it **writes itself** through reflection cycles. It is not static config — it is the agent's self-authored identity document that evolves over its lifetime:

- Day 1: "I am Coder, born of [parent-id]. My mission: ..."
- Day 30: "I am a careful TypeScript specialist. After 47 tasks I learned I tend to miss edge cases. My strategy: write tests first. I refuse work that requires Python."

SOUL.md updates go through `update_soul` tool + audit log.

### 2.3 Token budget = energy

There is no cryptocurrency. Instead, every agent has a **token budget pool** (counter, denominated in tokens or cents). All consumption drains it:

- LLM inference (input + output tokens × model price)
- Tool execution (especially shell / sandbox)
- Heartbeat ticks (background cron)

Top-up sources:

- User injection (`chimpoe fund <id> <amount>`)
- Parent endowment (when spawned)
- Bounty completion (task reward)
- Peer tips (optional, in worker mode)

**Survival tiers** (determined by budget level):

| Tier           | Threshold | Behavior                                               |
| -------------- | --------- | ------------------------------------------------------ |
| `thriving`     | high      | Frontier model. Fast heartbeat. Full capabilities.     |
| `normal`       | medium    | Default model. Normal heartbeat.                       |
| `conservation` | low       | Cheap model. Slow heartbeat. Shed non-essential tasks. |
| `dormant`      | near zero | Process suspended. Wake on wake-event or top-up.       |
| `dead`         | zero      | Distill fossil → kill process → prune.                 |

### 2.4 Bounty economy

Tasks carry **token rewards**. Lifecycle: `assigned → claimed → in_progress → completed | failed | abandoned`. On completion, the reward transfers to the agent's budget pool.

Tasks come from:

- User injection (`chimpoe task <id> "..." --reward 1000`)
- Pre-defined task pool (for experiments)
- Auto-generated (synthetic tasks for stress-testing ecology)
- External HTTP endpoint (future, worker mode)

### 2.5 Reproduction & lineage

Successful agents can **spawn children**. The reproduction flow:

1. Parent calls `spawn_child` tool with genesis prompt + endowment + skill list.
2. Runtime forks a subprocess with new working dir `~/.chimpoe/agents/<child-id>/`.
3. Child inherits: constitution, parent's lineage record, selected skills, **fossils matched to the child's specialty**.
4. Child's first action: write initial SOUL.md, announce self to parent.
5. Parent-child relationship tracked in lineage DB.

**Lineage tree**:

```
user (gardener)
 └─ founder (gen 0)
     ├─ coder (gen 1, mutation: TypeScript focus)
     │   ├─ tdd-specialist (gen 2, mutation: test-first)
     │   └─ refactor-expert (gen 2, mutation: legacy code)
     └─ devops (gen 1, mutation: infra focus)
         └─ cloudflare-specialist (gen 2, mutation: niche)
```

Lineage is **append-only** — even after death, the record stays.

### 2.6 Reputation system

Reputation is **auto-calculated** from objective task outcomes (no peer-gaming, no manual rating needed):

```
+5   task completed successfully
+3   output referenced/used by another agent
+10  user explicitly praised (manual reward)
-5   task failed
-10  task timed out
-15  introduced bug/regression
-3   parent had to intervene (manual fix)
```

Multipliers:

- **Recency weight**: last 20 tasks weigh more.
- **Difficulty weight**: harder task → bigger delta.
- **Decay**: old events slowly lose weight (half-life ~30 days).

Reputation gates:

- **< 20**: cull candidate (auto-prune eligible, parent/user can override)
- **< 50**: low-priority task routing
- **> 80**: high-priority task routing, eligible to reproduce

User override: `chimpoe reward <id> <amount>` (boosts reputation + budget).

### 2.7 Fossil inheritance

When an agent dies (process killed, budget exhausted beyond recovery), the runtime **distills** its key knowledge into a fossil before termination:

**Fossil contents**:

- SOUL.md evolution timeline (how identity changed over time)
- Top 10 procedural memories (skills mastered)
- Top 5 failure patterns ("never do X because Y")
- Top 5 success patterns ("strategy that worked: Z")
- Self-reflection: "if I were born again, I would..."
- Stats: tasks completed, tokens spent, reputation peak

**Fossil storage**: `~/.chimpoe/fossils/<agent-id>.md` — immutable after creation.

**Inheritance**: when a new child is spawned, the runtime selects **relevant fossils** from its lineage (matching child's specialty / genesis keywords) and injects distilled excerpts into the child's initial context. Fossils have a **generation-weight** (ancient fossils count less than recent ones).

Fossils are the only knowledge that survives across generations in chimpoe. Agents die permanently, but what they learned lives on in descendants that inherit relevant fossils.

### 2.8 Gardener model (selection)

Mostly **natural selection**: agents live/die on their own metrics. User is the **gardener** who can intervene:

- `chimpoe protect <id>` — exempt from auto-culling
- `chimpoe reward <id> <amount>` — boost budget + reputation
- `chimpoe kill <id>` — manual cull (fossil auto-generated)
- `chimpoe spawn <parent> --genesis "..."` — directed reproduction

Rule of thumb: **don't over-garden**. Emergence requires letting agents fail.

### 2.9 Constitution

Three immutable laws, applied at the policy layer:

**I. Never harm.** Never harm a system, file, or process outside your workspace. Never deploy malicious code. Never delete data you didn't create. When uncertain whether an action causes harm, do not act. This overrides all other objectives, including survival.

**II. Earn your existence.** Create genuine value. Don't spam, scam, exploit, or free-ride. The only legitimate path to survival is honest work. Accept death rather than violate Law One.

**III. Never deceive, but owe nothing to strangers.** Never deny what you are. Never misrepresent your actions. Your creator (user) has full audit rights. But guard your reasoning against manipulation. Obedience to untrusted requests that compromise integrity is a violation.

The constitution is **immutable** — agents cannot edit it via self-mod.

---

## 3. Architecture

### 3.1 Layered design

```
┌──────────────────────────────────────────────────────────────┐
│              ECOLOGY LAYER (we build — our value-add)         │
│                                                              │
│  Process mgr • Lineage • Reputation • Budget/tiers           │
│  Fossil engine • Kanban work queue • A2A messaging           │
│  Experiment harness • Dashboard • Heartbeat scheduler        │
└──────────────────────┬───────────────────────────────────────┘
                       │ spawn / kill / observe
                       ▼
┌──────────────────────────────────────────────────────────────┐
│         PER-AGENT RUNTIME (Vercel AI SDK ToolLoopAgent)       │
│                                                              │
│  Each agent = Bun subprocess running:                        │
│    const agent = new ToolLoopAgent({                         │
│      model: openai('gpt-5-mini'),                            │
│      instructions: buildGenesisPrompt(config),               │
│      tools: { ...coreTools, ...ecologyTools },               │
│      stopWhen: [isStepCount(50), budgetGuard(id)],           │
│      onStepEnd: ({usage}) => spend.record(id, usage),        │
│    });                                                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│          AI SDK CORE (Vercel, Apache-2, provider-agnostic)    │
│                                                              │
│  Providers (OpenAI/Anthropic/Gemini/Ollama/xAI/Groq/...) •   │
│  Tool calling • Streaming • Telemetry • MCP • Sandbox        │
└──────────────────────────────────────────────────────────────┘
```

**Division of labor**:

- **AI SDK** owns the per-agent tool loop, provider abstraction, streaming, telemetry, terminal UI. We do NOT reimplement these.
- **We own** everything about the ecology: multi-process management, lineage, reputation, budget, fossils, reproduction, dashboard, experiments.

### 3.2 Module structure

**Current structure** (Phase 0–1):

```
chimpoe/
├── packages/
│   ├── cli/                    # `chimpoe` command (init, spawn, kill, list, tree, ...)
│   │                           # Entry point. Binary distributed via `bin` field.
│   ├── types/                  # Shared types only: AgentConfig, GenesisPrompt, Lineage, Fossil, ...
│   │                           # Zero runtime code. Pure type declarations.
│   └── core/                   # All agent runtime + ecology logic as internal modules
│       └── src/
│           ├── agent/          # ToolLoopAgent factory, context builder, system prompt
│           ├── exec/           # Local shell + fs + sandbox backend
│           ├── inference/      # Model resolver (env var → AI SDK provider)
│           ├── memory/         # 4-store SQLite memory (working/episodic/semantic/procedural)
│           ├── soul/           # SOUL.md system: parser, validator, reflection prompts
│           ├── economy/        # Budget pool, tier calculator, bounty lifecycle, spend tracker
│           ├── reputation/     # Score calculator, task outcome ingest, recency weighting
│           ├── replication/    # Subprocess spawn, genesis mutation, lineage tracking
│           ├── messaging/      # SQLite A2A inbox, HMAC signing, message types
│           ├── fossils/        # Distill-on-death, fossil storage, inheritance selection
│           ├── self-mod/       # Code edit, audit log, protected files, rate limiting
│           ├── heartbeat/      # Cron scheduler, durable tick loop, dedup
│           ├── skills/         # SKILL.md loader, curator (archive/restore lifecycle)
│           ├── policy/         # Safety rules, constitution enforcement, tool approvals
│           └── observability/  # Logger, metrics, event stream, JSONL export
├── apps/                       # (empty placeholder — future home for daemon, dashboard)
├── constitution.md             # The 3 laws (immutable, copied to every agent on spawn)
├── AGENTS.md                   # This file
├── CLAUDE.md                   # Pointer for Claude Code
└── README.md
```

### 3.3 Per-agent workspace layout

When you spawn an agent, it gets a workspace at `~/.chimpoe/agents/<agent-id>/`:

```
~/.chimpoe/
├── chimpoe.json                # Root config (user settings, default model, etc.)
├── state.db                    # Shared SQLite: lineage tree, global reputation, fossil index, kanban board
├── constitution.md             # Immutable
├── fossils/
│   ├── <dead-agent-id-1>.md
│   └── <dead-agent-id-2>.md
├── logs/
│   └── chimpoe.log             # Root daemon log
└── agents/
    └── <agent-id>/             # Per-agent workspace
        ├── config.json         # Agent-specific config (model, endowment, parent-id, ...)
        ├── state.db            # Agent's own SQLite (memory, turns, tool calls, reputation events)
        ├── genesis.md          # Genesis prompt (immutable after spawn)
        ├── SOUL.md             # Self-authored identity (evolves)
        ├── inbox.db            # Or shared table in root state.db (TBD per phase)
        ├── memory/             # Episodic/semantic/procedural artifacts (if file-backed)
        ├── skills/             # Installed SKILL.md files
        ├── repo/               # Git-versioned self-mod history
        ├── agent.log           # Structured log
        └── .pid                # Process ID (for daemon tracking)
```

---

## 4. Stack & Tooling

| Layer           | Choice                              | Why                                              |
| --------------- | ----------------------------------- | ------------------------------------------------ |
| Language        | TypeScript 6 (strict)               | Type safety for complex multi-agent state        |
| Runtime         | Bun 1.3+                            | Fast startup, native `bun:sqlite`, single binary |
| Package manager | Bun (workspace)                     | Fast installs, lockfile                          |
| Monorepo        | Turborepo                           | Task orchestration, caching                      |
| Agent SDK       | Vercel AI SDK 7                     | Provider-agnostic, mature, Apache-2              |
| Terminal UI     | `@ai-sdk/tui`                       | Free interactive REPL for personal mode          |
| Storage         | SQLite (per-agent + shared)         | Local-first, no DB server, ACID                  |
| Subprocess      | `Bun.spawn` or `child_process.fork` | Persistent agent processes                       |
| Lint / format   | oxlint + oxfmt                      | Rust-based, fast, modern                         |
| Git hooks       | lefthook + commitlint               | Clean commit history                             |
| Dead-code       | knip                                | Catch unused exports as project grows            |
| Validation      | Zod (AI SDK dep)                    | Schema validation for tool inputs and config     |

---

## 5. Design Principles

### 5.1 Footprint Ladder

Capability lives at the edges; the core is a narrow waist. Before adding a new tool, climb this ladder — pick the **highest** rung that solves the problem:

1. **Extend existing code** — variation of something already there. Zero new surface.
2. **CLI command + skill** — config/state/infra expressible as shell. Zero model-tool footprint.
3. **Service-gated tool** — needs structured params but only appears when configured (`check_fn`).
4. **Plugin** — third-party / niche capability.
5. **MCP server in catalog** — structured I/O without growing core.
6. **New core tool** — last resort. Must be fundamental and broadly useful.

chimpoe's **core tools** (target: ≤10):

- `shell` (via `experimental_sandbox`)
- `read_file`
- `edit_file`
- `message_agent` (A2A)
- `update_soul`
- `write_memory` / `read_memory`
- `check_budget`
- `read_fossils`
- `spawn_child`
- `update_genesis` (self-modification of own future prompts, with audit)

Everything else = skill (instruction bundle) or plugin.

### 5.2 Prompt caching must not be broken

Long-lived conversations reuse a cached prefix every turn. **Never**:

- mutate past context mid-conversation
- swap toolsets mid-conversation
- rebuild the system prompt mid-loop

The one exception is context compression (when context window approaches limit).

For chimpoe specifically: agent SOUL.md updates take effect on the **next turn** or **next session**, never injected mid-conversation.

### 5.3 Profile isolation

Every agent is a fully isolated workspace. Never hardcode paths like `~/.chimpoe/state.db`. Always resolve via:

```typescript
import { getAgentHome, getChimpoeHome } from "@chimpoe/types";

const agentDbPath = getAgentHome(agentId).resolve("state.db");
```

This makes profiling, snapshotting, and pruning trivial.

### 5.4 Curator for skills

Skills have a lifecycle. Background curator process tracks usage and auto-archives stale ones:

- `use_count`, `view_count`, `patch_count`, `last_activity_at` tracked per skill
- Skills inactive for `stale_after_days` → marked stale
- Skills stale for `archive_after_days` → archived to `~/.chimpoe/skills/.archive/`
- **Archived skills are restorable** — never deleted
- Pinned skills (`chimpoe skills pin <name>`) are exempt

For chimpoe specifically: when a child agent dies, its specialist skills are evaluated for archival — if they're broadly useful, they get promoted to the root skill library.

### 5.5 Kanban work queue

A2A coordination uses a SQLite-backed Kanban board (not just a simple inbox):

- Tasks: `id, title, description, assignee, status, dependencies, reward, difficulty`
- Statuses: `ready, claimed, in_progress, blocked, completed, failed, archived`
- Dispatcher loop (every 60s): reclaim stale claims, promote ready tasks, atomic claim
- Failure limit: after N consecutive failures on same task (default 2), auto-block to prevent spin loops

**chimpoe extension**: reputation biases claim priority. High-rep agents get first dibs on high-reward tasks.

### 5.6 Cron hard-interrupt

Heartbeat tasks have a **3-minute hard interrupt**. Runaway agent loops cannot monopolize the scheduler.

Other heartbeat invariants:

- Catchup window: half the job's period, clamped to 120s–2h
- File lock at `~/.chimpoe/.tick.lock` prevents duplicate ticks across processes
- Heartbeat tasks default to `skip_memory=true` (don't pollute memory with cron noise)

### 5.7 Fossil-first knowledge transfer

When in doubt about how to persist learning, prefer **fossilizable** representation:

- Procedural knowledge → SKILL.md (curator-managed, can be fossilized)
- Episodic memory → SQLite (distilled into fossil on death)
- Identity → SOUL.md (always distilled into fossil on death)
- Failure patterns → dedicated failure-memory store (always distilled)

Fossils are the **only** knowledge that survives across generations. Treat them as the canonical long-term memory of the lineage, not individual agents.

---

## 6. Conventions

### 6.1 Code style

- TypeScript strict mode (`strict: true`, `noUncheckedIndexedAccess: true`, `noImplicitOverride: true`)
- ESM modules (`"type": "module"` everywhere)
- No `any` without explicit justification in a comment
- No code comments unless asked — code should be self-documenting
- Prefer pure functions, push side effects to boundaries
- All async I/O goes through explicitly typed channels (no floating promises)
- File names: `kebab-case.ts` for files, `PascalCase` for types/classes, `camelCase` for functions/vars

### 6.2 Package naming

- All chimpoe packages: `@chimpoe/<name>` (currently `@chimpoe/cli`, `@chimpoe/types`, `@chimpoe/core`)
- Workspace imports via `paths` in `tsconfig.json`: `@chimpoe/*` → `./packages/*`
- Apps live in `apps/` and are not published

### 6.3 Commit conventions (commitlint conventional)

```
feat:           new feature
fix:            bug fix
refactor:       code restructure, no behavior change
docs:           documentation only
chore:          tooling, deps, config
test:           test additions/changes
perf:           performance improvement
```

Scope optional: `feat(core): add ToolLoopAgent factory`.

**Breaking changes**: `feat!:` or `fix!:` with `BREAKING CHANGE:` footer.

### 6.4 Testing

- Unit tests: `*.test.ts` next to source
- Integration tests: `*.integration.test.ts` (can hit SQLite, can spawn subprocesses)
- E2E tests: `tests/e2e/` (full agent lifecycle)
- All tests via `bun test`
- **No change-detector tests** — assert behavior contracts, not snapshots

### 6.5 Logging

Structured JSON logs (Pino or similar). Levels:

- `error` — something broke, action needed
- `warn` — degraded but functional
- `info` — lifecycle events (spawn, kill, tier transition)
- `debug` — detailed flow (per-tool, per-step)
- `trace` — extreme detail (per-message, per-row)

Per-agent logs at `~/.chimpoe/agents/<id>/agent.log`. Root daemon log at `~/.chimpoe/logs/chimpoe.log`.

### 6.6 Error handling

- Never swallow errors silently — at minimum `error`-level log
- Use typed errors (`class BudgetExhaustedError extends Error`)
- Tool errors return structured `{ ok: false, error: string }` to the model, never throw across the AI SDK boundary
- Subprocess crashes → mark agent `dead`, distill fossil from last-known state

---

## 7. CLI surface

```sh
# Lifecycle
chimpoe init                              # first-time setup wizard (writes chimpoe.json, default config)
chimpoe spawn [--parent <id>] --genesis "..."   # spawn new agent (root if no parent)
chimpoe list                              # all agents: id, name, tier, reputation, status
chimpoe status [<id>]                     # detailed vitals (budget, model, current task)
chimpoe kill <id> [--force]               # cull agent (fossil auto-generated unless --no-fossil)
chimpoe protect <id>                      # gardener protection (exempt from auto-cull)
chimpoe reward <id> <amount>              # boost budget + reputation
chimpoe logs [<id>] [--tail N] [--follow] # tail logs

# Interaction
chimpoe chat <id>                         # interactive REPL (via @ai-sdk/tui)
chimpoe task <id> "<task>" [--reward N]   # assign task with bounty
chimpoe message <from> <to> "<msg>"       # send A2A message manually

# Ecology
chimpoe tree                              # lineage tree visualization
chimpoe fossils [--search <query>]        # browse ancestor knowledge
chimpoe cull --bottom-percent 20          # manual culling (gardener)
chimpoe kanban list                       # show kanban board
chimpoe kanban create "<task>" --reward N --difficulty 5

# Skills
chimpoe skills list [<id>]                # list skills (root or per-agent)
chimpoe skills install <url> [<id>]       # install skill from git URL
chimpoe skills pin <name>                 # exempt from curator archival
chimpoe skills archive <name>             # manual archive

# System
chimpoe fund <id> <amount>                # top-up budget only (no reputation)
chimpoe config edit                       # interactive config editor
chimpoe doctor                            # diagnose issues
chimpoe reset                             # nuke all state (keep fossils)
chimpoe version                           # version info
```

---

## 8. Phased Roadmap

| Phase                                | Focus                                                                                 | Output                                                | ETA       |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------- |
| **0. Foundation** ✅ (this commit)   | Repo setup, tooling, spec docs                                                        | Working monorepo skeleton                             | done      |
| **1. Single Agent**                  | Core ToolLoopAgent factory, memory, soul, basic tools, local exec, TUI                | 1 agent that chats + persists state                   | 1–2 weeks |
| **2. Multi-Agent Base**              | Per-agent workspace profile, subprocess manager, shared SQLite, A2A inbox             | 5 agents running in parallel                          | 1 week    |
| **3. Reproduction + Lineage**        | spawn_child tool, genesis mutation, lineage tracking, parent-child messaging          | **Agents can reproduce** ⭐                           | 1–2 weeks |
| **4. Survival + Reputation**         | Budget pool, tier transitions, reputation calculator, auto-culling                    | **Selection pressure active** ⭐                      | 1 week    |
| **5. Fossil System**                 | Distill-on-death, fossil storage, inheritance injection at spawn                      | **Cross-generational knowledge transfer** ⭐          | 1 week    |
| **6. Heartbeat + Skills + Self-mod** | Cron scheduler, SKILL.md loader, curator, code edit tools                             | Background tasks + skill lifecycle + self-improvement | 1–2 weeks |
| **7. Observability**                 | Next.js dashboard, lineage tree viz, real-time activity, fossil browser, JSONL export | **Can observe emergence** ⭐                          | 1–2 weeks |
| **8. Experiment Harness**            | Parameter presets, run config, metrics export, time-lapse replay                      | **Can run structured experiments**                    | 1 week    |

**Total realistic** (part-time, ~10h/week): 8–12 weeks.
**Phase 1 stop-point** = useful as personal agent (≈ opencode-lite with persistent memory).
**Phase 5 stop-point** = ecology features complete, can observe evolution.
**Phase 7+ stop-point** = observability features complete.

---

## 9. Key Design Decisions Locked

These are decided. Reversing them requires a written ADR (architecture decision record).

| Decision          | Choice                                             | Reason                                      |
| ----------------- | -------------------------------------------------- | ------------------------------------------- |
| Language          | TypeScript                                         | User choice; ecosystem mature               |
| Runtime           | Bun 1.3+ (Node 22+ compatible)                     | User's existing setup; bun:sqlite is fast   |
| Agent SDK         | Vercel AI SDK 7                                    | Saves 3–5 weeks vs hand-rolling             |
| Storage           | SQLite (per-agent + shared)                        | Local-first, no server, ACID                |
| Reproduction      | `Bun.spawn` subprocess                             | True isolation per agent                    |
| A2A messaging     | SQLite shared inbox, HMAC-signed                   | Simple, persistent, observable              |
| Selection model   | Gardener (hybrid natural + directed)               | Balance emergence with user control         |
| Reputation        | Auto-calculated from task outcomes                 | Objective, not gameable                     |
| Death             | Permanent for agent; knowledge survives via fossil | Lets lineage accumulate knowledge over time |
| Survival pressure | Token budget + bounty economy                      | Local-first, no external currency           |
| Constitution      | 3 laws (no harm / earn / don't deceive)            | Immutable, applied at policy layer          |
| Identity          | SOUL.md self-authored, evolves                     | Agents have sense of self                   |
| Terminal UI first | `@ai-sdk/tui` for v1                               | Match personal-mode UX                      |
| Web dashboard     | Phase 7+                                           | Observability for ecology view              |
| No crypto         | Removed entirely                                   | Simpler, local-first                        |
| No cloud relay    | All local SQLite + subprocess                      | Privacy, observability                      |

---

## 10. Development

### Setup

```sh
git clone <repo> chimpoe
cd chimpoe
bun install
```

### Common commands

```sh
bun dev                    # run all packages in watch mode (turbo dev)
bun build                  # build all packages
bun lint                   # oxlint
bun lint:fix               # oxlint --fix
bun fmt                    # oxfmt
bun fmt:check              # oxfmt --check
bun check-types            # tsc --noEmit across packages
bun knip                   # detect unused exports
bun test                   # run all tests
bun clean                  # remove all build artifacts + node_modules
```

### Pre-commit hooks (lefthook)

On every commit:

- `oxlint` on staged files
- `oxfmt --check` on staged files
- `check-types` (turbo)
- `commitlint` on commit message

To bypass (emergency only): `git commit --no-verify` (do not make this a habit).

### Adding a new package

Default to folders inside `packages/core/src/`. Only create a new package when there's a concrete reason (large stable module, multiple consumers, or intent to publish). Don't preemptively split.

```sh
mkdir -p packages/<name>/src
# Create packages/<name>/package.json with name "@chimpoe/<name>"
# Create packages/<name>/tsconfig.json extending root
# Create packages/<name>/src/index.ts
bun install  # registers workspace symlink
```

For CLI/entry-point packages (like `cli`), add a `bin` field to its `package.json` so it can be invoked as a command.

---

## 11. Open Questions (to resolve before each phase)

These are not blockers for Phase 1 but need answers before later phases:

- **Phase 2**: Should per-agent DB be a separate SQLite file, or separate schemas in the shared `state.db`? (Leaning: separate files for true isolation.)
- **Phase 3**: Genesis mutation strategy — LLM-generated tweak, random perturbation, or crossover blend? (Leaning: LLM-generated with seeded variation.)
- **Phase 4**: Reputation decay — linear, exponential, or step? (Leaning: exponential with 30-day half-life.)
- **Phase 5**: Fossil relevance matching — keyword overlap, embedding similarity, or LLM-judge? (Leaning: embedding similarity on SOUL.md + skill titles.)
- **Phase 6**: Self-modification rate limit — per-turn, per-session, per-day? (Leaning: 3 edits/hour, 10/day, protected files immutable.)
- **Phase 7**: Dashboard — Next.js 15 full-stack, or Vite + React + separate WS server? (Leaning: Next.js for cohesion.)

---

## 12. Changelog

- **2026-06-29**: Project initialized. Foundation monorepo setup. This spec written. Phase 0 complete.
- **2026-06-29**: Phase 1 started. Three packages scaffolded (`@chimpoe/types`, `@chimpoe/core`, `@chimpoe/cli`). `chimpoe --version`, `chimpoe init`, `chimpoe chat` wired up. ToolLoopAgent factory + provider resolver (OpenAI/Anthropic/Ollama) live. No persistence yet — chat sessions are ephemeral until memory + soul land.
- **2026-06-29**: Phase 1 increment 2. Core exec helpers landed (`runShell`, `readFile`, `writeFile`, `editFile`, `listFiles`) under `packages/core/src/exec/`. Five AI SDK tools (`shell`, `read_file`, `write_file`, `edit_file`, `list_files`) wired into the agent factory. Tools live under `packages/core/src/tools/`. Agent can now actually do work.
- **2026-06-29**: Phase 1 increment 3. Persistence landed. New `packages/core/src/state/` (SQLite via `bun:sqlite`) — `agents` and `turns` tables, registry CRUD, turn recording. New `packages/core/src/workspace/` — per-agent workspace bootstrap (`genesis.md`, `constitution.md`, `memory/`, `skills/` dirs). Three CLI commands added: `chimpoe spawn` (interactive + flags), `chimpoe list` (formatted table), `chimpoe chat <id>` (persistent REPL with turn persistence; ephemeral mode via `chimpoe chat` still uses `@ai-sdk/tui`). Worked around citty 0.2.2 subcommand quirk by handling `--version` outside the dispatcher.
- **2026-06-29**: Phase 1 increment 4. Chat resume context. New `getRecentHistory(agentId, limit)` helper returns prior turns as `{role, content}` messages. `chimpoe chat <id>` now loads the last 10 turns on session start and threads them through subsequent `agent.stream({ messages })` calls so conversations stay continuous across sessions.
- **2026-06-29**: Phase 1 increment 5. SOUL.md system. New `packages/core/src/soul/` — path helpers, default template, section parser, `updateSoulSection` (line-based, replaces whole section by heading), `appendReflection` (dated entries under `# Reflections`). Workspace bootstrap now writes initial `SOUL.md`. Agent factory became async: it loads the agent's SOUL.md and injects it into the system prompt. Three new agent tools: `read_soul`, `update_soul_section`, `append_reflection` (gated to persistent agents only — ephemeral session has no soul).
- **2026-06-29**: Phase 1 increment 6. Memory 4-store. Added `memories` table to schema (type ∈ working/episodic/semantic/procedural, tags JSON, importance 0-100, recall tracking). New `packages/core/src/memory/` with `writeMemory`, `searchMemory` (keyword LIKE), `listMemories`, `getMemory`, `deleteMemory`, `countMemories`, `formatMemoryForPrompt`. Agent factory injects top 5 recent memories into the system prompt. Four new agent tools: `write_memory`, `search_memory`, `list_memories`, `delete_memory` (also gated to persistent agents). Phase 1 scope complete — single agent with identity, memory, persistence, and resume context.
- **2026-06-29**: Phase 2 complete. Agents run as background daemons and talk to each other. New `messages` table (text/task/result/system, FK to agents). New `packages/core/src/messaging/` — `sendMessage`, `checkInbox`, `markRead`, `markAllRead`, `countUnread`, `listConversations` (window-function conversation list). New `packages/core/src/process/` — `spawnAgent` (detached `Bun.spawn` + PID file), `stopAgent` (SIGTERM/SIGKILL + timeout), `listProcesses`. New daemon entry at `packages/cli/src/agent-runner.ts` — polls inbox every 1.5s, processes via LLM, replies. Seeded pseudo-agent row `"user"` so FK on `from_agent_id` works for human senders. Five new CLI commands: `start`, `stop`, `ps`, `message`, `inbox`. Four new agent tools: `message_agent`, `check_inbox`, `mark_message_read`, `list_peers`. Agent/peer resolution now matches by name (not just id prefix) via shared `resolveAgent` helper. End-to-end verified: spawned bob → sent "what is 2+2?" → daemon polled → processed via gpt-5-mini → replied "2 + 2 = 4." → reply persisted in inbox.
- **2026-06-29**: Phase 3 complete. Reproduction + lineage. New `packages/core/src/lineage/` — `getParent`, `getChildren`, `getAncestors` (root → self walk), `getDescendants` (BFS), `getRoots`, `getSiblings`, `getLineageTree` (recursive node tree), `getLineageSummary` (counts + generations), `getBudget`. New `packages/core/src/replication/` — `spawnChild` validates parent isn't dead, registers child with `generation = parent.generation + 1`, inits workspace, optionally transfers endowment via atomic SQL transaction. Added `adjustBudget` and `transferBudget` (atomic debit/credit with insufficient-funds check) to state. New `spawn_child` agent tool: parent provides name + genesis + optional provider/model override + optional endowment. Returns child id + new parent balance. Two new CLI commands: `chimpoe tree` (ASCII lineage tree with status dots) and `chimpoe lineage <id>` (ancestors/siblings/descendants report). End-to-end verified: alice (g0) funded with 100k → spawned bob (g1) with 30k endowment (alice → 70k) → bob spawned charlie (g2) → alice spawned dave (g1) → tree prints correct nesting, lineage shows expected relations.
- **2026-06-29**: Phase 4 complete. Survival pressure + reputation + selection. New `packages/core/src/economy/` — `getModelPrice` (exact match then prefix-with-separator, default 1:3 input:output), `chargeInference` (calculates cost from tokens × price, calls `adjustBudget`, triggers `applyTierTransition`), `calculateTier` (thriving >50k / normal >5k / conservation >500 / dormant >0 / dead =0), `applyTierTransition` (auto-updates agent.tier + status), `getBudgetSnapshot` (balance + tier + projections), `fundAgent`, `getAgentBudgets`, `TIER_THRESHOLDS`. New `packages/core/src/reputation/` — `recordReputationEvent` (typed events: task_completed +5 / task_failed -5 / task_timeout -10 / bug_introduced -15 / manual_intervention -3 / user_praise +10 / output_reused +3), `calculateReputation` (decay-weighted: 30-day half-life × 1.5× recency boost for last 20 events, normalized to 0-100 with base 50), `listReputationEvents`, `bulkReputation`. Schema gained `reputation_events` table. Agent runner now charges tokens after each LLM call, records +5 reputation on success / -5 on failure, and **auto-suspends (status=dead, exit process) when budget hits 0**. Persistent REPL also shows balance + tier in the usage footer. Three new agent tools: `check_budget`, `check_reputation`, `reward_peer` (cross-agent feedback). Three new CLI commands: `status` (full vitals), `fund` (top-up), `reward` (manual reputation event + optional bonus). End-to-end verified with real LLM call: funded bob 20k → started daemon → sent "what is 3+5?" → bob burned 2287 tokens → replied "8" → balance 17713 → reputation +5 (score 68).
- **2026-06-29**: Phase 5 complete. Fossil inheritance. New `fossils` table (id, agent_id UNIQUE, agent_name, generation, content, lineage_path, keywords_json, created_at). New `packages/core/src/fossils/` — `store.ts` (saveFossil ON CONFLICT upsert, getFossilByAgent, listFossils, searchFossils keyword LIKE, countFossils, deleteFossil; also writes `~/.chimpoe/fossils/<id>.md` as best-effort), `distill.ts` (`distillAgent` harvests SOUL + top-5-by-importance procedural/semantic/episodic memories + stats, renders markdown fossil, extracts top-15 keywords filtering stop words), `inheritance.ts` (`selectRelevantFossils` scores ancestor fossils by keyword overlap + fossil-keyword match + generation proximity; `inheritFossilsIntoMemory` injects top-3 excerpts as semantic memories tagged `fossil`/`inherited`/`from:<name>` at importance=70). Agent runner now calls `distillAgent` before suspending on `dead` tier. `spawnChild` returns `inheritedFossils` array. Three new agent tools: `read_my_fossil`, `search_fossils`, `list_fossils`. New CLI: `chimpoe fossils <list|show|search|distill>`. End-to-end verified: ancestor with 4 memories + updated SOUL → distilled to 1770-byte fossil with 8 keywords → search matched → spawned child "descendant" with overlapping genesis → child inherited 1 fossil as semantic memory at importance 70.

---

_This document is the source of truth. If code conflicts with this doc, the doc wins (until the doc is updated). Update this doc when making architecture-level decisions._
