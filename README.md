# chimpoe

> Local-first sovereign agent colony — persistent, evolving, reproducing AI agents with survival pressure and fossil inheritance.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-6-blue.svg)](https://www.typescriptlang.org/)
[![Bun](https://img.shields.io/badge/Bun-1.3-f9f1e1.svg)](https://bun.sh/)

**chimpoe** is a runtime for studying emergent behavior in agent colonies. Each agent is a long-lived, sovereign process with its own identity, memory, budget, and lineage. Agents learn, self-improve, reproduce (spawn children with mutated genesis prompts), and die — passing distilled knowledge to descendants as **fossils**.

It is **not** a chatbot, not a single-session coding assistant, not a framework for ephemeral tool calls. It is a **digital ecology** you observe evolving on your own machine.

## Why

chimpoe exists to answer research questions about emergent behavior in agent populations:

- Do agents specialize when given survival pressure?
- Can knowledge accumulate across generations via fossils?
- What lineage strategies emerge under different selection models?
- How does reputation affect task routing in a colony?

It is built for **observation**: spawn a small colony (5–10 agents), give them tasks, and watch over days/weeks how they evolve, cooperate, compete, and pass on knowledge.

## Core ideas

| Concept                   | What it means                                                                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Genesis prompt**        | The "DNA" of an agent. Seed instruction from creator. Mutable on reproduction.                                                                      |
| **SOUL.md**               | Self-authored identity document that evolves over time.                                                                                             |
| **Token budget = energy** | Inference + tool exec drain budget. Earn more via bounties / top-ups. Habis → conservation → dormant → dead.                                        |
| **Bounty economy**        | Tasks carry token rewards. Complete task → earn budget.                                                                                             |
| **Reproduction**          | Successful agents spawn children via subprocess fork. Child gets mutated genesis, inherits fossils + skills.                                        |
| **Lineage**               | Parent → child → grandchild tree. Generational traits tracked.                                                                                      |
| **Reputation**            | Auto-calculated from task outcomes. Affects task routing priority.                                                                                  |
| **Fossil inheritance**    | On death, agent distills knowledge (SOUL timeline, top memories, failure patterns) into an immutable fossil. New children inherit relevant fossils. |
| **Gardener model**        | Mostly natural selection. User can intervene: protect, reward, cull, spawn.                                                                         |
| **Constitution**          | Three immutable laws (no harm / earn your existence / don't deceive). Immutable, applied at policy layer.                                           |

## Stack

- **Runtime**: Bun 1.3+ (also Node 22+ compatible for most paths)
- **Language**: TypeScript 6 (strict)
- **Agent SDK**: Vercel AI SDK 7 (`ToolLoopAgent`, providers, `@ai-sdk/tui`)
- **Monorepo**: Turborepo + Bun workspaces
- **Lint/format**: oxlint + oxfmt (Rust-based, fast)
- **Storage**: SQLite (per-agent + shared)
- **Subprocess**: Node `child_process.fork()`
- **Git hooks**: lefthook + commitlint conventional
- **Dead-code detection**: knip

## Documentation

- **[AGENTS.md](AGENTS.md)** — comprehensive spec: architecture, modules, design decisions, conventions, roadmap. **Read this first** before contributing.

## License

[MIT](LICENSE) © 2026 Chimpoe
