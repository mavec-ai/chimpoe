export const SCHEMA_VERSION = 1;

export const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  genesis_prompt TEXT NOT NULL,
  parent_id TEXT,
  generation INTEGER NOT NULL DEFAULT 0,
  budget_tokens INTEGER NOT NULL DEFAULT 0,
  tier TEXT NOT NULL DEFAULT 'normal',
  status TEXT NOT NULL DEFAULT 'idle',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agents_parent ON agents(parent_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);

CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  user_message TEXT NOT NULL,
  assistant_text TEXT,
  tool_calls_json TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  duration_ms INTEGER,
  status TEXT NOT NULL DEFAULT 'complete',
  created_at INTEGER NOT NULL,
  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_agent_time ON turns(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('working', 'episodic', 'semantic', 'procedural')),
  content TEXT NOT NULL,
  tags_json TEXT,
  importance INTEGER NOT NULL DEFAULT 50,
  created_at INTEGER NOT NULL,
  last_recalled_at INTEGER,
  recall_count INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_agent_type ON memories(agent_id, type);
CREATE INDEX IF NOT EXISTS idx_memories_agent_recent ON memories(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_agent_importance ON memories(agent_id, importance DESC);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  from_agent_id TEXT NOT NULL,
  to_agent_id TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'text' CHECK (type IN ('text', 'task', 'result', 'system')),
  content TEXT NOT NULL,
  in_reply_to TEXT,
  metadata_json TEXT,
  created_at INTEGER NOT NULL,
  read_at INTEGER,
  FOREIGN KEY (from_agent_id) REFERENCES agents(id) ON DELETE CASCADE,
  FOREIGN KEY (to_agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_to_unread ON messages(to_agent_id, read_at);
CREATE INDEX IF NOT EXISTS idx_messages_to_time ON messages(to_agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_from_time ON messages(from_agent_id, created_at);
`;
