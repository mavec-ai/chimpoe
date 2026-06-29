export { resolveModel, type ResolvedModel } from "./inference/resolver.ts";
export { createAgent, type CreateAgentOptions } from "./agent/factory.ts";
export {
  loadConfig,
  saveConfig,
  ensureHome,
  configExists,
  makeDefaultConfig,
  type LoadedConfig,
} from "./config/loader.ts";
export {
  runShell,
  formatShellResult,
  readFile,
  writeFile,
  editFile,
  listFiles,
  type ShellResult,
  type RunShellOptions,
  type ReadFileOptions,
  type EditFileOptions,
  type EditFileResult,
  type ListFilesResult,
} from "./exec/index.ts";
export { builtinTools } from "./tools/index.ts";
export {
  getSoulPath,
  readSoul,
  soulExists,
  parseSoulSections,
  updateSoulSection,
  appendReflection,
  renderDefaultSoul,
  type SoulSection,
  type UpdateSectionResult,
} from "./soul/index.ts";
export {
  openSharedDb,
  closeSharedDb,
  withDb,
  registerAgent,
  getAgent,
  listAgents,
  updateAgentStatus,
  deleteAgent,
  recordTurn,
  listTurns,
  listTurnsBySession,
  countTurns,
  getRecentHistory,
  type CreateAgentInput,
  type RecordTurnInput,
  type ChatMessage,
} from "./state/index.ts";
export {
  initAgentWorkspace,
  getAgentWorkspacePath,
  type InitWorkspaceOptions,
} from "./workspace/index.ts";
export {
  writeMemory,
  searchMemory,
  listMemories,
  getMemory,
  deleteMemory,
  countMemories,
  formatMemoryForPrompt,
  type MemoryType,
  type MemoryRecord,
  type WriteMemoryInput,
  type SearchMemoryOptions,
  type ListMemoriesOptions,
} from "./memory/index.ts";
