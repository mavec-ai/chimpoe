export { openSharedDb, closeSharedDb, withDb } from "./db.ts";
export { SCHEMA_SQL, SCHEMA_VERSION } from "./schema.ts";
export {
  registerAgent,
  getAgent,
  listAgents,
  updateAgentStatus,
  deleteAgent,
  adjustBudget,
  transferBudget,
  type CreateAgentInput,
  type TransferResult,
} from "./agents.ts";
export {
  createTask,
  getTask,
  listTasks,
  claimTask,
  completeTask,
  failTask,
  abandonTask,
  countTasks,
  type TaskStatus,
  type Task,
  type CreateTaskInput,
  type ClaimResult,
  type CompleteResult,
} from "./tasks.ts";
export {
  recordTurn,
  listTurns,
  listTurnsBySession,
  countTurns,
  getRecentHistory,
  type RecordTurnInput,
  type ChatMessage,
} from "./turns.ts";
