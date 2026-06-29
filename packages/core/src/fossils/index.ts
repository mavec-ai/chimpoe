export {
  saveFossil,
  getFossil,
  getFossilByAgent,
  listFossils,
  searchFossils,
  countFossils,
  deleteFossil,
  type Fossil,
  type SaveFossilInput,
  type SearchFossilsOptions,
} from "./store.ts";
export {
  distillAgent,
  extractKeywords,
  type DistillOptions,
  type DistillResult,
} from "./distill.ts";
export {
  selectRelevantFossils,
  inheritFossilsIntoMemory,
  type InheritedFossil,
} from "./inheritance.ts";
