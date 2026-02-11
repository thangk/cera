/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as actions_runComposition from "../actions/runComposition.js";
import type * as actions_runPipeline from "../actions/runPipeline.js";
import type * as compositionAction from "../compositionAction.js";
import type * as datasets from "../datasets.js";
import type * as evaluationAction from "../evaluationAction.js";
import type * as generationAction from "../generationAction.js";
import type * as jobs from "../jobs.js";
import type * as llmPresets from "../llmPresets.js";
import type * as logs from "../logs.js";
import type * as pipelineAction from "../pipelineAction.js";
import type * as settings from "../settings.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  "actions/runComposition": typeof actions_runComposition;
  "actions/runPipeline": typeof actions_runPipeline;
  compositionAction: typeof compositionAction;
  datasets: typeof datasets;
  evaluationAction: typeof evaluationAction;
  generationAction: typeof generationAction;
  jobs: typeof jobs;
  llmPresets: typeof llmPresets;
  logs: typeof logs;
  pipelineAction: typeof pipelineAction;
  settings: typeof settings;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
