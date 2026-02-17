// Pipeline action - orchestrates full pipeline execution
// Runs in V8 isolate (no "use node")

import { action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

// Parse avg sentences string (e.g. "4-7" → 5.5, "5" → 5) into numeric midpoint
function parseAvgSentences(val: string): number {
  const rangeMatch = val.match(/^(\d+)\s*-\s*(\d+)$/);
  if (rangeMatch) return (parseInt(rangeMatch[1]) + parseInt(rangeMatch[2])) / 2;
  const num = parseFloat(val);
  return isNaN(num) || num <= 0 ? 5 : num;
}

/**
 * Run the full pipeline (selected phases) for a job.
 *
 * Calls the Python API's /api/run-pipeline endpoint which
 * executes composition, generation, and/or evaluation sequentially.
 */
export const runPipeline = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    // Get settings (for API keys)
    const settings = await ctx.runQuery(api.settings.get, {});
    const phases = job.phases || ["composition", "generation", "evaluation"];

    // Validate required API keys based on phases
    if (phases.includes("composition") || phases.includes("generation")) {
      if (!settings.openrouterApiKey) {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: "OpenRouter API key not configured. Please set it in Settings.",
        });
        return { success: false, error: "OpenRouter API key not configured" };
      }
    }

    const jobsDirectory = settings.jobsDirectory || "./jobs";
    const convexAdminKey = settings.convexAdminKey || "";

    if (!convexAdminKey) {
      console.log("WARNING: Convex Admin Key not configured - progress updates will not work");
    }

    // If reusing composition from another job, get source job's data
    let reusedFromJobDir: string | null = null;
    if (job.reusedFrom && !phases.includes("composition")) {
      const sourceJob = await ctx.runQuery(api.jobs.getSourceJobConfig, { jobId: job.reusedFrom });
      if (sourceJob?.jobDir) {
        reusedFromJobDir = sourceJob.jobDir;
        await ctx.runMutation(api.logs.add, {
          jobId: args.jobId,
          level: "INFO",
          phase: "Pipeline",
          message: `Reusing composition data from: ${sourceJob.name}`,
        });
      }
    }

    // Mark job as composing if composition is first phase, otherwise running
    const jobDir = job.jobDir || `${jobsDirectory}/${args.jobId}-${sanitizeJobName(job.name)}`;

    // Handle heuristic vs CERA method differently
    const isHeuristic = job.method === "heuristic";

    const isReal = job.method === "real";

    if (isReal) {
      // Real eval-only jobs: set jobDir and mark as evaluating
      try {
        await ctx.runMutation(api.jobs.setJobDir, { jobId: args.jobId, jobDir: jobDir });
        await ctx.runMutation(api.jobs.startEvaluation, { id: args.jobId });
      } catch (e) {
        // Job might already be in a valid state
      }
    } else if (isHeuristic) {
      // Heuristic jobs skip composition, go straight to generation
      const totalReviews = job.heuristicConfig?.targetMode === "reviews"
        ? job.heuristicConfig.targetValue
        : Math.ceil((job.heuristicConfig?.targetValue || 100) / (parseAvgSentences(job.heuristicConfig?.avgSentencesPerReview || '5')));
      const totalBatches = Math.ceil(totalReviews / (job.heuristicConfig?.reviewsPerBatch || 50));

      try {
        await ctx.runMutation(api.jobs.startHeuristicGeneration, {
          jobId: args.jobId,
          jobDir: jobDir,
          totalBatches: totalBatches,
        });
      } catch (e) {
        // Job might already be in a valid state
      }
    } else if (phases.includes("composition")) {
      try {
        await ctx.runMutation(api.jobs.startComposing, {
          jobId: args.jobId,
          jobDir: jobDir,
        });
      } catch (e) {
        // Job might already be in a valid state
      }
    }

    const pythonApiUrl = "http://cli:8000";

    // Debug logging for multi-run and multi-model
    if (!isHeuristic && job.config?.generation) {
      console.log(`[Pipeline] DEBUG: job.config.generation.total_runs = ${job.config.generation.total_runs}`);
      console.log(`[Pipeline] DEBUG: job.config.generation.models = ${JSON.stringify(job.config.generation.models)}`);
      console.log(`[Pipeline] DEBUG: job.config.generation.parallel_models = ${job.config.generation.parallel_models}`);
    }

    // Only pass Tavily API key if Tavily is explicitly enabled
    const tavilyApiKey = (settings.tavilyEnabled !== false && settings.tavilyApiKey) ? settings.tavilyApiKey : null;

    try {
      const response = await fetch(`${pythonApiUrl}/api/run-pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: (isHeuristic || (phases.length === 1 && phases[0] === "evaluation")) ? null : job.config, // Heuristic and eval-only jobs don't use CERA config
          phases: phases,
          apiKey: settings.openrouterApiKey || "",
          tavilyApiKey: tavilyApiKey,
          jobsDirectory: jobsDirectory,
          convexUrl: "http://convex:3210",
          convexToken: convexAdminKey,
          evaluationConfig: job.evaluationConfig || null,
          datasetFile: job.datasetFile || null,
          reusedFromJobDir: reusedFromJobDir,
          referenceDataset: job.referenceDataset || null,
          // Heuristic method fields
          method: job.method || "cera",
          heuristicConfig: job.heuristicConfig || null,
          // Pre-job RDE token usage (from context extraction)
          rdeUsage: job.rdeUsage || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Pipeline failed: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      return { success: true, jobId: args.jobId, phases };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      try {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Pipeline failed: ${errorMessage}`,
        });
      } catch (e) {
        // Ignore failures in error handling
      }

      return { success: false, error: errorMessage };
    }
  },
});

// Helper function to sanitize job name for directory
function sanitizeJobName(name: string): string {
  return (name || 'job')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .substring(0, 30) || "job";
}
