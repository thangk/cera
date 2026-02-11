// Evaluation action - runs only the MDQA evaluation phase
// Uses existing pipeline endpoint with phases=["evaluation"]

import { action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

/**
 * Run only the evaluation (MDQA) phase for a job.
 *
 * This is used for:
 * - Rerunning evaluation on a completed job
 * - Running evaluation on uploaded datasets
 * - Re-evaluating after generation changes
 *
 * Prerequisites:
 * - Job must have a jobDir (directory with dataset files)
 * - Dataset files must exist in jobDir/dataset/
 */
export const runEvaluation = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    // Verify job has a directory
    if (!job.jobDir) {
      throw new Error("Job has no directory. Cannot run evaluation without job data.");
    }

    // Check if job has completed generation or has a dataset file
    const validStatuses = ["completed", "failed", "terminated"];
    if (!validStatuses.includes(job.status) && !job.datasetFile) {
      throw new Error(
        `Cannot run evaluation: job status is "${job.status}". ` +
        `Must be "completed", "failed", or "terminated", or have an uploaded dataset.`
      );
    }

    // Get settings
    const settings = await ctx.runQuery(api.settings.get, {});
    const jobsDirectory = settings.jobsDirectory || "./jobs";
    const convexAdminKey = settings.convexAdminKey || "";

    if (!convexAdminKey) {
      console.log("WARNING: Convex Admin Key not configured - progress updates will not work");
    }

    // Add log entry
    try {
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "MDQA",
        message: "Starting evaluation-only rerun...",
      });
    } catch (e) {
      // Ignore log failures
    }

    // Mark job as evaluating
    try {
      await ctx.runMutation(api.jobs.startEvaluation, { id: args.jobId });
    } catch (e) {
      // Job might already be in a valid state
      console.log("Could not start evaluation (may already be running):", e);
    }

    const pythonApiUrl = "http://cli:8000";

    try {
      // Call the pipeline endpoint with only evaluation phase
      const response = await fetch(`${pythonApiUrl}/api/run-pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: null, // Not needed for evaluation-only
          phases: ["evaluation"], // Only run evaluation
          apiKey: settings.openrouterApiKey || "",
          tavilyApiKey: settings.tavilyApiKey || null,
          jobsDirectory: jobsDirectory,
          convexUrl: "http://convex:3210",
          convexToken: convexAdminKey,
          evaluationConfig: job.evaluationConfig || null,
          datasetFile: job.datasetFile || null,
          reusedFromJobDir: null,
          referenceDataset: job.referenceDataset || null,
          method: job.method || "cera",
          heuristicConfig: job.heuristicConfig || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Evaluation failed: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      return { success: true, jobId: args.jobId, phases: ["evaluation"] };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      try {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Evaluation failed: ${errorMessage}`,
        });
      } catch (e) {
        // Ignore failures in error handling
      }

      return { success: false, error: errorMessage };
    }
  },
});

/**
 * Rerun generation phase (and evaluation if it was in original phases).
 *
 * This resets the job to "composed" state and triggers generation.
 * Composition data is preserved. If the original job included evaluation
 * in its phases, evaluation will also run after generation completes.
 */
export const rerunGeneration = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    // Verify job has a directory (means composition ran)
    if (!job.jobDir) {
      throw new Error("Job has no directory. Composition may not have run.");
    }

    // Check if job is in a rerunnable state
    const validStatuses = ["completed", "failed", "terminated"];
    if (!validStatuses.includes(job.status)) {
      throw new Error(
        `Cannot rerun generation: job status is "${job.status}". ` +
        `Must be "completed", "failed", or "terminated".`
      );
    }

    // Get settings
    const settings = await ctx.runQuery(api.settings.get, {});
    if (!settings.openrouterApiKey) {
      await ctx.runMutation(api.jobs.setFailed, {
        jobId: args.jobId,
        error: "OpenRouter API key not configured. Please set it in Settings.",
      });
      return { success: false, error: "OpenRouter API key not configured" };
    }

    const jobsDirectory = settings.jobsDirectory || "./jobs";
    const convexAdminKey = settings.convexAdminKey || "";
    if (!convexAdminKey) {
      console.log("WARNING: Convex Admin Key not configured - progress updates will not work");
    }

    // Determine which phases to run - generation + evaluation if originally included
    const originalPhases = job.phases || ["composition", "generation", "evaluation"];
    const rerunPhases = originalPhases.filter((p: string) => p !== "composition");
    // Ensure generation is included
    if (!rerunPhases.includes("generation")) {
      rerunPhases.unshift("generation");
    }

    // Reset job to composed state (preserves composition, restarts generation)
    try {
      await ctx.runMutation(api.jobs.rerun, { id: args.jobId });
    } catch (e) {
      console.log("Could not reset job state:", e);
      throw new Error("Failed to reset job state for rerun");
    }

    // Add log entry
    const includesEval = rerunPhases.includes("evaluation");
    try {
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "AML",
        message: includesEval
          ? "Starting generation + evaluation rerun..."
          : "Starting generation-only rerun...",
      });
    } catch (e) {
      // Ignore log failures
    }

    // Use the pipeline endpoint with generation (and optionally evaluation) phases
    const pythonApiUrl = "http://cli:8000";

    try {
      const response = await fetch(`${pythonApiUrl}/api/run-pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: job.config,
          phases: rerunPhases, // ["generation"] or ["generation", "evaluation"]
          apiKey: settings.openrouterApiKey,
          tavilyApiKey: settings.tavilyApiKey || null,
          jobsDirectory: jobsDirectory,
          convexUrl: "http://convex:3210",
          convexToken: convexAdminKey,
          evaluationConfig: job.evaluationConfig || null,
          datasetFile: job.datasetFile || null,
          reusedFromJobDir: null, // Already have composition data
          referenceDataset: job.referenceDataset || null,
          method: job.method || "cera",
          heuristicConfig: job.heuristicConfig || null,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Generation rerun failed: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      return { success: true, jobId: args.jobId, phases: rerunPhases };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      try {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Generation rerun failed: ${errorMessage}`,
        });
      } catch (e) {
        // Ignore failures in error handling
      }

      return { success: false, error: errorMessage };
    }
  },
});
