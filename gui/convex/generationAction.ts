// Generation action - runs in V8 isolate
// Uses admin key from settings (stored in database) for Python to call back

import { action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

/**
 * Run the generation phase of a job.
 *
 * Architecture:
 * - This action runs in V8 isolate
 * - It triggers the Python API to run generation in the background
 * - Python handles progress updates via its own Convex client using the admin key from settings
 * - The job must already be in "composed" or "running" status with composition files
 */
export const runGeneration = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    // Verify job is in correct state
    if (job.status !== "composed" && job.status !== "running") {
      throw new Error(`Cannot start generation: job status is "${job.status}". Must be "composed" or "running".`);
    }

    if (!job.jobDir) {
      throw new Error("Job has no directory. Composition may not have completed properly.");
    }

    // Get settings (for API keys)
    const settings = await ctx.runQuery(api.settings.get, {});
    if (!settings.openrouterApiKey) {
      await ctx.runMutation(api.jobs.setFailed, {
        jobId: args.jobId,
        error: "OpenRouter API key not configured. Please set it in Settings.",
      });
      return { success: false, error: "OpenRouter API key not configured" };
    }

    // Warn if admin key is missing (progress updates won't work)
    const convexAdminKey = settings.convexAdminKey || "";
    if (!convexAdminKey) {
      console.log("WARNING: Convex Admin Key not configured - progress updates will not work");
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "WARN",
        phase: "AML",
        message: "Convex Admin Key not configured. Progress updates will not appear until job completes.",
      });
    }

    // Mark job as running (generation phase)
    try {
      await ctx.runMutation(api.jobs.startGeneration, { id: args.jobId });
    } catch (e) {
      // Job might already be running - continue anyway
      console.log("Could not start generation (may already be running):", e);
    }

    // Add initial log
    try {
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "AML",
        message: "Starting generation phase...",
      });
    } catch (e) {
      // Ignore log failures
    }

    // Python API URL - from action's perspective (running in Convex container)
    // Both Convex and CLI containers are on the same Docker network
    const pythonApiUrl = "http://cli:8000";
    const fullUrl = `${pythonApiUrl}/api/generate-job`;

    try {
      // Call Python FastAPI to run generation
      // This returns immediately - Python runs generation in the background
      const response = await fetch(fullUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobDir: job.jobDir,
          config: job.config,
          apiKey: settings.openrouterApiKey,
          // Pass Convex credentials so Python can update progress
          convexUrl: "http://convex:3210",
          convexToken: convexAdminKey,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Generation failed to start: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      const result = await response.json();

      // Log that generation has started
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "AML",
        message: `Generation started. Model: ${job.config.generation.provider}/${job.config.generation.model}`,
      });

      return {
        success: true,
        jobId: args.jobId,
        status: result.status,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      try {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Generation failed: ${errorMessage}`,
        });
      } catch (e) {
        // Ignore failures in error handling
      }

      return { success: false, error: errorMessage };
    }
  },
});
