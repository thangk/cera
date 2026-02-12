// Composition action - runs in V8 isolate (no "use node")
// This avoids the "Invalid URL" bug with ctx.runQuery in self-hosted Node.js actions

import { action } from "./_generated/server";
import { v } from "convex/values";
import { api } from "./_generated/api";

/**
 * Run the composition phase of a job.
 *
 * Architecture:
 * - This action runs in V8 isolate (not Node.js)
 * - It handles all database mutations via ctx.runMutation
 * - It calls the Python API for file operations via fetch
 */
export const runComposition = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
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

    // Mark job as composing
    const jobsDirectory = settings.jobsDirectory || "./jobs";
    const tempJobDir = `${jobsDirectory}/${args.jobId}-${sanitizeJobName(job.name)}`;

    try {
      await ctx.runMutation(api.jobs.startComposing, {
        jobId: args.jobId,
        jobDir: tempJobDir,
      });
    } catch (e) {
      // Job might already be composing - ignore
    }

    // Add initial log
    try {
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "SIL",
        message: "Starting composition phase...",
      });
    } catch (e) {
      // Ignore log failures
    }

    // Get admin key for progress updates
    const convexAdminKey = settings.convexAdminKey || "";
    if (!convexAdminKey) {
      console.log("WARNING: Convex Admin Key not configured - progress updates will not work");
    }

    // Python API URL - from action's perspective (running in Convex container)
    // Both Convex and CLI containers are on the same Docker network
    const pythonApiUrl = "http://cli:8000";
    const fullUrl = `${pythonApiUrl}/api/compose-job-simple`;

    // Only pass Tavily API key if Tavily is explicitly enabled
    const tavilyApiKey = (settings.tavilyEnabled !== false && settings.tavilyApiKey) ? settings.tavilyApiKey : null;

    try {
      // Call Python FastAPI to run composition
      const response = await fetch(fullUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: job.config,
          apiKey: settings.openrouterApiKey,
          tavilyApiKey: tavilyApiKey,
          jobsDirectory: jobsDirectory,
          // Pass Convex credentials so Python can update progress
          convexUrl: "http://convex:3210",
          convexToken: convexAdminKey,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Composition failed: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      const result = await response.json();

      // Log what we received from Python for debugging
      const hasSubjectContext = !!result.subjectContext;
      const hasReviewerContext = !!result.reviewerContext;
      const hasAttributesContext = !!result.attributesContext;

      // Add completion log with context info
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "Composition",
        message: `Composition complete. Contexts received: subject=${hasSubjectContext}, reviewer=${hasReviewerContext}, attributes=${hasAttributesContext}`,
      });

      // Mark composition as complete and save the generated contexts
      await ctx.runMutation(api.jobs.completeComposition, {
        jobId: args.jobId,
        subjectContext: result.subjectContext,
        reviewerContext: result.reviewerContext,
        attributesContext: result.attributesContext,
      });

      return {
        success: true,
        jobId: args.jobId,
        jobDir: result.jobDir,
        status: "composed",
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";

      try {
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Composition failed: ${errorMessage}`,
        });
      } catch (e) {
        // Ignore failures in error handling
      }

      return { success: false, error: errorMessage };
    }
  },
});

/**
 * Create only reviewer and attributes contexts (RGM + ACM).
 * This is instant (no LLM) and runs during job creation.
 * SIL/MAV is deferred to pipeline execution.
 */
export const runContextsOnly = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    const settings = await ctx.runQuery(api.settings.get, {});
    const jobsDirectory = settings.jobsDirectory || "./jobs";

    const pythonApiUrl = "http://cli:8000";

    try {
      const response = await fetch(`${pythonApiUrl}/api/create-contexts-only`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: job.config,
          jobsDirectory,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Failed to create contexts: ${errorText}`,
        });
        return { success: false, error: errorText };
      }

      const result = await response.json();

      // Store the contexts and jobDir
      await ctx.runMutation(api.jobs.setContexts, {
        jobId: args.jobId,
        jobDir: result.jobDir,
        reviewerContext: result.reviewerContext,
        attributesContext: result.attributesContext,
      });

      return {
        success: true,
        jobId: args.jobId,
        jobDir: result.jobDir,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      await ctx.runMutation(api.jobs.setFailed, {
        jobId: args.jobId,
        error: `Failed to create contexts: ${errorMessage}`,
      });
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
