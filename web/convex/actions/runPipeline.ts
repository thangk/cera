"use node";

import { action } from "../_generated/server";
import { v } from "convex/values";
import { api } from "../_generated/api";

export const run = action({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    // Get job details
    const job = await ctx.runQuery(api.jobs.get, { id: args.jobId });
    if (!job) {
      throw new Error("Job not found");
    }

    // Get settings (including API key)
    const settings = await ctx.runQuery(api.settings.get, {});
    if (!settings.openrouterApiKey) {
      await ctx.runMutation(api.jobs.setFailed, {
        jobId: args.jobId,
        error: "OpenRouter API key not configured. Please set it in Settings.",
      });
      return;
    }

    // In Docker on Windows/Mac, use host.docker.internal to reach host ports
    const pythonApiUrl = "http://host.docker.internal:8000";

    try {
      // Call Python FastAPI to start the pipeline
      const response = await fetch(`${pythonApiUrl}/api/run-job`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jobId: args.jobId,
          jobName: job.name,
          config: job.config,
          apiKey: settings.openrouterApiKey,
          jobsDirectory: settings.jobsDirectory || "./jobs",
          // CLI container uses Docker network to reach Convex
          convexUrl: "http://convex:3210",
          convexToken: process.env.CONVEX_DEPLOY_KEY || "",
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        await ctx.runMutation(api.jobs.setFailed, {
          jobId: args.jobId,
          error: `Python API error: ${errorText}`,
        });
        return;
      }

      // Log that the job has started
      await ctx.runMutation(api.logs.add, {
        jobId: args.jobId,
        level: "INFO",
        phase: "init",
        message: "Pipeline job started successfully",
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      await ctx.runMutation(api.jobs.setFailed, {
        jobId: args.jobId,
        error: `Failed to connect to Python API: ${errorMessage}`,
      });
    }
  },
});
