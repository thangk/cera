import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Query to get logs for a job
export const getByJob = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("logs")
      .withIndex("by_job", (q) => q.eq("jobId", args.jobId))
      .order("asc")
      .collect();
  },
});

// Query to get recent logs for a job (last N entries)
export const getRecentByJob = query({
  args: { jobId: v.id("jobs"), limit: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const limit = args.limit ?? 100;
    const logs = await ctx.db
      .query("logs")
      .withIndex("by_job", (q) => q.eq("jobId", args.jobId))
      .order("desc")
      .take(limit);
    return logs.reverse(); // Return in chronological order
  },
});

// Mutation to add a log entry
export const add = mutation({
  args: {
    jobId: v.id("jobs"),
    level: v.union(v.literal("INFO"), v.literal("WARN"), v.literal("ERROR")),
    phase: v.string(),
    message: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("logs", {
      jobId: args.jobId,
      timestamp: Date.now(),
      level: args.level,
      phase: args.phase,
      message: args.message,
    });
  },
});

// Mutation to add multiple log entries at once
export const addBatch = mutation({
  args: {
    logs: v.array(
      v.object({
        jobId: v.id("jobs"),
        level: v.union(v.literal("INFO"), v.literal("WARN"), v.literal("ERROR")),
        phase: v.string(),
        message: v.string(),
      })
    ),
  },
  handler: async (ctx, args) => {
    const timestamp = Date.now();
    for (const log of args.logs) {
      await ctx.db.insert("logs", {
        ...log,
        timestamp,
      });
    }
  },
});

// Mutation to clear logs for a job
export const clearByJob = mutation({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    const logs = await ctx.db
      .query("logs")
      .withIndex("by_job", (q) => q.eq("jobId", args.jobId))
      .collect();
    for (const log of logs) {
      await ctx.db.delete(log._id);
    }
  },
});

// Query to get logs for a job with optional phase filtering
export const getByJobFiltered = query({
  args: {
    jobId: v.id("jobs"),
    phases: v.optional(v.array(v.string())),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const limit = args.limit ?? 500;
    let logs = await ctx.db
      .query("logs")
      .withIndex("by_job", (q) => q.eq("jobId", args.jobId))
      .order("asc")
      .collect();

    // Filter by phase if provided
    if (args.phases && args.phases.length > 0) {
      const phaseSet = new Set(args.phases);
      logs = logs.filter((log) => phaseSet.has(log.phase));
    }

    // Return last N entries
    return logs.slice(-limit);
  },
});
