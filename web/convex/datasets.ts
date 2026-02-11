import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Query to list all datasets
export const list = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("datasets").order("desc").collect();
  },
});

// Query to get a single dataset
export const get = query({
  args: { id: v.id("datasets") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});

// Query to get dataset by job ID
export const getByJob = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("datasets")
      .withIndex("by_job", (q) => q.eq("jobId", args.jobId))
      .first();
  },
});

// Mutation to create a dataset
export const create = mutation({
  args: {
    jobId: v.id("jobs"),
    name: v.string(),
    subject: v.string(),
    category: v.string(),
    reviewCount: v.number(),
    metrics: v.object({
      bertscore: v.number(),
      distinct_1: v.number(),
      distinct_2: v.number(),
      self_bleu: v.number(),
    }),
    outputPath: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("datasets", {
      ...args,
      createdAt: Date.now(),
    });
  },
});

// Mutation to update dataset metrics
export const updateMetrics = mutation({
  args: {
    id: v.id("datasets"),
    metrics: v.object({
      bertscore: v.number(),
      distinct_1: v.number(),
      distinct_2: v.number(),
      self_bleu: v.number(),
    }),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.id, { metrics: args.metrics });
  },
});

// Mutation to delete a dataset
export const remove = mutation({
  args: { id: v.id("datasets") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.id);
  },
});
