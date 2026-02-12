import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// List all presets
export const list = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("llm_presets").collect();
  },
});

// Get the default preset (if any)
export const getDefault = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("llm_presets")
      .withIndex("by_default", (q) => q.eq("isDefault", true))
      .first();
  },
});

// Get a single preset by ID
export const get = query({
  args: { id: v.id("llm_presets") },
  handler: async (ctx, { id }) => {
    return await ctx.db.get(id);
  },
});

// Create a new preset
export const create = mutation({
  args: {
    name: v.string(),
    isDefault: v.boolean(),
    rdeModel: v.optional(v.string()),
    mavModels: v.optional(v.array(v.string())),
    savModel: v.optional(v.string()),
    genModel: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const now = Date.now();

    // If setting as default, clear any existing default
    if (args.isDefault) {
      const existingDefault = await ctx.db
        .query("llm_presets")
        .withIndex("by_default", (q) => q.eq("isDefault", true))
        .first();
      if (existingDefault) {
        await ctx.db.patch(existingDefault._id, { isDefault: false });
      }
    }

    return await ctx.db.insert("llm_presets", {
      name: args.name,
      isDefault: args.isDefault,
      rdeModel: args.rdeModel,
      mavModels: args.mavModels,
      savModel: args.savModel,
      genModel: args.genModel,
      createdAt: now,
      updatedAt: now,
    });
  },
});

// Update an existing preset
export const update = mutation({
  args: {
    id: v.id("llm_presets"),
    name: v.optional(v.string()),
    isDefault: v.optional(v.boolean()),
    rdeModel: v.optional(v.string()),
    mavModels: v.optional(v.array(v.string())),
    savModel: v.optional(v.string()),
    genModel: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { id, ...updates } = args;

    // If setting as default, clear any existing default (except this one)
    if (updates.isDefault === true) {
      const existingDefault = await ctx.db
        .query("llm_presets")
        .withIndex("by_default", (q) => q.eq("isDefault", true))
        .first();
      if (existingDefault && existingDefault._id !== id) {
        await ctx.db.patch(existingDefault._id, { isDefault: false });
      }
    }

    // Build update object with only defined fields
    const patchData: Record<string, unknown> = { updatedAt: Date.now() };
    if (updates.name !== undefined) patchData.name = updates.name;
    if (updates.isDefault !== undefined) patchData.isDefault = updates.isDefault;
    if (updates.rdeModel !== undefined) patchData.rdeModel = updates.rdeModel;
    if (updates.mavModels !== undefined) patchData.mavModels = updates.mavModels;
    if (updates.savModel !== undefined) patchData.savModel = updates.savModel;
    if (updates.genModel !== undefined) patchData.genModel = updates.genModel;

    await ctx.db.patch(id, patchData);
    return id;
  },
});

// Set a preset as default
export const setDefault = mutation({
  args: { id: v.id("llm_presets") },
  handler: async (ctx, { id }) => {
    // Clear existing default
    const existingDefault = await ctx.db
      .query("llm_presets")
      .withIndex("by_default", (q) => q.eq("isDefault", true))
      .first();
    if (existingDefault) {
      await ctx.db.patch(existingDefault._id, { isDefault: false });
    }

    // Set new default
    await ctx.db.patch(id, { isDefault: true, updatedAt: Date.now() });
    return id;
  },
});

// Remove default status from a preset
export const clearDefault = mutation({
  args: { id: v.id("llm_presets") },
  handler: async (ctx, { id }) => {
    await ctx.db.patch(id, { isDefault: false, updatedAt: Date.now() });
    return id;
  },
});

// Delete a preset
export const remove = mutation({
  args: { id: v.id("llm_presets") },
  handler: async (ctx, { id }) => {
    await ctx.db.delete(id);
  },
});
