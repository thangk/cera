import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Simple ping query to check if Convex is reachable
export const ping = query({
  args: {},
  handler: async () => {
    return { ok: true, timestamp: Date.now() };
  },
});

// Query to get settings (there should only be one settings document)
export const get = query({
  args: {},
  handler: async (ctx) => {
    const settings = await ctx.db.query("settings").first();
    if (!settings) {
      // Return default settings if none exist
      return {
        openrouterApiKey: undefined,
        tavilyApiKey: undefined,
        tavilyEnabled: true, // Default to enabled if API key exists
        convexAdminKey: undefined,
        defaultProvider: "anthropic",
        defaultModel: "claude-sonnet-4-20250514",
        theme: "system" as const,
        jobsDirectory: "./jobs",
      };
    }
    return settings;
  },
});

// Mutation to update or create settings
export const update = mutation({
  args: {
    openrouterApiKey: v.optional(v.string()),
    tavilyApiKey: v.optional(v.string()),
    tavilyEnabled: v.optional(v.boolean()),
    convexAdminKey: v.optional(v.string()),
    defaultProvider: v.optional(v.string()),
    defaultModel: v.optional(v.string()),
    theme: v.optional(v.union(v.literal("light"), v.literal("dark"), v.literal("system"))),
    jobsDirectory: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("settings").first();

    if (existing) {
      // Update existing settings
      const updates: Record<string, unknown> = {};
      if (args.openrouterApiKey !== undefined) updates.openrouterApiKey = args.openrouterApiKey;
      if (args.tavilyApiKey !== undefined) updates.tavilyApiKey = args.tavilyApiKey;
      if (args.tavilyEnabled !== undefined) updates.tavilyEnabled = args.tavilyEnabled;
      if (args.convexAdminKey !== undefined) updates.convexAdminKey = args.convexAdminKey;
      if (args.defaultProvider !== undefined) updates.defaultProvider = args.defaultProvider;
      if (args.defaultModel !== undefined) updates.defaultModel = args.defaultModel;
      if (args.theme !== undefined) updates.theme = args.theme;
      if (args.jobsDirectory !== undefined) updates.jobsDirectory = args.jobsDirectory;

      await ctx.db.patch(existing._id, updates);
      return existing._id;
    } else {
      // Create new settings with defaults
      return await ctx.db.insert("settings", {
        openrouterApiKey: args.openrouterApiKey,
        tavilyApiKey: args.tavilyApiKey,
        tavilyEnabled: args.tavilyEnabled ?? true,
        convexAdminKey: args.convexAdminKey,
        defaultProvider: args.defaultProvider ?? "anthropic",
        defaultModel: args.defaultModel ?? "claude-sonnet-4-20250514",
        theme: args.theme ?? "system",
        jobsDirectory: args.jobsDirectory ?? "./jobs",
      });
    }
  },
});

// Mutation to update OpenRouter API key only
export const updateApiKey = mutation({
  args: { openrouterApiKey: v.string() },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("settings").first();

    if (existing) {
      await ctx.db.patch(existing._id, { openrouterApiKey: args.openrouterApiKey });
      return existing._id;
    } else {
      return await ctx.db.insert("settings", {
        openrouterApiKey: args.openrouterApiKey,
        defaultProvider: "anthropic",
        defaultModel: "claude-sonnet-4-20250514",
        theme: "system",
        jobsDirectory: "./jobs",
      });
    }
  },
});

// Mutation to clear OpenRouter API key
export const clearApiKey = mutation({
  args: {},
  handler: async (ctx) => {
    const existing = await ctx.db.query("settings").first();
    if (existing) {
      await ctx.db.patch(existing._id, { openrouterApiKey: undefined });
    }
  },
});

// Mutation to update Tavily API key
export const updateTavilyApiKey = mutation({
  args: { tavilyApiKey: v.string() },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("settings").first();

    if (existing) {
      await ctx.db.patch(existing._id, { tavilyApiKey: args.tavilyApiKey });
      return existing._id;
    } else {
      return await ctx.db.insert("settings", {
        tavilyApiKey: args.tavilyApiKey,
        defaultProvider: "anthropic",
        defaultModel: "claude-sonnet-4-20250514",
        theme: "system",
        jobsDirectory: "./jobs",
      });
    }
  },
});

// Mutation to clear Tavily API key
export const clearTavilyApiKey = mutation({
  args: {},
  handler: async (ctx) => {
    const existing = await ctx.db.query("settings").first();
    if (existing) {
      await ctx.db.patch(existing._id, { tavilyApiKey: undefined });
    }
  },
});

// Mutation to update Convex Admin key
export const updateConvexAdminKey = mutation({
  args: { convexAdminKey: v.string() },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("settings").first();

    if (existing) {
      await ctx.db.patch(existing._id, { convexAdminKey: args.convexAdminKey });
      return existing._id;
    } else {
      return await ctx.db.insert("settings", {
        convexAdminKey: args.convexAdminKey,
        defaultProvider: "anthropic",
        defaultModel: "claude-sonnet-4-20250514",
        theme: "system",
        jobsDirectory: "./jobs",
      });
    }
  },
});

// Mutation to clear Convex Admin key
export const clearConvexAdminKey = mutation({
  args: {},
  handler: async (ctx) => {
    const existing = await ctx.db.query("settings").first();
    if (existing) {
      await ctx.db.patch(existing._id, { convexAdminKey: undefined });
    }
  },
});

// Mutation to seed settings from environment variables (first load only)
// Only creates settings if none exist - does NOT overwrite existing keys
export const seedFromEnv = mutation({
  args: {
    openrouterApiKey: v.optional(v.string()),
    tavilyApiKey: v.optional(v.string()),
    convexAdminKey: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db.query("settings").first();

    if (existing) {
      // Settings exist - only fill in missing keys, don't overwrite
      const updates: Record<string, unknown> = {};
      if (!existing.openrouterApiKey && args.openrouterApiKey) {
        updates.openrouterApiKey = args.openrouterApiKey;
      }
      if (!existing.tavilyApiKey && args.tavilyApiKey) {
        updates.tavilyApiKey = args.tavilyApiKey;
      }
      if (!existing.convexAdminKey && args.convexAdminKey) {
        updates.convexAdminKey = args.convexAdminKey;
      }

      if (Object.keys(updates).length > 0) {
        await ctx.db.patch(existing._id, updates);
      }
      return { seeded: Object.keys(updates).length > 0, isNew: false };
    } else {
      // No settings exist - create with env values
      await ctx.db.insert("settings", {
        openrouterApiKey: args.openrouterApiKey,
        tavilyApiKey: args.tavilyApiKey,
        convexAdminKey: args.convexAdminKey,
        defaultProvider: "anthropic",
        defaultModel: "claude-sonnet-4-20250514",
        theme: "system",
        jobsDirectory: "./jobs",
      });
      return { seeded: true, isNew: true };
    }
  },
});
