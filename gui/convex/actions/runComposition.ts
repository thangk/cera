"use node";

/**
 * @deprecated This action is deprecated due to "Invalid URL" bug with ctx.runQuery
 * in self-hosted Node.js actions. Use compositionAction.runComposition instead.
 *
 * See: https://github.com/get-convex/convex-backend/issues (self-hosted action issues)
 */

import { action } from "../_generated/server";
import { v } from "convex/values";

export const run = action({
  args: { jobId: v.id("jobs") },
  handler: async () => {
    throw new Error(
      "This action is deprecated. Use compositionAction.runComposition instead. " +
      "The 'use node' actions have issues with ctx.runQuery in self-hosted Convex."
    );
  },
});
