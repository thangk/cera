import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

// Query to list all jobs
export const list = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("jobs").order("desc").collect();
  },
});

// Query to list jobs by status
export const listByStatus = query({
  args: { status: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("jobs")
      .withIndex("by_status", (q) =>
        q.eq("status", args.status as "pending" | "composing" | "composed" | "running" | "paused" | "completed" | "terminated" | "failed")
      )
      .collect();
  },
});

// Query to get a single job
export const get = query({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    return await ctx.db.get(args.id);
  },
});

// Query to list jobs available for reuse (any job with composition data)
// Returns jobs that have composition complete (composed, running, paused, completed)
export const listForReuse = query({
  args: {},
  handler: async (ctx) => {
    // Get all jobs and filter for those with composition data
    const allJobs = await ctx.db
      .query("jobs")
      .order("desc")
      .collect();

    // Jobs that have completed composition phase
    const reusableStatuses = ["composed", "running", "paused", "completed"];
    const reusableJobs = allJobs.filter((job) => reusableStatuses.includes(job.status));

    // Return simplified data for the dropdown
    return reusableJobs.map((job) => ({
      _id: job._id,
      name: job.name,
      subject: job.config.subject_profile?.query ?? "N/A",
      model: job.config.generation?.model ?? "N/A",
      status: job.status,
      jobDir: job.jobDir,
      createdAt: job.createdAt,
      completedAt: job.completedAt,
      referenceDataset: job.referenceDataset,
    }));
  },
});

// Query to list jobs that reused from a specific source job (job family)
export const listByReusedFrom = query({
  args: { sourceJobId: v.id("jobs") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("jobs")
      .withIndex("by_reused_from", (q) => q.eq("reusedFrom", args.sourceJobId))
      .collect();
  },
});

// Query to list completed CERA jobs for knowledge source selection in heuristic wizard
export const listCompletedCeraJobs = query({
  args: {},
  handler: async (ctx) => {
    const allJobs = await ctx.db.query("jobs").order("desc").collect();
    return allJobs
      .filter((j) => j.status === "completed" && j.method !== "heuristic")
      .map((j) => ({
        _id: j._id,
        name: j.name,
        method: j.method || "cera",
        createdAt: j.createdAt,
      }));
  },
});

// Query to list completed jobs with evaluation metrics for Research Tools
export const listForResearch = query({
  args: {},
  handler: async (ctx) => {
    const allJobs = await ctx.db.query("jobs").order("desc").collect();

    // Filter for completed jobs (metrics may be in Convex or on disk)
    const evaluated = allJobs.filter(
      (job) => job.status === "completed"
    );

    return evaluated.map((job) => {
      // Extract available target sizes from config
      const ceraTargets = job.config?.generation?.targets as Array<{ target_value: number; count_mode: string }> | undefined;
      const heuristicTargets = job.heuristicConfig?.targets as Array<{ targetValue: number; targetMode: string }> | undefined;
      const targets = ceraTargets?.map(t => ({ value: t.target_value, countMode: t.count_mode }))
        ?? heuristicTargets?.map(t => ({ value: t.targetValue, countMode: t.targetMode }))
        ?? null;

      // Extract generation model slug(s) from config
      const ceraModel = job.config?.generation?.model as string | undefined;
      const ceraModels = job.config?.generation?.models as string[] | undefined;
      const heuristicModel = job.heuristicConfig?.model as string | undefined;
      const modelSlugs = (ceraModels ?? (ceraModel ? [ceraModel] : (heuristicModel ? [heuristicModel] : [])))
        .map((m: string) => m.split("/").pop() || m);

      return {
        _id: job._id,
        name: job.name,
        method: job.method || "cera",
        generationCount:
          job.config?.generation?.count ??
          job.heuristicConfig?.targetValue ??
          0,
        totalRuns: job.totalRuns ?? 1,
        averageMetrics: job.averageMetrics,
        evaluationMetrics: job.evaluationMetrics,
        perRunMetrics: job.perRunMetrics,
        perModelMetrics: job.perModelMetrics,
        perTargetMetrics: job.perTargetMetrics ?? null,
        targets,
        jobDir: job.jobDir,
        createdAt: job.createdAt,
        modelSlugs,
      };
    });
  },
});

// Mutation to create a new job
export const create = mutation({
  args: {
    name: v.string(),
    config: v.object({
      subject_profile: v.optional(v.object({
        query: v.string(),
        additional_context: v.optional(v.string()),
        region: v.string(),
        category: v.optional(v.string()),
        domain: v.optional(v.string()),
        aspect_category_mode: v.optional(v.string()),
        aspect_categories: v.optional(v.array(v.string())),
        sentiment_depth: v.string(),
        mav: v.optional(v.object({
          enabled: v.boolean(),
          models: v.array(v.string()),
          similarity_threshold: v.optional(v.number()),
          max_queries: v.optional(v.number()),
        })),
      })),
      reviewer_profile: v.optional(v.object({
        age_range: v.union(v.array(v.number()), v.null()),
        sex_distribution: v.object({
          male: v.number(),
          female: v.number(),
          unspecified: v.number(),
        }),
        additional_context: v.optional(v.string()),
        persona_ratio: v.optional(v.number()),
      })),
      attributes_profile: v.optional(v.object({
        polarity: v.object({
          positive: v.number(),
          neutral: v.number(),
          negative: v.number(),
        }),
        noise: v.object({
          typo_rate: v.number(),
          colloquialism: v.boolean(),
          grammar_errors: v.boolean(),
          preset: v.optional(v.union(
            v.literal("none"),
            v.literal("light"),
            v.literal("moderate"),
            v.literal("heavy"),
            v.literal("ref_dataset")
          )),
          advanced: v.optional(v.boolean()),
          use_ocr: v.optional(v.boolean()),
          use_contextual: v.optional(v.boolean()),
        }),
        length_range: v.array(v.number()),
        edge_lengths: v.optional(v.object({
          min_length: v.number(),
          min_chance: v.number(),
          max_length: v.number(),
          max_chance: v.number(),
        })),
        temperature_range: v.optional(v.array(v.number())),
        cap_weights: v.optional(v.object({
          standard: v.number(),
          lowercase: v.number(),
          mixed: v.number(),
          emphasis: v.number(),
        })),
      })),
      generation: v.optional(v.object({
        count: v.number(),
        count_mode: v.optional(v.union(v.literal("reviews"), v.literal("sentences"))),
        target_sentences: v.optional(v.number()),
        batch_size: v.number(),
        request_size: v.number(),
        provider: v.string(),
        model: v.string(),
        output_formats: v.optional(v.array(v.string())),
        dataset_mode: v.optional(v.string()),
        total_runs: v.optional(v.number()),
        neb_enabled: v.optional(v.boolean()),
        neb_depth: v.optional(v.number()),
        models: v.optional(v.array(v.string())),
        parallel_models: v.optional(v.boolean()),
        // Multi-target dataset support
        target_prefix: v.optional(v.string()),
        targets: v.optional(v.array(v.object({
          count_mode: v.union(v.literal("reviews"), v.literal("sentences")),
          target_value: v.number(),
          batch_size: v.number(),
          request_size: v.number(),
          total_runs: v.number(),
          runs_mode: v.union(v.literal("parallel"), v.literal("sequential")),
          neb_depth: v.optional(v.number()),
        }))),
        parallel_targets: v.optional(v.boolean()),
      })),
      // Ablation settings for reproducibility
      ablation: v.optional(v.object({
        sil_enabled: v.boolean(),
        mav_enabled: v.boolean(),
        polarity_enabled: v.boolean(),
        noise_enabled: v.boolean(),
        age_enabled: v.boolean(),
        sex_enabled: v.boolean(),
      })),
    }),
    // Selected pipeline phases
    phases: v.optional(v.array(v.string())),
    // Evaluation configuration
    evaluationConfig: v.optional(v.object({
      metrics: v.array(v.string()),
      reference_metrics_enabled: v.optional(v.boolean()),
      reference_file: v.optional(v.string()),
      self_test: v.optional(v.object({
        enabled: v.boolean(),
        split_mode: v.string(),
      })),
    })),
    // Path to uploaded dataset (for EVALUATION-only jobs)
    datasetFile: v.optional(v.string()),
    // Optional: reuse composition from an existing job
    reusedFrom: v.optional(v.id("jobs")),
    // Reference dataset configuration (for context extraction and/or MDQA comparison)
    referenceDataset: v.optional(v.object({
      fileName: v.optional(v.string()),
      useForEvaluation: v.boolean(),
      extractedSubjectContext: v.optional(v.boolean()),
      extractedReviewerContext: v.optional(v.boolean()),
    })),
    // Pre-job RDE token usage records (from context extraction, forwarded to pipeline)
    rdeUsage: v.optional(v.any()),
    // Cost estimates from job creation (for comparison with actual)
    estimatedCost: v.optional(v.object({
      composition: v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.optional(v.number()),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      }),
      generation: v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.optional(v.number()),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      }),
      evaluation: v.object({
        cost: v.number(),
        calls: v.number(),
      }),
      rde: v.optional(v.object({
        cost: v.number(),
        calls: v.number(),
      })),
      total: v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.number(),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      }),
    })),
    // === HEURISTIC METHOD (RQ1 Baseline) ===
    method: v.optional(v.union(v.literal("cera"), v.literal("heuristic"), v.literal("real"))),
    heuristicConfig: v.optional(v.object({
      prompt: v.string(),
      useFormatPrompt: v.optional(v.boolean()),
      formatPrompt: v.optional(v.string()),
      targetMode: v.union(v.literal("reviews"), v.literal("sentences")),
      targetValue: v.number(),
      reviewsPerBatch: v.number(),
      requestSize: v.optional(v.number()),
      avgSentencesPerReview: v.string(),
      model: v.string(),
      outputFormat: v.string(),
      totalRuns: v.optional(v.number()),
      parallelRuns: v.optional(v.boolean()),
      knowledgeSourceJobId: v.optional(v.string()),
      // Multi-target dataset support
      targetPrefix: v.optional(v.string()),
      targets: v.optional(v.array(v.object({
        targetMode: v.union(v.literal("reviews"), v.literal("sentences")),
        targetValue: v.number(),
        reviewsPerBatch: v.number(),
        requestSize: v.number(),
        totalRuns: v.number(),
        runsMode: v.union(v.literal("parallel"), v.literal("sequential")),
      }))),
      parallelTargets: v.optional(v.boolean()),
      models: v.optional(v.array(v.string())),
      parallelModels: v.optional(v.boolean()),
    })),
  },
  handler: async (ctx, args) => {
    const jobId = await ctx.db.insert("jobs", {
      name: args.name,
      config: args.config,
      phases: args.phases || ["composition", "generation", "evaluation"],
      evaluationConfig: args.evaluationConfig,
      datasetFile: args.datasetFile,
      status: "pending",
      progress: 0,
      createdAt: Date.now(),
      reusedFrom: args.reusedFrom,
      referenceDataset: args.referenceDataset,
      estimatedCost: args.estimatedCost,
      // Heuristic method fields (optional, only set for heuristic jobs)
      method: args.method || "cera",
      heuristicConfig: args.heuristicConfig,
      // Pre-job RDE token usage (from context extraction)
      rdeUsage: args.rdeUsage,
    });
    return jobId;
  },
});

// Mutation to start composition phase (idempotent)
export const startComposing = mutation({
  args: {
    jobId: v.id("jobs"),
    jobDir: v.string(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    // Allow if pending or already composing (idempotent)
    if (job.status !== "pending" && job.status !== "composing") {
      throw new Error(`Cannot start composition on job with status: ${job.status}`);
    }
    await ctx.db.patch(args.jobId, {
      status: "composing",
      jobDir: args.jobDir,
      currentPhase: "SIL",
      progress: job.status === "composing" ? job.progress : 0, // Preserve progress if already composing
    });
  },
});

// Mutation to update composition progress
export const updateCompositionProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    progress: v.number(),
    currentPhase: v.string(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job || job.status !== "composing") {
      throw new Error("Can only update composition progress on composing jobs");
    }
    await ctx.db.patch(args.jobId, {
      progress: args.progress,
      currentPhase: args.currentPhase,
    });
  },
});

// Mutation to mark composition as complete and store generated contexts
export const completeComposition = mutation({
  args: {
    jobId: v.id("jobs"),
    subjectContext: v.optional(v.any()),
    reviewerContext: v.optional(v.any()),
    attributesContext: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job || job.status !== "composing") {
      throw new Error("Can only complete composition on composing jobs");
    }
    await ctx.db.patch(args.jobId, {
      status: "composed",
      progress: 100,
      currentPhase: "Composition complete",
      subjectContext: args.subjectContext,
      reviewerContext: args.reviewerContext,
      attributesContext: args.attributesContext,
    });
  },
});

// Mutation to update job progress
export const updateProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    progress: v.number(),
    currentPhase: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      progress: args.progress,
      currentPhase: args.currentPhase,
      status: "running",
    });
  },
});

// Mutation to mark job as completed
export const complete = mutation({
  args: {
    jobId: v.id("jobs"),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      status: "completed",
      progress: 100,
      completedAt: Date.now(),
    });
  },
});

// Mutation to mark job as failed
export const setFailed = mutation({
  args: {
    jobId: v.id("jobs"),
    error: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      status: "failed",
      error: args.error,
    });
  },
});

// Mutation to delete a job
export const remove = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.id);
  },
});

// Mutation to pause a job
export const pause = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job || job.status !== "running") {
      throw new Error("Can only pause running jobs");
    }
    await ctx.db.patch(args.id, {
      status: "paused",
    });
  },
});

// Mutation to resume a paused job
export const resume = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job || job.status !== "paused") {
      throw new Error("Can only resume paused jobs");
    }
    await ctx.db.patch(args.id, {
      status: "running",
    });
  },
});

// Mutation to terminate a job (supports all active phases)
export const terminate = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    const terminableStatuses = ["composing", "running", "paused", "evaluating"];
    if (!job || !terminableStatuses.includes(job.status)) {
      throw new Error(`Can only terminate jobs in active phases (composing, running, paused, evaluating). Current: ${job?.status}`);
    }
    await ctx.db.patch(args.id, {
      status: "terminated",
      completedAt: Date.now(),
    });
  },
});

// Mutation to start generation phase (from composed job)
export const startGeneration = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job || job.status !== "composed") {
      throw new Error("Can only start generation on composed jobs");
    }
    await ctx.db.patch(args.id, {
      status: "running",
      progress: 0,
      generatedCount: 0, // Reset generated count when starting generation
      generatedSentences: 0, // Reset generated sentences when starting generation
      failedCount: 0, // Reset failed count when starting generation
      currentPhase: "AML",
    });
  },
});

// Mutation to update the generated review count during generation
export const updateGeneratedCount = mutation({
  args: {
    jobId: v.id("jobs"),
    count: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    // Allow updating count for running/completed/failed jobs
    await ctx.db.patch(args.jobId, {
      generatedCount: args.count,
    });
  },
});

// Mutation to update the generated sentence count (for sentence mode)
export const updateGeneratedSentences = mutation({
  args: {
    jobId: v.id("jobs"),
    count: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.jobId, {
      generatedSentences: args.count,
    });
  },
});

// Mutation to update the failed count during generation
export const updateFailedCount = mutation({
  args: {
    jobId: v.id("jobs"),
    count: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.jobId, {
      failedCount: args.count,
    });
  },
});

// Mutation to rerun a job (reset to composed to re-run generation only)
export const rerun = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job || (job.status !== "terminated" && job.status !== "failed" && job.status !== "completed")) {
      throw new Error("Can only rerun terminated, failed, or completed jobs");
    }
    // Reset to composed (keeps composition data, re-runs generation)
    await ctx.db.patch(args.id, {
      status: "composed",
      progress: 100,
      generatedCount: 0, // Reset generated count for rerun
      generatedSentences: 0, // Reset generated sentences for rerun
      failedCount: 0, // Reset failed count for rerun
      currentPhase: "Composition complete",
      error: undefined,
      completedAt: undefined,
    });
  },
});

// Mutation to fully reset a job (reset to pending, re-run composition + generation)
export const fullReset = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.id, {
      status: "pending",
      progress: 0,
      generatedCount: 0, // Reset generated count for full reset
      generatedSentences: 0, // Reset generated sentences for full reset
      failedCount: 0, // Reset failed count for full reset
      currentPhase: undefined,
      error: undefined,
      completedAt: undefined,
      jobDir: undefined, // Clear job dir so new one is created
      // Clear composition contexts so they're regenerated fresh
      subjectContext: undefined,
      reviewerContext: undefined,
      attributesContext: undefined,
    });
  },
});

// Mutation to start evaluation phase
export const startEvaluation = mutation({
  args: { id: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.id);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.id, {
      status: "evaluating",
      currentPhase: "MDQA",
      // Reset progress to evaluation start point (80% in overall pipeline)
      progress: 80,
      // Clear previous evaluation metrics so progress shows correctly on rerun
      evaluationMetrics: undefined,
    });
  },
});

// Mutation to store job directory path (used when creating contexts only)
export const setJobDir = mutation({
  args: {
    jobId: v.id("jobs"),
    jobDir: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      jobDir: args.jobDir,
    });
  },
});

// Mutation to store conformity report after generation
export const setConformityReport = mutation({
  args: {
    jobId: v.id("jobs"),
    conformityReport: v.object({
      polarity: v.number(),
      length: v.number(),
      noise: v.number(),
      validation: v.optional(v.number()),
      temperature: v.optional(v.number()),
    }),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      conformityReport: args.conformityReport,
    });
  },
});

// Mutation to store evaluation metrics after MDQA
export const setEvaluationMetrics = mutation({
  args: {
    jobId: v.id("jobs"),
    evaluationMetrics: v.object({
      bleu: v.optional(v.number()),
      rouge_l: v.optional(v.number()),
      bertscore: v.optional(v.number()),
      moverscore: v.optional(v.number()),
      distinct_1: v.optional(v.number()),
      distinct_2: v.optional(v.number()),
      self_bleu: v.optional(v.number()),
      // Standard deviation fields for multi-run evaluations
      bleu_std: v.optional(v.number()),
      rouge_l_std: v.optional(v.number()),
      bertscore_std: v.optional(v.number()),
      moverscore_std: v.optional(v.number()),
      distinct_1_std: v.optional(v.number()),
      distinct_2_std: v.optional(v.number()),
      self_bleu_std: v.optional(v.number()),
    }),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      evaluationMetrics: args.evaluationMetrics,
    });
  },
});

// Mutation to store evaluation compute device (GPU/CPU)
export const setEvaluationDevice = mutation({
  args: {
    jobId: v.id("jobs"),
    device: v.object({
      type: v.string(),
      name: v.optional(v.string()),
    }),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      evaluationDevice: args.device,
    });
  },
});

// Mutation to store reviewer and attributes contexts (from create-contexts-only)
export const setContexts = mutation({
  args: {
    jobId: v.id("jobs"),
    jobDir: v.string(),
    reviewerContext: v.optional(v.any()),
    attributesContext: v.optional(v.any()),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      jobDir: args.jobDir,
      reviewerContext: args.reviewerContext,
      attributesContext: args.attributesContext,
    });
  },
});

// Mutation to store config.json path after pipeline saves it
export const setConfigPath = mutation({
  args: {
    jobId: v.id("jobs"),
    configPath: v.string(),
  },
  handler: async (ctx, args) => {
    await ctx.db.patch(args.jobId, {
      configPath: args.configPath,
    });
  },
});

// Query to get source job's data for composition reuse
export const getSourceJobConfig = query({
  args: { jobId: v.id("jobs") },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) return null;

    return {
      _id: job._id,
      name: job.name,
      config: job.config,
      jobDir: job.jobDir,
      configPath: job.configPath,
      subjectContext: job.subjectContext,
      reviewerContext: job.reviewerContext,
      attributesContext: job.attributesContext,
    };
  },
});

// === HEURISTIC METHOD MUTATIONS ===

// Mutation to start heuristic generation (directly from pending/failed/terminated, skips composition)
export const startHeuristicGeneration = mutation({
  args: {
    jobId: v.id("jobs"),
    jobDir: v.string(),
    totalBatches: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    if (job.method !== "heuristic") {
      throw new Error("This mutation is only for heuristic jobs");
    }
    // Allow starting from pending, failed, terminated, or completed (for rerun)
    const allowedStatuses = ["pending", "failed", "terminated", "completed"];
    if (!allowedStatuses.includes(job.status)) {
      throw new Error(`Cannot start heuristic generation on job with status: ${job.status}`);
    }
    await ctx.db.patch(args.jobId, {
      status: "running",
      jobDir: args.jobDir,
      currentPhase: "Heuristic Generation",
      progress: 0,
      error: undefined, // Clear any previous error
      heuristicProgress: {
        currentBatch: 0,
        totalBatches: args.totalBatches,
        reviewsCollected: 0,
      },
    });
  },
});

// Mutation to update heuristic generation progress
export const updateHeuristicProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    currentBatch: v.number(),
    totalBatches: v.number(),
    reviewsCollected: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    // Calculate progress percentage based on batches
    const progress = Math.min(Math.round((args.currentBatch / args.totalBatches) * 100), 100);
    await ctx.db.patch(args.jobId, {
      progress,
      heuristicProgress: {
        currentBatch: args.currentBatch,
        totalBatches: args.totalBatches,
        reviewsCollected: args.reviewsCollected,
      },
    });
  },
});

// Mutation to complete heuristic generation and start evaluation
export const completeHeuristicGeneration = mutation({
  args: {
    jobId: v.id("jobs"),
    reviewsCollected: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job || !["running", "evaluating"].includes(job.status as string)) {
      throw new Error("Can only complete heuristic generation on running/evaluating jobs");
    }
    await ctx.db.patch(args.jobId, {
      generatedCount: args.reviewsCollected,
    });
  },
});

// Per-run progress tracking for parallel heuristic runs (upsert by run number)
export const updateRunProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    run: v.number(),
    status: v.string(),
    currentBatch: v.number(),
    totalBatches: v.number(),
    reviewsCollected: v.number(),
    evalProgress: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) throw new Error("Job not found");

    const existing = job.runProgress || [];
    const entry = {
      run: args.run,
      status: args.status,
      currentBatch: args.currentBatch,
      totalBatches: args.totalBatches,
      reviewsCollected: args.reviewsCollected,
      evalProgress: args.evalProgress,
    };

    // Upsert: find by run number, update or append
    const idx = existing.findIndex((e: { run: number }) => e.run === args.run);
    if (idx >= 0) {
      existing[idx] = entry;
    } else {
      existing.push(entry);
    }

    // Compute overall job progress as average of all runs
    const totalRuns = existing.length;
    const avgProgress = Math.round(
      existing.reduce((sum: number, e: { status: string; currentBatch: number; totalBatches: number; evalProgress?: number }) => {
        if (e.status === "completed") return sum + 100;
        if (e.status === "evaluating") return sum + 50 + (e.evalProgress || 0) / 2;
        // Generating phase = 0-50% of run's contribution
        const genPct = e.totalBatches > 0 ? (e.currentBatch / e.totalBatches) * 100 : 0;
        return sum + genPct / 2;
      }, 0) / totalRuns
    );

    await ctx.db.patch(args.jobId, {
      runProgress: existing,
      progress: avgProgress,
    });
  },
});

// === MULTI-RUN TRACKING MUTATIONS ===

// Mutation to update current run progress (for total_runs > 1)
export const updateCurrentRun = mutation({
  args: {
    jobId: v.id("jobs"),
    currentRun: v.number(),
    totalRuns: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.jobId, {
      currentRun: args.currentRun,
      totalRuns: args.totalRuns,
    });
  },
});

// Mutation to update actual cost from Python API after job completion
export const updateActualCost = mutation({
  args: {
    jobId: v.id("jobs"),
    actualCost: v.object({
      composition: v.optional(v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.number(),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      })),
      generation: v.optional(v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.number(),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      })),
      total: v.optional(v.object({
        cost: v.number(),
        calls: v.number(),
        tokens: v.number(),
        promptTokens: v.optional(v.number()),
        completionTokens: v.optional(v.number()),
      })),
    }),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.jobId, {
      actualCost: args.actualCost,
    });
  },
});

// Mutation to save per-run evaluation metrics (for total_runs > 1)
export const savePerRunMetrics = mutation({
  args: {
    jobId: v.id("jobs"),
    run: v.number(),
    datasetFile: v.string(),
    metrics: v.object({
      bleu: v.optional(v.number()),
      rouge_l: v.optional(v.number()),
      bertscore: v.optional(v.number()),
      moverscore: v.optional(v.number()),
      distinct_1: v.optional(v.number()),
      distinct_2: v.optional(v.number()),
      self_bleu: v.optional(v.number()),
    }),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    // Append to existing perRunMetrics array
    const existing = job.perRunMetrics || [];
    const updated = [...existing, {
      run: args.run,
      datasetFile: args.datasetFile,
      metrics: args.metrics,
    }];
    await ctx.db.patch(args.jobId, {
      perRunMetrics: updated,
    });
  },
});

// Mutation to save average metrics across all runs (for total_runs > 1)
export const saveAverageMetrics = mutation({
  args: {
    jobId: v.id("jobs"),
    averageMetrics: v.object({
      bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      rouge_l: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      bertscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      moverscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_1: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_2: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      self_bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
    }),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) {
      throw new Error("Job not found");
    }
    await ctx.db.patch(args.jobId, {
      averageMetrics: args.averageMetrics,
    });
  },
});

// Upsert per-model progress (multi-model jobs)
export const updateModelProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    model: v.string(),
    modelSlug: v.string(),
    generated: v.number(),
    failed: v.number(),
    target: v.number(),
    progress: v.number(),
    status: v.string(),
    evalProgress: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) throw new Error("Job not found");

    const existing = job.modelProgress || [];
    const entry = {
      model: args.model,
      modelSlug: args.modelSlug,
      generated: args.generated,
      failed: args.failed,
      target: args.target,
      progress: args.progress,
      status: args.status,
      evalProgress: args.evalProgress,
    };

    const idx = existing.findIndex((e: { model: string }) => e.model === args.model);
    if (idx >= 0) {
      existing[idx] = entry;
    } else {
      existing.push(entry);
    }

    // Compute overall job progress as average of all models
    const totalModels = existing.length;
    const avgProgress = Math.round(
      existing.reduce((sum: number, e: { progress: number; evalProgress?: number; status: string }) => {
        // Each model contributes: gen progress (0-50%) + eval progress (50-100%)
        if (e.status === "completed") return sum + 100;
        if (e.status === "evaluating") return sum + 50 + (e.evalProgress || 0) / 2;
        return sum + e.progress / 2; // generating phase = 0-50% of model's contribution
      }, 0) / totalModels
    );

    await ctx.db.patch(args.jobId, {
      modelProgress: existing,
      progress: avgProgress,
    });
  },
});

// Save per-model evaluation metrics (multi-model jobs)
export const savePerModelMetrics = mutation({
  args: {
    jobId: v.id("jobs"),
    model: v.string(),
    modelSlug: v.string(),
    metrics: v.object({
      bleu: v.optional(v.number()),
      rouge_l: v.optional(v.number()),
      bertscore: v.optional(v.number()),
      moverscore: v.optional(v.number()),
      distinct_1: v.optional(v.number()),
      distinct_2: v.optional(v.number()),
      self_bleu: v.optional(v.number()),
    }),
    conformity: v.optional(v.object({
      polarity: v.number(),
      length: v.number(),
      noise: v.number(),
      validation: v.optional(v.number()),
      temperature: v.optional(v.number()),
    })),
    perRunMetrics: v.optional(v.array(v.object({
      run: v.number(),
      datasetFile: v.string(),
      metrics: v.object({
        bleu: v.optional(v.number()),
        rouge_l: v.optional(v.number()),
        bertscore: v.optional(v.number()),
        moverscore: v.optional(v.number()),
        distinct_1: v.optional(v.number()),
        distinct_2: v.optional(v.number()),
        self_bleu: v.optional(v.number()),
      }),
    }))),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) throw new Error("Job not found");

    const existing = job.perModelMetrics || [];
    const entry = {
      model: args.model,
      modelSlug: args.modelSlug,
      metrics: args.metrics,
      conformity: args.conformity,
      perRunMetrics: args.perRunMetrics,
    };

    const idx = existing.findIndex((e: { model: string }) => e.model === args.model);
    if (idx >= 0) {
      existing[idx] = entry;
    } else {
      existing.push(entry);
    }

    await ctx.db.patch(args.jobId, {
      perModelMetrics: existing,
    });
  },
});

// Update target-level progress (multi-target jobs)
export const updateTargetProgress = mutation({
  args: {
    jobId: v.id("jobs"),
    targetIndex: v.number(),
    targetLabel: v.string(),
    status: v.string(),
    progress: v.number(),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) throw new Error("Job not found");

    const existing = (job.targetProgress as Array<{
      targetIndex: number;
      targetLabel: string;
      status: string;
      progress: number;
    }>) || [];
    const entry = {
      targetIndex: args.targetIndex,
      targetLabel: args.targetLabel,
      status: args.status,
      progress: args.progress,
    };

    const idx = existing.findIndex((e) => e.targetIndex === args.targetIndex);
    if (idx >= 0) {
      existing[idx] = entry;
    } else {
      existing.push(entry);
    }

    await ctx.db.patch(args.jobId, {
      targetProgress: existing,
    });
  },
});

// Save per-target evaluation metrics (multi-target jobs)
export const saveTargetMetrics = mutation({
  args: {
    jobId: v.id("jobs"),
    targetIndex: v.number(),
    targetLabel: v.string(),
    targetValue: v.number(),
    countMode: v.string(),
    metrics: v.optional(v.object({
      bleu: v.optional(v.number()),
      rouge_l: v.optional(v.number()),
      bertscore: v.optional(v.number()),
      moverscore: v.optional(v.number()),
      distinct_1: v.optional(v.number()),
      distinct_2: v.optional(v.number()),
      self_bleu: v.optional(v.number()),
    })),
    perModelMetrics: v.optional(v.array(v.object({
      model: v.string(),
      modelSlug: v.string(),
      metrics: v.optional(v.object({
        bleu: v.optional(v.number()),
        rouge_l: v.optional(v.number()),
        bertscore: v.optional(v.number()),
        moverscore: v.optional(v.number()),
        distinct_1: v.optional(v.number()),
        distinct_2: v.optional(v.number()),
        self_bleu: v.optional(v.number()),
      })),
    }))),
    conformity: v.optional(v.object({
      polarity: v.number(),
      length: v.number(),
      noise: v.number(),
      validation: v.optional(v.number()),
    })),
    perRunMetrics: v.optional(v.array(v.object({
      run: v.number(),
      datasetFile: v.string(),
      metrics: v.object({
        bleu: v.optional(v.number()),
        rouge_l: v.optional(v.number()),
        bertscore: v.optional(v.number()),
        moverscore: v.optional(v.number()),
        distinct_1: v.optional(v.number()),
        distinct_2: v.optional(v.number()),
        self_bleu: v.optional(v.number()),
      }),
    }))),
    averageMetrics: v.optional(v.object({
      bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      rouge_l: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      bertscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      moverscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_1: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_2: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      self_bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
    })),
  },
  handler: async (ctx, args) => {
    const job = await ctx.db.get(args.jobId);
    if (!job) throw new Error("Job not found");

    const existing = (job.perTargetMetrics as any[]) || [];
    // Build entry from provided args only (undefined fields left out)
    const entry: Record<string, unknown> = {
      targetIndex: args.targetIndex,
      targetLabel: args.targetLabel,
      targetValue: args.targetValue,
      countMode: args.countMode,
    };
    if (args.metrics !== undefined) entry.metrics = args.metrics;
    if (args.perModelMetrics !== undefined) entry.perModelMetrics = args.perModelMetrics;
    if (args.conformity !== undefined) entry.conformity = args.conformity;
    if (args.perRunMetrics !== undefined) entry.perRunMetrics = args.perRunMetrics;
    if (args.averageMetrics !== undefined) entry.averageMetrics = args.averageMetrics;

    const idx = existing.findIndex((e) => e.targetIndex === args.targetIndex);
    if (idx >= 0) {
      // Merge: preserve existing fields, override with new ones
      existing[idx] = { ...existing[idx], ...entry };
    } else {
      existing.push(entry);
    }

    await ctx.db.patch(args.jobId, {
      perTargetMetrics: existing,
    });
  },
});
