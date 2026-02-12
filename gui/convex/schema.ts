import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  // Generation jobs
  jobs: defineTable({
    name: v.string(),
    config: v.object({
      subject_profile: v.optional(v.object({
        query: v.string(),
        additional_context: v.optional(v.string()),
        region: v.string(),
        category: v.optional(v.string()), // Legacy field
        domain: v.optional(v.string()),
        aspect_category_mode: v.optional(v.string()),
        aspect_categories: v.optional(v.array(v.string())),
        sentiment_depth: v.string(),
        context_scope: v.optional(v.string()), // Deprecated - kept optional for existing data
        mav: v.optional(v.object({
          enabled: v.boolean(),
          models: v.array(v.string()), // 2+ models required for consensus
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
        persona_ratio: v.optional(v.number()), // 0.0-1.0, default 0.9 — percentage of unique personas relative to review count
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
        edge_lengths: v.optional(v.object({ // Edge-case sentence lengths with per-review generation chance
          min_length: v.number(),    // e.g., 1 sentence
          min_chance: v.number(),    // e.g., 0.10 = 10% chance per review
          max_length: v.number(),    // e.g., 10 sentences
          max_chance: v.number(),    // e.g., 0.05 = 5% chance per review
        })),
        temperature_range: v.optional(v.array(v.number())), // LLM generation temperature range [min, max]
        cap_weights: v.optional(v.object({ // Capitalization style distribution weights (decimals, e.g., 0.55 = 55%)
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
        total_runs: v.optional(v.number()), // Number of times to run generation (default: 1)
        neb_enabled: v.optional(v.boolean()), // NEB (Negative Example Buffer) for diversity
        neb_depth: v.optional(v.number()), // How many batches to remember (1-10)
        models: v.optional(v.array(v.string())), // Multiple generation models (multi-model comparison)
        parallel_models: v.optional(v.boolean()), // Run models concurrently
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
    // Selected pipeline phases for this job
    phases: v.optional(v.array(v.string())), // ["composition", "generation", "evaluation"]
    // Evaluation configuration
    evaluationConfig: v.optional(v.object({
      metrics: v.array(v.string()), // ["bertscore", "bleu", "rouge_l", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
      reference_metrics_enabled: v.optional(v.boolean()), // Whether Lexical/Semantic metrics use reference
      reference_file: v.optional(v.string()), // Reference file name for MDQA comparison
      ceiling_test: v.optional(v.object({
        enabled: v.boolean(),
        split_mode: v.string(), // "random" | "sequential"
      })),
    })),
    // Reference dataset configuration (for context extraction and/or MDQA comparison)
    referenceDataset: v.optional(v.object({
      fileName: v.optional(v.string()),           // e.g., "restaurant-reviews.jsonl"
      useForEvaluation: v.boolean(),              // Use for MDQA comparison (Dreal vs Dgen)
      extractedSubjectContext: v.optional(v.boolean()),  // Was subject context extracted?
      extractedReviewerContext: v.optional(v.boolean()), // Was reviewer context extracted?
    })),
    // Path to uploaded dataset (for EVALUATION-only jobs)
    datasetFile: v.optional(v.string()),
    status: v.union(
      v.literal("pending"),
      v.literal("composing"),   // Running composition (SIL, RGM, ACM)
      v.literal("composed"),    // Composition complete, ready for generation
      v.literal("running"),     // Running generation (AML)
      v.literal("evaluating"),  // Running evaluation (MDQA)
      v.literal("paused"),
      v.literal("completed"),
      v.literal("terminated"),
      v.literal("failed")
    ),
    progress: v.number(),
    generatedCount: v.optional(v.number()), // Actual reviews generated (0 until generation starts)
    generatedSentences: v.optional(v.number()), // Actual sentences generated (for sentence mode)
    failedCount: v.optional(v.number()), // Actual generation failures (LLM errors, etc.)
    currentPhase: v.optional(v.string()),
    error: v.optional(v.string()),
    createdAt: v.number(),
    completedAt: v.optional(v.number()),
    // Job directory path (e.g., "./jobs/abc123-my-job")
    jobDir: v.optional(v.string()),
    // Path to config.json in job directory
    configPath: v.optional(v.string()),
    // Reference to source job when reusing composition config
    reusedFrom: v.optional(v.id("jobs")),
    // Generated contexts from composition phase
    subjectContext: v.optional(v.any()), // SIL output - subject intelligence
    reviewerContext: v.optional(v.any()), // RGM output - reviewer personas
    attributesContext: v.optional(v.any()), // ACM output - attributes config
    // Conformity report from generation phase
    conformityReport: v.optional(v.object({
      polarity: v.number(), // 0-1: how well actual polarity matches target distribution
      length: v.number(), // 0-1: fraction of reviews within target length range
      noise: v.number(), // 0-1: noise application success rate
      validation: v.optional(v.number()), // 0-1: fraction of reviews with valid JSON structure
      temperature: v.optional(v.number()), // 0-1: fraction of reviews with temperature in target range
    })),
    // MDQA evaluation metrics
    evaluationMetrics: v.optional(v.object({
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
    })),
    // Compute device used for MDQA evaluation (GPU/CPU)
    evaluationDevice: v.optional(v.object({
      type: v.string(),  // "GPU" or "CPU"
      name: v.optional(v.string()),  // e.g., "NVIDIA GeForce RTX 3080"
    })),
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
    // Actual costs from pipeline execution (populated after completion)
    actualCost: v.optional(v.object({
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
    })),

    // Multi-run tracking (when total_runs > 1)
    currentRun: v.optional(v.number()),  // Current run number (1-indexed)
    totalRuns: v.optional(v.number()),   // Total number of runs

    // Per-run evaluation metrics (when total_runs > 1)
    perRunMetrics: v.optional(v.array(v.object({
      run: v.number(),
      datasetFile: v.string(),  // e.g., "reviews-run1.jsonl"
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

    // Average metrics across all runs (with std deviation)
    averageMetrics: v.optional(v.object({
      bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      rouge_l: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      bertscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      moverscore: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_1: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      distinct_2: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
      self_bleu: v.optional(v.object({ mean: v.number(), std: v.optional(v.number()) })),
    })),

    // Per-model progress tracking (multi-model jobs)
    modelProgress: v.optional(v.array(v.object({
      model: v.string(),        // Full ID e.g. "google/gemini-3-flash-preview"
      modelSlug: v.string(),    // Short name "gemini-3-flash-preview"
      generated: v.number(),
      failed: v.number(),
      target: v.number(),
      progress: v.number(),     // 0-100
      status: v.string(),       // "pending" | "generating" | "evaluating" | "completed" | "failed"
      evalProgress: v.optional(v.number()),
    }))),

    // Per-model evaluation results (multi-model jobs)
    perModelMetrics: v.optional(v.array(v.object({
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
    }))),

    // === HEURISTIC METHOD (RQ1 Baseline) ===
    // Generation method: "cera" (default), "heuristic", or "real" (eval-only on real datasets)
    method: v.optional(v.union(v.literal("cera"), v.literal("heuristic"), v.literal("real"))),

    // Heuristic-specific configuration (only for method="heuristic")
    heuristicConfig: v.optional(v.object({
      prompt: v.string(),                    // Prompt template with {review_count}, {avg_sentences}
      useFormatPrompt: v.optional(v.boolean()), // Whether to append format prompt (default true)
      formatPrompt: v.optional(v.string()),  // ABSA JSON format instructions
      targetMode: v.union(v.literal("reviews"), v.literal("sentences")),
      targetValue: v.number(),               // Total reviews OR total sentences based on mode
      reviewsPerBatch: v.number(),           // e.g., 50
      requestSize: v.optional(v.number()),   // Parallel batch concurrency (default: 3)
      avgSentencesPerReview: v.union(v.string(), v.number()),  // e.g., "5" or "4-7" (string for new jobs, number for legacy)
      model: v.string(),                     // LLM model
      outputFormat: v.string(),              // "semeval_xml" | "jsonl" | "csv"
      totalRuns: v.optional(v.number()),     // Number of times to run generation (default: 1)
      parallelRuns: v.optional(v.boolean()), // Run all runs concurrently (default: true)
      knowledgeSourceJobId: v.optional(v.string()), // Job ID to import SIL knowledge from
    })),

    // Heuristic job progress tracking
    heuristicProgress: v.optional(v.object({
      currentBatch: v.number(),
      totalBatches: v.number(),
      reviewsCollected: v.number(),
    })),
    // Per-run progress tracking (parallel heuristic runs)
    runProgress: v.optional(v.array(v.object({
      run: v.number(),           // Run number (1-based)
      status: v.string(),        // "generating" | "evaluating" | "completed" | "failed"
      currentBatch: v.number(),
      totalBatches: v.number(),
      reviewsCollected: v.number(),
      evalProgress: v.optional(v.number()), // 0-100 for evaluation phase
    }))),
  }).index("by_status", ["status"]).index("by_reused_from", ["reusedFrom"]),

  // Job logs (separate table for performance)
  logs: defineTable({
    jobId: v.id("jobs"),
    timestamp: v.number(),
    level: v.union(v.literal("INFO"), v.literal("WARN"), v.literal("ERROR")),
    phase: v.string(),
    message: v.string(),
  }).index("by_job", ["jobId"]),

  // Completed datasets
  datasets: defineTable({
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
    createdAt: v.number(),
  }).index("by_job", ["jobId"]),

  // User settings
  settings: defineTable({
    openrouterApiKey: v.optional(v.string()),
    tavilyApiKey: v.optional(v.string()), // For better web search results
    tavilyEnabled: v.optional(v.boolean()), // Toggle to use Tavily (true) or OpenRouter native search (false)
    convexAdminKey: v.optional(v.string()), // For Python API to call Convex mutations
    defaultProvider: v.string(),
    defaultModel: v.string(),
    theme: v.union(v.literal("light"), v.literal("dark"), v.literal("system")),
    jobsDirectory: v.string(), // Directory for storing job files (AML prompts, reports, dataset)
  }),

  // LLM Selection Presets
  llm_presets: defineTable({
    name: v.string(),
    isDefault: v.boolean(),
    rdeModel: v.optional(v.string()),           // Reference Dataset Extraction model
    mavModels: v.optional(v.array(v.string())), // MAV models (up to 3 for consensus)
    savModel: v.optional(v.string()),           // Single Agent Verification model (when MAV disabled)
    genModel: v.optional(v.string()),           // Generation model
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_default", ["isDefault"]),
});
