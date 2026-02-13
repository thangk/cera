/// <reference path="../pb_data/types.d.ts" />

// PocketBase migration: Create progress and logs collections for real-time updates

migrate(
  (app) => {
    // job_progress: single record per active job, upserted on each update
    const progress = new Collection({
      type: "base",
      name: "job_progress",
      fields: [
        { type: "text", name: "job_id", required: true },
        { type: "number", name: "progress" },
        { type: "text", name: "current_phase" },
        { type: "number", name: "generated_count" },
        { type: "number", name: "generated_sentences" },
        { type: "number", name: "failed_count" },
        { type: "number", name: "current_run" },
        { type: "number", name: "total_runs" },
        { type: "json", name: "model_progress" },
        { type: "json", name: "target_progress" },
        { type: "json", name: "run_progress" },
        { type: "json", name: "heuristic_progress" },
        // PB v0.36 requires explicit autodate fields
        { type: "autodate", name: "created", onCreate: true, onUpdate: false },
        { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
      ],
      indexes: [
        "CREATE UNIQUE INDEX idx_job_progress_job_id ON job_progress (job_id)",
      ],
      listRule: "",
      viewRule: "",
      createRule: "",
      updateRule: "",
      deleteRule: "",
    })
    app.save(progress)

    // job_logs: append-only log entries for real-time log panel
    const logs = new Collection({
      type: "base",
      name: "job_logs",
      fields: [
        { type: "text", name: "job_id", required: true },
        { type: "text", name: "level", required: true },
        { type: "text", name: "phase", required: true },
        { type: "text", name: "message", required: true },
        // PB v0.36 requires explicit autodate fields
        { type: "autodate", name: "created", onCreate: true, onUpdate: false },
        { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
      ],
      indexes: [
        "CREATE INDEX idx_job_logs_job_id ON job_logs (job_id)",
      ],
      listRule: "",
      viewRule: "",
      createRule: "",
      updateRule: null,
      deleteRule: "",
    })
    app.save(logs)
  },
  (app) => {
    // Rollback
    const progress = app.findCollectionByNameOrId("job_progress")
    app.delete(progress)
    const logs = app.findCollectionByNameOrId("job_logs")
    app.delete(logs)
  }
)
