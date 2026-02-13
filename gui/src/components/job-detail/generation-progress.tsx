import { CheckCircle2, FileText, Target, XCircle, Sliders, Clock } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Progress } from '@/components/ui/progress'
import { CircularProgress } from '@/components/ui/circular-progress'

// Generation phase color (orange/terracotta)
const PHASE_COLOR = {
  strong: '#f2aa84',
  light: '#fbe3d6',
}

// Phase definitions with substeps
interface Substep {
  range: [number, number]
  text: string
}

interface GenerationPhase {
  id: string
  label: string
  overallRange: [number, number]
  substeps: Substep[]
  dynamic?: boolean // For phases with dynamic substep text
}

// Backend sends progress 1-99 during generation, 100 on completion
// Personas phase runs first (1-5%), then AML+Reviews (5-90%), then Output+Validate
const GENERATION_PHASES: GenerationPhase[] = [
  {
    id: 'personas',
    label: 'Personas',
    overallRange: [0, 5],  // Persona generation + writing patterns + structure variants
    substeps: [
      { range: [0, 40], text: 'Generating reviewer personas...' },
      { range: [40, 70], text: 'Generating writing patterns...' },
      { range: [70, 100], text: 'Generating structure variants...' },
    ],
  },
  {
    id: 'aml',
    label: 'AML',
    overallRange: [5, 10],  // ~5% of progress (building prompts)
    substeps: [
      { range: [0, 50], text: 'Loading context docs...' },
      { range: [50, 100], text: 'Building AML prompts...' },
    ],
  },
  {
    id: 'reviews',
    label: 'Reviews',
    overallRange: [10, 90],  // Main generation phase
    dynamic: true,
    substeps: [
      { range: [0, 100], text: 'Generating reviews...' },
    ],
  },
  {
    id: 'output',
    label: 'Output',
    overallRange: [90, 98],  // Saving dataset (backend sends 99% here)
    substeps: [
      { range: [0, 50], text: 'Assembling dataset...' },
      { range: [50, 100], text: 'Writing output files...' },
    ],
  },
  {
    id: 'validation',
    label: 'Validate',
    overallRange: [98, 100],  // Final completion
    substeps: [
      { range: [0, 100], text: 'Finalizing...' },
    ],
  },
]

// Helper to get phase-local progress (0-100) from overall progress
function getPhaseProgress(overallProgress: number, phaseRange: [number, number]): number {
  const [start, end] = phaseRange
  if (overallProgress <= start) return 0
  if (overallProgress >= end) return 100
  return ((overallProgress - start) / (end - start)) * 100
}

// Helper to get current substep text
function getSubstepText(
  phase: GenerationPhase,
  phaseProgress: number,
  generatedCount?: number,
  targetCount?: number
): string {
  if (phaseProgress >= 100) return 'Done'
  if (phaseProgress === 0) return 'Pending'

  // Special handling for Reviews phase (dynamic count)
  if (phase.dynamic && phase.id === 'reviews' && generatedCount !== undefined && targetCount !== undefined) {
    const current = Math.min(generatedCount + 1, targetCount)
    return `Generating ${current}/${targetCount}...`
  }

  for (const substep of phase.substeps) {
    if (phaseProgress >= substep.range[0] && phaseProgress < substep.range[1]) {
      return substep.text
    }
  }
  return phase.substeps[phase.substeps.length - 1].text
}

type PhaseStatus = 'pending' | 'in_progress' | 'done'

function getPhaseStatus(phaseProgress: number): PhaseStatus {
  if (phaseProgress >= 100) return 'done'
  if (phaseProgress > 0) return 'in_progress'
  return 'pending'
}

interface ModelProgressEntry {
  model: string
  modelSlug: string
  generated: number
  failed: number
  target: number
  progress: number // 0-100
  status: string   // "pending" | "generating" | "evaluating" | "completed" | "failed"
  evalProgress?: number
}

interface GenerationProgressProps {
  job: {
    status: string
    progress?: number
    currentPhase?: string
    startedAt?: number
    config: {
      generation: {
        n_reviews: number
        model: string
        models?: string[]
        count_mode?: 'reviews' | 'sentences'
        target_sentences?: number
      }
    }
    generatedCount?: number
    generatedSentences?: number // Sentence count when in sentence mode
    failedCount?: number
    error?: string
    conformityReport?: {
      polarity: number
      length: number
      noise: number
      validation?: number
    }
    modelProgress?: ModelProgressEntry[]
  }
}

// Status badge colors for multi-model view
function getModelStatusBadge(status: string) {
  switch (status) {
    case 'generating':
      return { label: 'Generating', color: PHASE_COLOR.strong, bg: 'bg-orange-100 dark:bg-orange-950/30', animate: true }
    case 'evaluating':
      return { label: 'Evaluating', color: '#8ed973', bg: 'bg-green-100 dark:bg-green-950/30', animate: true }
    case 'completed':
      return { label: 'Done', color: '#8ed973', bg: 'bg-green-100 dark:bg-green-950/30', animate: false }
    case 'failed':
      return { label: 'Failed', color: 'hsl(var(--destructive))', bg: 'bg-red-100 dark:bg-red-950/30', animate: false }
    default:
      return { label: 'Pending', color: '#9ca3af', bg: 'bg-gray-100 dark:bg-gray-800', animate: false }
  }
}

export function GenerationProgress({ job }: GenerationProgressProps) {
  const progress = job.progress ?? 0
  const isMultiModel = (job.modelProgress?.length ?? 0) > 1
  const countMode = job.config?.generation?.count_mode || 'reviews'
  const isSentenceMode = countMode === 'sentences'

  // Target: use target_sentences for sentence mode, n_reviews for review mode
  const targetCount = isSentenceMode
    ? (job.config?.generation?.target_sentences ?? 0)
    : (job.config?.generation?.n_reviews ?? 0)

  // Generated: use generatedSentences for sentence mode, generatedCount for review mode
  const generatedCount = isSentenceMode
    ? (job.generatedSentences ?? job.generatedCount ?? 0)
    : (job.generatedCount ?? 0)

  const failedCount = job.failedCount ?? 0
  const modeLabel = isSentenceMode ? 'Sentences' : 'Reviews'

  // Calculate elapsed time
  const formatElapsedTime = () => {
    if (!job.startedAt) return '--:--'
    const elapsed = Math.floor((Date.now() - job.startedAt) / 1000)
    const hours = Math.floor(elapsed / 3600)
    const minutes = Math.floor((elapsed % 3600) / 60)
    const seconds = elapsed % 60
    if (hours > 0) {
      return `${hours}h ${minutes}m`
    }
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  // Always show - displays pending state when job hasn't reached generation yet
  const isActive =
    job.status === 'running' ||
    job.status === 'paused' ||
    job.status === 'completed' ||
    job.status === 'terminated' ||
    job.status === 'failed' ||
    job.status === 'evaluating'

  // Show as pending/ready when job is queued, pending, or still composing
  const isPending = job.status === 'pending' || job.status === 'queued' || job.status === 'composing'

  // Calculate generation-specific progress (0-100)
  // Backend sends progress 5-99 during generation, 100 on completion
  // We map this directly since it's already a percentage
  let generationProgress: number
  if (isPending) {
    generationProgress = 0
  } else if (job.status === 'completed' || job.status === 'evaluating') {
    generationProgress = 100
  } else if (job.status === 'running' || job.status === 'paused') {
    // Use progress directly (backend sends 5-99 during generation)
    generationProgress = Math.max(0, Math.min(100, progress))
  } else if (job.status === 'terminated' || job.status === 'failed') {
    // Show progress at termination point
    generationProgress = Math.max(0, Math.min(100, progress))
  } else {
    generationProgress = 0
  }

  const isTerminated = job.status === 'terminated' || job.status === 'failed'

  return (
    <div
      className="rounded-lg border overflow-hidden mb-6"
      style={{ borderColor: isTerminated ? 'hsl(var(--destructive))' : PHASE_COLOR.strong }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 border-b bg-[rgba(242,170,132,0.15)] dark:bg-[rgba(242,170,132,0.1)]"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASE_COLOR.light }}>
            <FileText className="h-5 w-5" style={{ color: PHASE_COLOR.strong }} />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Generation Progress</h3>
            <p className="text-sm text-muted-foreground">
              AML, Reviews, Noise, Validation
            </p>
          </div>
        </div>
        <div className="text-right">
          <motion.span
            key={generationProgress}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-lg font-semibold"
            style={{ color: isPending ? '#9ca3af' : isTerminated ? 'hsl(var(--destructive))' : PHASE_COLOR.strong }}
          >
            {isPending ? 'Pending' : generationProgress >= 100 ? 'Complete' : `${Math.round(generationProgress)}%`}
          </motion.span>
        </div>
      </div>

      {/* Overall Progress bar */}
      <div className="px-4 pt-4">
        <Progress
          value={generationProgress}
          className="h-2"
          indicatorColor={isTerminated ? 'hsl(var(--destructive))' : PHASE_COLOR.strong}
          trackColor={PHASE_COLOR.light}
        />
      </div>

      {isMultiModel ? (
        /* Per-model progress bars (multi-model mode) */
        <div className="p-4 space-y-3">
          {job.modelProgress!.map((mp) => {
            const badge = getModelStatusBadge(mp.status)
            const genProgress = mp.status === 'completed' || mp.status === 'evaluating' ? 100 : mp.progress
            return (
              <div key={mp.model} className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-sm font-medium truncate" title={mp.model}>
                      {mp.modelSlug}
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full ${badge.bg}`}
                      style={{ color: badge.color }}
                    >
                      {badge.animate && (
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: badge.color }} />
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ backgroundColor: badge.color }} />
                        </span>
                      )}
                      {badge.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0">
                    <span>
                      <span className="text-foreground font-medium">{mp.generated}</span>/{mp.target}
                      {mp.failed > 0 && (
                        <span className="text-red-500 ml-1">({mp.failed} failed)</span>
                      )}
                    </span>
                    <span className="font-medium w-8 text-right" style={{ color: genProgress >= 100 ? '#8ed973' : PHASE_COLOR.strong }}>
                      {Math.round(genProgress)}%
                    </span>
                  </div>
                </div>
                <Progress
                  value={genProgress}
                  className="h-1.5"
                  indicatorColor={mp.status === 'failed' ? 'hsl(var(--destructive))' : mp.status === 'evaluating' || mp.status === 'completed' ? '#8ed973' : PHASE_COLOR.strong}
                  trackColor={PHASE_COLOR.light}
                />
              </div>
            )
          })}
          {/* Elapsed time */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground pt-1">
            <Clock className="h-3.5 w-3.5" />
            <span>Elapsed: {formatElapsedTime()}</span>
          </div>
        </div>
      ) : (
        <>
          {/* Circular progress rings (single-model mode) */}
          <div className="p-4 pt-6">
            <div className="flex items-start justify-between gap-2">
              {GENERATION_PHASES.map((phase, index) => {
                const phaseProgress = getPhaseProgress(generationProgress, phase.overallRange)
                const status = getPhaseStatus(phaseProgress)
                const substepText = getSubstepText(phase, phaseProgress, generatedCount, targetCount)

                return (
                  <div key={phase.id} className="flex items-center gap-2">
                    {/* Phase indicator */}
                    <div className="flex flex-col items-center gap-2">
                      {/* Circular progress */}
                      <CircularProgress
                        value={phaseProgress}
                        size={56}
                        strokeWidth={4}
                        color={PHASE_COLOR.strong}
                        trackColor={status === 'pending' ? '#e5e7eb' : PHASE_COLOR.light}
                      >
                        <AnimatePresence mode="wait">
                          {status === 'done' ? (
                            <motion.div
                              key="check"
                              initial={{ scale: 0, rotate: -180 }}
                              animate={{ scale: 1, rotate: 0 }}
                              exit={{ scale: 0 }}
                              transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                            >
                              <CheckCircle2 className="h-5 w-5" style={{ color: PHASE_COLOR.strong }} />
                            </motion.div>
                          ) : (
                            <motion.span
                              key={`progress-${Math.round(phaseProgress)}`}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0, scale: 0.8 }}
                              className="text-xs font-semibold"
                              style={{ color: status === 'pending' ? '#9ca3af' : PHASE_COLOR.strong }}
                            >
                              {Math.round(phaseProgress)}%
                            </motion.span>
                          )}
                        </AnimatePresence>
                      </CircularProgress>

                      {/* Labels */}
                      <div className="text-center min-w-[70px]">
                        <p
                          className="text-xs font-medium transition-colors duration-300"
                          style={{
                            color:
                              status === 'pending'
                                ? '#9ca3af'
                                : PHASE_COLOR.strong,
                          }}
                        >
                          {phase.label}
                        </p>
                        <div className="h-[28px] mt-0.5 overflow-hidden">
                          <AnimatePresence mode="wait">
                            <motion.p
                              key={substepText}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -10 }}
                              transition={{ duration: 0.2 }}
                              className={`text-[10px] leading-tight ${
                                status === 'pending'
                                  ? 'text-muted-foreground/50'
                                  : 'text-muted-foreground'
                              }`}
                            >
                              {substepText}
                            </motion.p>
                          </AnimatePresence>
                        </div>
                      </div>
                    </div>

                    {/* Connector line */}
                    {index < GENERATION_PHASES.length - 1 && (
                      <motion.div
                        className="flex-1 h-0.5 min-w-4 mt-7"
                        initial={{ backgroundColor: '#e5e7eb' }}
                        animate={{
                          backgroundColor:
                            getPhaseStatus(
                              getPhaseProgress(generationProgress, GENERATION_PHASES[index + 1].overallRange)
                            ) !== 'pending'
                              ? PHASE_COLOR.strong
                              : '#e5e7eb',
                        }}
                        transition={{ duration: 0.3 }}
                      />
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Stats Grid (single-model mode) */}
          <div className="px-4 pb-4">
            <div className="grid gap-3 sm:grid-cols-5">
              <div className="rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Target className="h-4 w-4 text-blue-500" />
                  <span className="text-xs">Target ({modeLabel})</span>
                </div>
                <p className="text-lg font-semibold">{targetCount.toLocaleString()}</p>
              </div>
              <div className="rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-xs">Generated ({modeLabel})</span>
                </div>
                <p className="text-lg font-semibold text-green-600">
                  {generatedCount.toLocaleString()}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-xs">Failed</span>
                </div>
                <p className={`text-lg font-semibold ${failedCount > 0 ? 'text-red-500' : 'text-muted-foreground'}`}>
                  {failedCount.toLocaleString()}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Sliders className="h-4 w-4" />
                  <span className="text-xs">Model</span>
                </div>
                <p className="text-sm font-semibold truncate" title={job.config.generation.model}>
                  {job.config.generation.model.split('/').pop()}
                </p>
              </div>
              <div className="rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-muted-foreground mb-1">
                  <Clock className="h-4 w-4" />
                  <span className="text-xs">Elapsed</span>
                </div>
                <p className="text-lg font-semibold">{formatElapsedTime()}</p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Error message if failed */}
      {job.error && (
        <div className="px-4 pb-4">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3">
            <p className="text-sm text-destructive">
              <span className="font-medium">Error:</span> {job.error}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
