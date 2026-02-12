import { CheckCircle2, BarChart3, Cpu, Zap } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Progress } from '@/components/ui/progress'
import { CircularProgress } from '@/components/ui/circular-progress'
import { Badge } from '@/components/ui/badge'

// Evaluation phase color (green)
const PHASE_COLOR = {
  strong: '#8ed973',
  light: '#d9f2d0',
}

// Phase definitions with substeps
interface Substep {
  range: [number, number]
  text: string
}

interface EvaluationPhase {
  id: string
  label: string
  shortLabel: string
  overallRange: [number, number]
  substeps: Substep[]
  conditional?: 'referenceRequired' // Only show if reference dataset is provided
}

// Phases are defined for the "with reference" scenario (all 4 circles)
// When no reference, Setup and Diversity still run but with adjusted ranges
const EVALUATION_PHASES: EvaluationPhase[] = [
  {
    id: 'setup',
    label: 'Setup',
    shortLabel: 'Setup',
    overallRange: [0, 15],
    substeps: [
      { range: [0, 40], text: 'Loading generated dataset...' },
      { range: [40, 70], text: 'Loading reference dataset...' },
      { range: [70, 100], text: 'Initializing models...' },
    ],
  },
  {
    id: 'lexical',
    label: 'Lexical',
    shortLabel: 'Lex',
    overallRange: [15, 40],
    conditional: 'referenceRequired',
    substeps: [
      { range: [0, 50], text: 'Assessing BLEU scores...' },
      { range: [50, 100], text: 'Assessing ROUGE-L scores...' },
    ],
  },
  {
    id: 'semantic',
    label: 'Semantic',
    shortLabel: 'Sem',
    overallRange: [40, 70],
    conditional: 'referenceRequired',
    substeps: [
      { range: [0, 50], text: 'Computing BERTScore...' },
      { range: [50, 100], text: 'Computing MoverScore...' },
    ],
  },
  {
    id: 'diversity',
    label: 'Diversity',
    shortLabel: 'Div',
    overallRange: [70, 100],
    substeps: [
      { range: [0, 35], text: 'Computing Distinct-1...' },
      { range: [35, 70], text: 'Computing Distinct-2...' },
      { range: [70, 100], text: 'Computing Self-BLEU...' },
    ],
  },
]

// Alternative phases when no reference dataset (only Setup + Diversity)
const EVALUATION_PHASES_NO_REF: EvaluationPhase[] = [
  {
    id: 'setup',
    label: 'Setup',
    shortLabel: 'Setup',
    overallRange: [0, 25],
    substeps: [
      { range: [0, 60], text: 'Loading generated dataset...' },
      { range: [60, 100], text: 'Initializing models...' },
    ],
  },
  {
    id: 'diversity',
    label: 'Diversity',
    shortLabel: 'Div',
    overallRange: [25, 100],
    substeps: [
      { range: [0, 35], text: 'Computing Distinct-1...' },
      { range: [35, 70], text: 'Computing Distinct-2...' },
      { range: [70, 100], text: 'Computing Self-BLEU...' },
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
function getSubstepText(phaseProgress: number, substeps: Substep[]): string {
  if (phaseProgress >= 100) return 'Done'
  if (phaseProgress === 0) return 'Pending'

  for (const substep of substeps) {
    if (phaseProgress >= substep.range[0] && phaseProgress < substep.range[1]) {
      return substep.text
    }
  }
  return substeps[substeps.length - 1].text
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
  progress: number
  status: string   // "pending" | "generating" | "evaluating" | "completed" | "failed"
  evalProgress?: number
}

interface EvaluationProgressProps {
  job: {
    status: string
    progress?: number
    currentPhase?: string
    evaluationMetrics?: {
      bleu?: number
      rouge_l?: number
      bertscore?: number
      moverscore?: number
      distinct_1?: number
      distinct_2?: number
      self_bleu?: number
    }
    evaluationDevice?: {
      type: string
      name?: string
    }
    referenceDataset?: {
      useForEvaluation?: boolean
      fileName?: string
    }
    evaluationConfig?: {
      reference_file?: string
      reference_metrics_enabled?: boolean
    }
    modelProgress?: ModelProgressEntry[]
  }
}

// Status badge colors for multi-model eval view
function getEvalModelStatusBadge(status: string) {
  switch (status) {
    case 'evaluating':
      return { label: 'Evaluating', color: PHASE_COLOR.strong, bg: 'bg-green-100 dark:bg-green-950/30', animate: true }
    case 'completed':
      return { label: 'Done', color: PHASE_COLOR.strong, bg: 'bg-green-100 dark:bg-green-950/30', animate: false }
    case 'failed':
      return { label: 'Failed', color: 'hsl(var(--destructive))', bg: 'bg-red-100 dark:bg-red-950/30', animate: false }
    case 'generating':
      return { label: 'Generating', color: '#f2aa84', bg: 'bg-orange-100 dark:bg-orange-950/30', animate: true }
    default:
      return { label: 'Pending', color: '#9ca3af', bg: 'bg-gray-100 dark:bg-gray-800', animate: false }
  }
}

export function EvaluationProgress({ job }: EvaluationProgressProps) {
  const overallProgress = job.progress ?? 0
  const currentPhase = job.currentPhase
  const isMultiModel = (job.modelProgress?.length ?? 0) > 1

  // Check if reference dataset is provided for lexical/semantic metrics
  const hasReferenceDataset =
    job.referenceDataset?.useForEvaluation ||
    !!job.evaluationConfig?.reference_file ||
    job.evaluationConfig?.reference_metrics_enabled ||
    false

  // Select the appropriate phases based on reference availability
  const visiblePhases = hasReferenceDataset ? EVALUATION_PHASES : EVALUATION_PHASES_NO_REF

  // Always show - displays pending state when job hasn't reached evaluation yet
  const isEvaluating = job.status === 'evaluating' || currentPhase?.toUpperCase() === 'MDQA'
  const hasMetrics = job.evaluationMetrics && Object.keys(job.evaluationMetrics).length > 0
  const isActive = isEvaluating || hasMetrics || job.status === 'completed'

  // Show as pending/ready when job hasn't reached evaluation phase
  const isPending = job.status === 'pending' || job.status === 'queued' || job.status === 'composing' || job.status === 'running'

  // Convert overall progress (80-100) to evaluation progress (0-100)
  // Evaluation starts at ~80% and ends at ~100% of overall pipeline
  let evaluationProgress: number
  if (isPending) {
    evaluationProgress = 0
  } else if (hasMetrics || job.status === 'completed') {
    evaluationProgress = 100
  } else if (isEvaluating) {
    // Map 80-100 to 0-100
    evaluationProgress = Math.max(0, Math.min(100, ((overallProgress - 80) / 20) * 100))
  } else {
    evaluationProgress = 0
  }

  // Multi-model eval progress: average of eval progresses across models that have started eval
  const multiModelEvalProgress = isMultiModel
    ? Math.round(
        job.modelProgress!.reduce((sum, mp) => {
          if (mp.status === 'completed') return sum + 100
          if (mp.status === 'evaluating') return sum + (mp.evalProgress ?? 0)
          return sum // pending/generating = 0
        }, 0) / job.modelProgress!.length
      )
    : evaluationProgress

  return (
    <div
      className="rounded-lg border overflow-hidden mb-6"
      style={{ borderColor: PHASE_COLOR.strong }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 border-b bg-[rgba(142,217,115,0.15)] dark:bg-[rgba(142,217,115,0.1)]"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASE_COLOR.light }}>
            <BarChart3 className="h-5 w-5" style={{ color: PHASE_COLOR.strong }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-foreground">Evaluation Progress</h3>
              {/* GPU/CPU Badge */}
              {job.evaluationDevice && (
                <Badge
                  variant="outline"
                  className={`text-[10px] px-1.5 py-0 h-5 ${
                    job.evaluationDevice.type === 'GPU'
                      ? 'border-green-500 text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/30'
                      : 'border-gray-400 text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/30'
                  }`}
                  title={job.evaluationDevice.name || job.evaluationDevice.type}
                >
                  {job.evaluationDevice.type === 'GPU' ? (
                    <Zap className="h-3 w-3 mr-0.5" />
                  ) : (
                    <Cpu className="h-3 w-3 mr-0.5" />
                  )}
                  {job.evaluationDevice.type}
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              MDQA - {visiblePhases.map((p) => p.label).join(', ')}
              {hasReferenceDataset && (
                <span className="ml-2 text-xs text-green-600 dark:text-green-400">
                  (vs. reference)
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="text-right">
          <motion.span
            key={isMultiModel ? multiModelEvalProgress : evaluationProgress}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-lg font-semibold"
            style={{ color: isPending && !isMultiModel ? '#9ca3af' : PHASE_COLOR.strong }}
          >
            {(() => {
              if (isMultiModel) {
                return multiModelEvalProgress >= 100 ? 'Complete' : `${multiModelEvalProgress}%`
              }
              return isPending ? 'Pending' : evaluationProgress === 100 ? 'Complete' : `${Math.round(evaluationProgress)}%`
            })()}
          </motion.span>
        </div>
      </div>

      {/* Overall Progress bar */}
      <div className="px-4 pt-4">
        <Progress
          value={isMultiModel ? multiModelEvalProgress : evaluationProgress}
          className="h-2"
          indicatorColor={PHASE_COLOR.strong}
          trackColor={PHASE_COLOR.light}
        />
      </div>

      {isMultiModel ? (
        /* Per-model evaluation bars (multi-model mode) */
        <div className="p-4 space-y-3">
          {job.modelProgress!.map((mp) => {
            const badge = getEvalModelStatusBadge(mp.status)
            const evalProg = mp.status === 'completed' ? 100 : (mp.evalProgress ?? 0)
            // Only show eval progress if model has reached evaluating or completed
            const showEvalBar = mp.status === 'evaluating' || mp.status === 'completed' || mp.status === 'failed'
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
                  <span className="text-xs font-medium shrink-0" style={{ color: evalProg >= 100 ? PHASE_COLOR.strong : '#9ca3af' }}>
                    {showEvalBar ? `${Math.round(evalProg)}%` : '--'}
                  </span>
                </div>
                <Progress
                  value={showEvalBar ? evalProg : 0}
                  className="h-1.5"
                  indicatorColor={mp.status === 'failed' ? 'hsl(var(--destructive))' : PHASE_COLOR.strong}
                  trackColor={showEvalBar ? PHASE_COLOR.light : '#e5e7eb'}
                />
              </div>
            )
          })}
        </div>
      ) : (
        /* Circular progress rings (single-model mode) */
        <div className="p-4 pt-6">
          <div className="flex items-start justify-between gap-2">
            {visiblePhases.map((phase, index) => {
              const phaseProgress = getPhaseProgress(evaluationProgress, phase.overallRange)
              const status = getPhaseStatus(phaseProgress)
              const substepText = getSubstepText(phaseProgress, phase.substeps)

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
                    <div className="text-center min-w-[80px]">
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
                  {index < visiblePhases.length - 1 && (
                    <motion.div
                      className="flex-1 h-0.5 min-w-4 mt-7"
                      initial={{ backgroundColor: '#e5e7eb' }}
                      animate={{
                        backgroundColor:
                          getPhaseStatus(
                            getPhaseProgress(evaluationProgress, visiblePhases[index + 1].overallRange)
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
      )}
    </div>
  )
}
