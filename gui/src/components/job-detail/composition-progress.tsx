import { CheckCircle2, Search, Zap, Cpu } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Progress } from '@/components/ui/progress'
import { CircularProgress } from '@/components/ui/circular-progress'
import { Badge } from '@/components/ui/badge'

// Composition phase color (blue)
const PHASE_COLOR = {
  strong: '#4e95d9',
  light: '#dceaf7',
  // Dark mode compatible - semi-transparent overlay
  bgLight: 'rgba(78, 149, 217, 0.1)',
  bgDark: 'rgba(78, 149, 217, 0.2)',
}

// Phase definitions with substeps
interface Substep {
  range: [number, number]
  text: string
}

interface CompositionPhase {
  id: string
  label: string
  phase: string
  overallRange: [number, number]
  substeps: Substep[]
  conditional?: 'mavEnabled' | 'contextExtractionEnabled'
}

// Persona generation, writing patterns, and structure variants have been moved
// to the Generation phase (per-target) for multi-target support.
// Composition now only contains CTX, SIL, MAV, and ACM.
const COMPOSITION_PHASES: CompositionPhase[] = [
  {
    id: 'ctx',
    label: 'CTX',
    phase: 'CTX',
    overallRange: [0, 10],
    conditional: 'contextExtractionEnabled',
    substeps: [
      { range: [0, 20], text: 'Sampling reviews...' },
      { range: [20, 60], text: 'Extracting subject context...' },
      { range: [60, 100], text: 'Extracting reviewer context...' },
    ],
  },
  {
    id: 'sil',
    label: 'SIL',
    phase: 'SIL',
    overallRange: [10, 50],
    substeps: [
      { range: [0, 15], text: 'Researching subject...' },
      { range: [15, 35], text: 'Generating queries...' },
      { range: [35, 55], text: 'Deduplicating queries...' },
      { range: [55, 80], text: 'Verifying answers...' },
      { range: [80, 100], text: 'Classifying facts...' },
    ],
  },
  {
    id: 'mav',
    label: 'MAV',
    phase: 'MAV',
    overallRange: [50, 70],
    conditional: 'mavEnabled',
    substeps: [
      { range: [0, 40], text: 'Verifying queries...' },
      { range: [40, 80], text: 'Computing consensus...' },
      { range: [80, 100], text: 'Generating reports...' },
    ],
  },
  {
    id: 'acm',
    label: 'ACM',
    phase: 'ACM',
    overallRange: [70, 100],
    substeps: [
      { range: [0, 50], text: 'Configuring polarity...' },
      { range: [50, 100], text: 'Setting review attributes...' },
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

interface CompositionProgressProps {
  job: {
    status: string
    progress?: number
    currentPhase?: string
    config?: {
      subject_profile?: {
        mav?: {
          enabled?: boolean
        }
      }
    }
    referenceDataset?: {
      extractedSubjectContext?: boolean
      extractedReviewerContext?: boolean
    }
    // GPU/CPU used for similarity computations in SIL/MAV
    evaluationDevice?: {
      type: string
      name?: string
    }
  }
}

export function CompositionProgress({ job }: CompositionProgressProps) {
  const progress = job.progress ?? 0
  const mavEnabled = job.config?.subject_profile?.mav?.enabled ?? false
  const contextExtractionEnabled =
    job.referenceDataset?.extractedSubjectContext ||
    job.referenceDataset?.extractedReviewerContext ||
    false

  // Filter phases based on conditions
  const filteredPhases = COMPOSITION_PHASES.filter((phase) => {
    if (phase.conditional === 'mavEnabled' && !mavEnabled) {
      return false
    }
    if (phase.conditional === 'contextExtractionEnabled' && !contextExtractionEnabled) {
      return false
    }
    return true
  })

  // Recalculate ranges to span 0-100 when some phases are filtered out
  const totalOriginalRange = filteredPhases.reduce((sum, phase) => {
    return sum + (phase.overallRange[1] - phase.overallRange[0])
  }, 0)

  let currentStart = 0
  const visiblePhases = filteredPhases.map((phase) => {
    const originalSize = phase.overallRange[1] - phase.overallRange[0]
    const normalizedSize = (originalSize / totalOriginalRange) * 100
    const newRange: [number, number] = [currentStart, currentStart + normalizedSize]
    currentStart += normalizedSize
    return { ...phase, overallRange: newRange }
  })

  // Always show - displays pending state when job hasn't started yet
  const isActive =
    job.status === 'composing' ||
    job.status === 'composed' ||
    job.status === 'running' ||
    job.status === 'completed' ||
    job.status === 'paused' ||
    job.status === 'terminated' ||
    job.status === 'failed'

  // Show as pending/ready when job is queued or pending
  const isPending = job.status === 'pending' || job.status === 'queued'

  // Calculate composition-specific progress (0-100 for composition phase only)
  let compositionProgress: number
  if (isPending) {
    compositionProgress = 0
  } else if (job.status === 'composing') {
    compositionProgress = Math.min(progress, 100)
  } else {
    compositionProgress = 100
  }

  return (
    <div
      className="rounded-lg border overflow-hidden mb-6"
      style={{ borderColor: PHASE_COLOR.strong }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 border-b bg-[rgba(78,149,217,0.1)] dark:bg-[rgba(78,149,217,0.15)]"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASE_COLOR.light }}>
            <Search className="h-5 w-5" style={{ color: PHASE_COLOR.strong }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-foreground">Composition Progress</h3>
              {/* GPU/CPU Badge - for SIL/MAV similarity computations */}
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
              {visiblePhases.map((p) => p.label).join(', ')} - Subject intelligence gathering
            </p>
          </div>
        </div>
        <div className="text-right">
          <motion.span
            key={compositionProgress}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-lg font-semibold"
            style={{ color: isPending ? '#9ca3af' : PHASE_COLOR.strong }}
          >
            {isPending ? 'Pending' : job.status === 'composing' ? `${compositionProgress}%` : 'Complete'}
          </motion.span>
        </div>
      </div>

      {/* Overall Progress bar */}
      <div className="px-4 pt-4">
        <Progress
          value={compositionProgress}
          className="h-2"
          indicatorColor={PHASE_COLOR.strong}
          trackColor={PHASE_COLOR.light}
        />
      </div>

      {/* Circular progress rings */}
      <div className="p-4 pt-6">
        <div className="flex items-start justify-between gap-2">
          {visiblePhases.map((phase, index) => {
            const phaseProgress = getPhaseProgress(compositionProgress, phase.overallRange)
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
                          getPhaseProgress(compositionProgress, visiblePhases[index + 1].overallRange)
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
    </div>
  )
}
