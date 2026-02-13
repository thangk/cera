import { useState } from 'react'
import { useMutation, useAction } from 'convex/react'
import { api } from 'convex/_generated/api'
import { Id } from 'convex/_generated/dataModel'
import {
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Play,
  Square,
  RefreshCcw,
  ChevronDown,
  BarChart3,
  FileText,
  Sparkles,
  Download,
  Trash2,
  DollarSign,
} from 'lucide-react'

import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Separator } from '../ui/separator'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '../ui/alert-dialog'

// Heuristic color (purple)
const HEURISTIC_COLOR = '#8b5cf6'
const HEURISTIC_LIGHT = '#f3e8ff'

// MDQA/Evaluation color (green) - matches CERA's evaluation phase
const MDQA_COLOR = '#8ed973'

// Parse avg sentences string (e.g. "4-7" → 5.5, "5" → 5) into numeric midpoint
function parseAvgSentences(val: string): number {
  const rangeMatch = val.match(/^(\d+)\s*-\s*(\d+)$/)
  if (rangeMatch) return (parseInt(rangeMatch[1]) + parseInt(rangeMatch[2])) / 2
  const num = parseFloat(val)
  return isNaN(num) || num <= 0 ? 5 : num
}

interface HeuristicTarget {
  targetMode: string
  targetValue: number
  reviewsPerBatch: number
  requestSize: number
  totalRuns: number
  runsMode: string
}

interface HeuristicJobViewProps {
  job: {
    _id: Id<'jobs'>
    name: string
    status: string
    progress: number
    currentPhase?: string
    error?: string
    createdAt: number
    completedAt?: number
    generatedCount?: number
    currentRun?: number
    totalRuns?: number
    heuristicConfig?: {
      prompt: string
      targetMode: string
      targetValue: number
      reviewsPerBatch: number
      avgSentencesPerReview: string
      model: string
      outputFormat: string
      totalRuns?: number
      targets?: HeuristicTarget[]
      parallelTargets?: boolean
      models?: string[]
      parallelModels?: boolean
    }
    heuristicProgress?: {
      currentBatch: number
      totalBatches: number
      reviewsCollected: number
    }
    runProgress?: Array<{
      run: number
      status: string
      currentBatch: number
      totalBatches: number
      reviewsCollected: number
      evalProgress?: number
    }>
    evaluationConfig?: {
      metrics: string[]
    }
    evaluationMetrics?: {
      bleu?: number
      rouge_l?: number
      bertscore?: number
      moverscore?: number
      distinct_1?: number
      distinct_2?: number
      self_bleu?: number
    }
    estimatedCost?: {
      total: { cost: number; calls: number; tokens: number }
    }
    actualCost?: {
      total?: { cost: number; calls: number; tokens: number }
      generation?: { cost: number; calls: number; tokens: number }
    }
    perRunMetrics?: Array<{
      run: number
      datasetFile: string
      metrics: {
        bleu?: number
        rouge_l?: number
        bertscore?: number
        moverscore?: number
        distinct_1?: number
        distinct_2?: number
        self_bleu?: number
      }
    }>
    averageMetrics?: {
      [key: string]: { mean: number; std?: number }
    }
    perTargetMetrics?: Array<{
      targetIndex: number
      targetLabel: string
      targetValue: number
      countMode: string
      metrics?: {
        bleu?: number
        rouge_l?: number
        bertscore?: number
        moverscore?: number
        distinct_1?: number
        distinct_2?: number
        self_bleu?: number
      }
      perRunMetrics?: Array<{
        run: number
        datasetFile: string
        metrics: {
          bleu?: number
          rouge_l?: number
          bertscore?: number
          moverscore?: number
          distinct_1?: number
          distinct_2?: number
          self_bleu?: number
        }
      }>
      averageMetrics?: {
        [key: string]: { mean: number; std?: number }
      }
    }>
  }
}

// Shared metric constants
const METRIC_THRESHOLDS: Record<string, { poor: number; good: number; excellent: number; lowerIsBetter?: boolean }> = {
  bleu: { poor: 0.25, good: 0.40, excellent: 0.50 },
  rouge_l: { poor: 0.25, good: 0.40, excellent: 0.50 },
  bertscore: { poor: 0.55, good: 0.70, excellent: 0.85 },
  moverscore: { poor: 0.35, good: 0.50, excellent: 0.65 },
  distinct_1: { poor: 0.30, good: 0.50, excellent: 0.70 },
  distinct_2: { poor: 0.60, good: 0.80, excellent: 0.90 },
  self_bleu: { poor: 0.50, good: 0.30, excellent: 0.20, lowerIsBetter: true },
}

const METRIC_ORDER: Array<{ key: string; label: string; higherIsBetter: boolean }> = [
  { key: 'bleu', label: 'BLEU', higherIsBetter: true },
  { key: 'rouge_l', label: 'ROUGE-L', higherIsBetter: true },
  { key: 'bertscore', label: 'BERTScore', higherIsBetter: true },
  { key: 'moverscore', label: 'MoverScore', higherIsBetter: true },
  { key: 'distinct_1', label: 'Distinct-1', higherIsBetter: true },
  { key: 'distinct_2', label: 'Distinct-2', higherIsBetter: true },
  { key: 'self_bleu', label: 'Self-BLEU', higherIsBetter: false },
]

const METRIC_CATEGORIES = [
  { label: 'Lexical', keys: ['bleu', 'rouge_l'], cols: 'sm:grid-cols-2' },
  { label: 'Semantic', keys: ['bertscore', 'moverscore'], cols: 'sm:grid-cols-2' },
  { label: 'Diversity', keys: ['distinct_1', 'distinct_2', 'self_bleu'], cols: 'sm:grid-cols-3' },
]

function getQualityIndicator(metricKey: string, value: number | undefined) {
  if (value === undefined) return null
  const threshold = METRIC_THRESHOLDS[metricKey]
  if (!threshold) return null
  if (threshold.lowerIsBetter) {
    if (value <= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400' }
    if (value <= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400' }
    if (value <= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400' }
    return { label: 'Poor', color: 'text-red-600 dark:text-red-400' }
  } else {
    if (value >= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400' }
    if (value >= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400' }
    if (value >= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400' }
    return { label: 'Poor', color: 'text-red-600 dark:text-red-400' }
  }
}

const fmt = (v: number | undefined) => v !== undefined ? v.toFixed(4) : '—'

function MetricCard({ label, value, higherIsBetter, metricKey, std }: { label: string; value: number | undefined; higherIsBetter: boolean; metricKey: string; std?: number }) {
  const threshold = METRIC_THRESHOLDS[metricKey]
  const quality = getQualityIndicator(metricKey, value)
  const thresholdText = threshold ? (
    threshold.lowerIsBetter
      ? `≤${threshold.good} good · ≤${threshold.excellent} excellent`
      : `≥${threshold.good} good · ≥${threshold.excellent} excellent`
  ) : ''

  return (
    <div className="rounded-lg border p-3 text-center">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <div className="flex items-center justify-center gap-2">
        <p className="text-xl font-bold" style={{ color: MDQA_COLOR }}>
          {fmt(value)}
          {std !== undefined && (
            <span className="text-xs font-normal text-muted-foreground"> ± {std.toFixed(4)}</span>
          )}
        </p>
        {quality && (
          <span className={`text-[10px] font-medium ${quality.color}`}>
            {quality.label}
          </span>
        )}
      </div>
      <p className="text-[10px] text-muted-foreground/70 mt-1">
        {higherIsBetter ? '↑ higher is better' : '↓ lower is better'}
      </p>
      {thresholdText && (
        <p className="text-[9px] text-muted-foreground/50 mt-0.5">
          {thresholdText}
        </p>
      )}
    </div>
  )
}

/** Summary tab: metric cards with optional avg ± std */
function SummaryContent({ metrics, averageMetrics, totalRuns }: {
  metrics: Record<string, number | undefined>
  averageMetrics?: Record<string, { mean: number; std?: number }>
  totalRuns: number
}) {
  const displayMetrics = averageMetrics || metrics
  const isAverage = !!averageMetrics
  return (
    <div className="space-y-4">
      {isAverage && (
        <p className="text-xs text-muted-foreground text-center">
          Average across {totalRuns} runs (with ± standard deviation)
        </p>
      )}
      {METRIC_CATEGORIES.map(({ label, keys, cols }) => {
        const hasMetrics = keys.some(k => (displayMetrics as Record<string, any>)[k] !== undefined)
        if (!hasMetrics) return null
        return (
          <div key={label}>
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{label}</h4>
            <div className={`grid gap-3 ${cols}`}>
              {keys.map(k => {
                const mo = METRIC_ORDER.find(m => m.key === k)!
                const value = isAverage ? (displayMetrics as any)[k]?.mean : (displayMetrics as Record<string, number | undefined>)[k]
                const std = isAverage ? (displayMetrics as any)[k]?.std : undefined
                return <MetricCard key={k} label={mo.label} value={value} higherIsBetter={mo.higherIsBetter} metricKey={k} std={std} />
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** Per Run tab: each run as a card with a full metric grid */
function PerRunContent({ runs }: { runs: Array<{ run: number; datasetFile: string; metrics: Record<string, number | undefined> }> }) {
  if (!runs || runs.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-4">No per-run metrics available.</p>
  }
  return (
    <div className="space-y-3">
      {runs.map((runData) => (
        <div key={runData.run} className="rounded-lg border p-3 bg-muted/30">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-sm">Run {runData.run}</span>
            <span className="text-xs text-muted-foreground">{runData.datasetFile}</span>
          </div>
          <div className="grid gap-2 grid-cols-2 sm:grid-cols-4 lg:grid-cols-7">
            {METRIC_ORDER.map(({ key, label }) => {
              const value = runData.metrics[key]
              if (value === undefined || value === null) return null
              const quality = getQualityIndicator(key, value)
              return (
                <div key={key} className="text-center p-2 rounded bg-background">
                  <p className="text-[10px] text-muted-foreground uppercase">{label}</p>
                  <p className="text-sm font-semibold" style={{ color: MDQA_COLOR }}>{(value as number).toFixed(4)}</p>
                  {quality && <span className={`text-[9px] ${quality.color}`}>{quality.label}</span>}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

/** Cross-target comparison table for "All" view */
function CrossTargetTable({ perTargetMetrics }: { perTargetMetrics: HeuristicJobViewProps['job']['perTargetMetrics'] }) {
  if (!perTargetMetrics || perTargetMetrics.length === 0) return null
  // Find which metrics are present across any target
  const presentKeys = METRIC_ORDER.filter(({ key }) =>
    perTargetMetrics.some(t => t.metrics && (t.metrics as Record<string, number | undefined>)[key] !== undefined)
  )
  if (presentKeys.length === 0) return null

  // Find best value per metric (for highlighting)
  const bestValues: Record<string, number> = {}
  for (const { key } of presentKeys) {
    const threshold = METRIC_THRESHOLDS[key]
    const values = perTargetMetrics
      .map(t => t.metrics ? (t.metrics as Record<string, number | undefined>)[key] : undefined)
      .filter((v): v is number => v !== undefined)
    if (values.length > 0) {
      bestValues[key] = threshold?.lowerIsBetter ? Math.min(...values) : Math.max(...values)
    }
  }

  return (
    <div>
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Cross-Target Comparison</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2 pr-3 font-medium text-muted-foreground">Target</th>
              {presentKeys.map(({ key, label }) => (
                <th key={key} className="text-center py-2 px-2 font-medium text-muted-foreground">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {perTargetMetrics.map((target) => (
              <tr key={target.targetIndex} className="border-b last:border-0">
                <td className="py-2 pr-3 font-medium whitespace-nowrap">
                  {target.targetLabel}
                </td>
                {presentKeys.map(({ key }) => {
                  const value = target.metrics ? (target.metrics as Record<string, number | undefined>)[key] : undefined
                  const isBest = value !== undefined && bestValues[key] === value && perTargetMetrics.length > 1
                  return (
                    <td key={key} className="text-center py-2 px-2">
                      <span className={isBest ? 'font-bold' : ''} style={isBest ? { color: MDQA_COLOR } : undefined}>
                        {fmt(value)}
                      </span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Wraps metrics content in Summary/Per Run tabs when per-run data exists */
function TabbedMetrics({ metrics, perRunMetrics, averageMetrics, totalRuns }: {
  metrics: Record<string, number | undefined>
  perRunMetrics?: Array<{ run: number; datasetFile: string; metrics: Record<string, number | undefined> }>
  averageMetrics?: Record<string, { mean: number; std?: number }>
  totalRuns: number
}) {
  const hasMultipleRuns = perRunMetrics && perRunMetrics.length > 1

  if (!hasMultipleRuns) {
    // Single run or no per-run data: just show metric cards directly
    return <SummaryContent metrics={metrics} totalRuns={1} />
  }

  // Multiple runs: show tabbed Summary / Per Run interface (matches CERA)
  return (
    <Tabs defaultValue="summary" className="w-full">
      <TabsList className="grid w-full grid-cols-2 mb-4">
        <TabsTrigger value="summary">Summary</TabsTrigger>
        <TabsTrigger value="per-run">Per Run ({perRunMetrics.length})</TabsTrigger>
      </TabsList>
      <TabsContent value="summary">
        <SummaryContent
          metrics={metrics}
          averageMetrics={averageMetrics}
          totalRuns={totalRuns}
        />
      </TabsContent>
      <TabsContent value="per-run">
        <PerRunContent runs={perRunMetrics} />
      </TabsContent>
    </Tabs>
  )
}

/** Main metrics display — target-aware */
function MetricsDisplay({ job, activeTarget, isMultiTarget }: {
  job: HeuristicJobViewProps['job']
  activeTarget: 'all' | number
  isMultiTarget: boolean
}) {
  const perTargetMetrics = job.perTargetMetrics

  // "All" view for multi-target: show cross-target comparison + aggregated cards
  if (isMultiTarget && activeTarget === 'all') {
    return (
      <>
        {perTargetMetrics && perTargetMetrics.length > 0 && (
          <CrossTargetTable perTargetMetrics={perTargetMetrics} />
        )}
        {job.evaluationMetrics && (
          <div className={perTargetMetrics && perTargetMetrics.length > 0 ? 'mt-4 pt-4 border-t' : ''}>
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Aggregated (All Targets)</h4>
            <TabbedMetrics
              metrics={job.evaluationMetrics as Record<string, number | undefined>}
              perRunMetrics={job.perRunMetrics as Array<{ run: number; datasetFile: string; metrics: Record<string, number | undefined> }> | undefined}
              averageMetrics={job.averageMetrics as Record<string, { mean: number; std?: number }> | undefined}
              totalRuns={job.perRunMetrics?.length || job.totalRuns || 1}
            />
          </div>
        )}
      </>
    )
  }

  // Specific target view: use per-target metrics if available
  if (typeof activeTarget === 'number' && perTargetMetrics) {
    const targetData = perTargetMetrics.find(t => t.targetIndex === activeTarget)
    if (targetData?.metrics) {
      const totalRuns = targetData.perRunMetrics?.length || 1
      return (
        <TabbedMetrics
          metrics={targetData.metrics as Record<string, number | undefined>}
          perRunMetrics={targetData.perRunMetrics as Array<{ run: number; datasetFile: string; metrics: Record<string, number | undefined> }> | undefined}
          averageMetrics={targetData.averageMetrics as Record<string, { mean: number; std?: number }> | undefined}
          totalRuns={totalRuns}
        />
      )
    }
  }

  // Fallback: legacy single-target display using job-level metrics
  if (job.evaluationMetrics) {
    const totalRuns = job.perRunMetrics?.length || job.totalRuns || job.heuristicConfig?.totalRuns || 1
    return (
      <TabbedMetrics
        metrics={job.evaluationMetrics as Record<string, number | undefined>}
        perRunMetrics={job.perRunMetrics as Array<{ run: number; datasetFile: string; metrics: Record<string, number | undefined> }> | undefined}
        averageMetrics={job.averageMetrics as Record<string, { mean: number; std?: number }> | undefined}
        totalRuns={totalRuns}
      />
    )
  }

  return null
}

export function HeuristicJobView({ job }: HeuristicJobViewProps) {
  const [promptExpanded, setPromptExpanded] = useState(false)
  const [metricsExpanded, setMetricsExpanded] = useState(true)

  // Multi-target support
  const targets = job.heuristicConfig?.targets || []
  const isMultiTarget = targets.length > 1
  const [activeTarget, setActiveTarget] = useState<'all' | number>(isMultiTarget ? 'all' : 0)

  const terminateJob = useMutation(api.jobs.terminate)
  const deleteJob = useMutation(api.jobs.remove)
  const runPipeline = useAction(api.pipelineAction.runPipeline)

  const handleTerminate = async () => {
    try {
      await terminateJob({ id: job._id })
    } catch (error) {
      console.error('Failed to terminate job:', error)
    }
  }

  const handleDelete = async () => {
    try {
      await deleteJob({ id: job._id })
      window.location.href = '/jobs'
    } catch (error) {
      console.error('Failed to delete job:', error)
    }
  }

  const handleRerun = async () => {
    try {
      await runPipeline({ jobId: job._id })
    } catch (error) {
      console.error('Failed to rerun pipeline:', error)
    }
  }

  // Status helpers
  const getStatusIcon = () => {
    switch (job.status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-muted-foreground" />
      case 'running':
        return <Loader2 className="h-5 w-5 text-purple-500 animate-spin" />
      case 'evaluating':
        return <Loader2 className="h-5 w-5 text-green-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
      case 'terminated':
        return <XCircle className="h-5 w-5 text-destructive" />
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getStatusLabel = () => {
    switch (job.status) {
      case 'pending':
        return 'Pending'
      case 'running':
        return 'Generating'
      case 'evaluating':
        return 'Evaluating'
      case 'completed':
        return 'Completed'
      case 'failed':
        return 'Failed'
      case 'terminated':
        return 'Terminated'
      default:
        return job.status
    }
  }

  const heuristicConfig = job.heuristicConfig
  const heuristicProgress = job.heuristicProgress

  // Active target config (for display in Generation Settings)
  const activeTargetConfig = typeof activeTarget === 'number' && targets[activeTarget]
    ? targets[activeTarget]
    : null

  // Calculate expected values based on active target (or legacy single-target config)
  const displayTargetMode = activeTargetConfig?.targetMode || heuristicConfig?.targetMode || 'reviews'
  const displayTargetValue = activeTargetConfig?.targetValue || heuristicConfig?.targetValue || 100
  const displayBatchSize = activeTargetConfig?.reviewsPerBatch || heuristicConfig?.reviewsPerBatch || 50
  const displayTotalRuns = activeTargetConfig?.totalRuns || heuristicConfig?.totalRuns || 1

  const totalReviews = displayTargetMode === 'reviews'
    ? displayTargetValue
    : Math.ceil(displayTargetValue / parseAvgSentences(heuristicConfig?.avgSentencesPerReview || '5'))

  const totalBatches = heuristicProgress?.totalBatches || Math.ceil(totalReviews / displayBatchSize)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold">{job.name}</h1>
            <Badge
              variant="secondary"
              style={{ backgroundColor: HEURISTIC_LIGHT, color: HEURISTIC_COLOR }}
            >
              Heuristic
            </Badge>
          </div>
          <p className="text-muted-foreground">
            Direct LLM Prompting
          </p>
        </div>

        {/* Status & Controls */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted">
            {getStatusIcon()}
            <span className="font-medium">{getStatusLabel()}</span>
          </div>

          {job.status === 'pending' && (
            <Button onClick={handleRerun} style={{ backgroundColor: HEURISTIC_COLOR }}>
              <Play className="mr-2 h-4 w-4" />
              Run Pipeline
            </Button>
          )}

          {(job.status === 'running' || job.status === 'evaluating') && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive">
                  <Square className="mr-2 h-4 w-4" />
                  Terminate
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Terminate Job?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will stop the generation process. Partial data may be lost.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleTerminate} className="bg-destructive text-destructive-foreground">
                    Terminate
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}

          {(job.status === 'completed' || job.status === 'failed' || job.status === 'terminated') && (
            <>
              <Button
                variant="outline"
                onClick={handleRerun}
                style={{ borderColor: HEURISTIC_COLOR, color: HEURISTIC_COLOR }}
              >
                <RefreshCcw className="mr-2 h-4 w-4" />
                Rerun
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="outline" className="text-destructive border-destructive">
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete Job?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will permanently delete this job and all its data.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground">
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          )}
        </div>
      </div>

      {/* Error Message */}
      {job.error && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
          <h3 className="font-medium text-destructive mb-1">Error</h3>
          <p className="text-sm text-destructive/80">{job.error}</p>
        </div>
      )}

      {/* Target selector - only for multi-target jobs */}
      {isMultiTarget && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Target:</span>
          <div className="flex gap-1 flex-wrap">
            <button
              onClick={() => setActiveTarget('all')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTarget === 'all'
                  ? 'text-white'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted'
              }`}
              style={activeTarget === 'all' ? { backgroundColor: HEURISTIC_COLOR } : undefined}
            >
              All
            </button>
            {targets.map((target, idx) => {
              const label = `${target.targetValue} ${target.targetMode === 'sentences' ? 'sent' : 'rev'}`
              return (
                <button
                  key={idx}
                  onClick={() => setActiveTarget(idx)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    activeTarget === idx
                      ? 'text-white'
                      : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                  }`}
                  style={activeTarget === idx ? { backgroundColor: HEURISTIC_COLOR } : undefined}
                >
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Progress Section */}
      {(job.status === 'running' || job.status === 'evaluating') && (
        <div className="rounded-lg border p-4 space-y-4" style={{ borderColor: HEURISTIC_COLOR }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin" style={{ color: HEURISTIC_COLOR }} />
              <span className="font-medium">
                {job.runProgress && job.runProgress.length > 1
                  ? `Running ${job.runProgress.length} Runs in Parallel`
                  : job.status === 'running' ? 'Generating Reviews' : 'Running MDQA Evaluation'}
              </span>
            </div>
            <span className="text-sm text-muted-foreground">
              {job.progress}%
            </span>
          </div>
          <Progress value={job.progress} className="h-2" indicatorColor={HEURISTIC_COLOR} />

          {/* Per-run progress bars (parallel runs) */}
          {job.runProgress && job.runProgress.length > 1 ? (
            <div className="space-y-3 pt-1">
              {job.runProgress
                .sort((a, b) => a.run - b.run)
                .map((rp) => {
                  const isCompleted = rp.status === 'completed'
                  const isEvaluating = rp.status === 'evaluating'
                  const isFailed = rp.status === 'failed'
                  const genPct = rp.totalBatches > 0 ? Math.round((rp.currentBatch / rp.totalBatches) * 100) : 0
                  const displayPct = isCompleted ? 100 : isFailed ? 100 : isEvaluating ? 50 + Math.round((rp.evalProgress || 0) / 2) : Math.round(genPct / 2)
                  const statusLabel = isCompleted ? 'Done' : isFailed ? 'Failed' : isEvaluating ? 'Evaluating' : 'Generating'
                  const statusColor = isCompleted ? '#22c55e' : isFailed ? '#ef4444' : isEvaluating ? '#f59e0b' : HEURISTIC_COLOR
                  return (
                    <div key={rp.run} className="space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-sm font-medium">Run {rp.run}</span>
                          <span
                            className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full"
                            style={{ color: statusColor, backgroundColor: `${statusColor}15` }}
                          >
                            {!isCompleted && !isFailed && (
                              <span className="relative flex h-1.5 w-1.5">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: statusColor }} />
                                <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ backgroundColor: statusColor }} />
                              </span>
                            )}
                            {statusLabel}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0">
                          <span>{rp.reviewsCollected} reviews</span>
                          {!isCompleted && !isEvaluating && (
                            <span>Batch {rp.currentBatch}/{rp.totalBatches}</span>
                          )}
                          <span className="font-medium w-8 text-right">{displayPct}%</span>
                        </div>
                      </div>
                      <Progress value={displayPct} className="h-1.5" indicatorColor={statusColor} />
                    </div>
                  )
                })}
            </div>
          ) : (
            <>
              {/* Single-run or sequential: show run indicator + batch counter */}
              {job.currentRun && job.totalRuns && job.totalRuns > 1 && (
                <div className="text-sm text-amber-600 dark:text-amber-400 font-medium">
                  {job.status === 'running' ? 'Generation' : 'Evaluation'} Run {job.currentRun}/{job.totalRuns}
                </div>
              )}

              {job.status === 'running' && heuristicProgress && (
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>
                    Batch {heuristicProgress.currentBatch} of {heuristicProgress.totalBatches}
                  </span>
                  <span>{heuristicProgress.reviewsCollected} reviews collected</span>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration */}
        <div className="space-y-4">
          {/* Generation Settings */}
          <div className="rounded-lg border p-4 space-y-4">
            <h3 className="font-semibold flex items-center gap-2">
              <Sparkles className="h-5 w-5" style={{ color: HEURISTIC_COLOR }} />
              Generation Settings
            </h3>
            <div className="grid gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">LLM Model{heuristicConfig?.models && heuristicConfig.models.length > 1 ? 's' : ''}</span>
                <div className="text-right">
                  {heuristicConfig?.models && heuristicConfig.models.length > 1
                    ? heuristicConfig.models.map((m, i) => (
                        <div key={i} className="font-mono text-xs">{m}</div>
                      ))
                    : <span className="font-mono text-xs">{heuristicConfig?.model || heuristicConfig?.models?.[0] || 'N/A'}</span>
                  }
                </div>
              </div>
              <Separator />
              {isMultiTarget && activeTarget === 'all' ? (
                <>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Targets</span>
                    <span>{targets.length} target datasets</span>
                  </div>
                  <div className="space-y-1">
                    {targets.map((t, i) => (
                      <div key={i} className="flex justify-between text-xs text-muted-foreground pl-2">
                        <span>Target {i + 1}</span>
                        <span>{t.targetValue.toLocaleString()} {t.targetMode} · {t.totalRuns} run{t.totalRuns > 1 ? 's' : ''}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Target Mode</span>
                    <span className="capitalize">{displayTargetMode}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Target Value</span>
                    <span>{displayTargetValue.toLocaleString()}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Reviews per Batch</span>
                    <span>{displayBatchSize}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avg Sentences</span>
                    <span>{heuristicConfig?.avgSentencesPerReview || 'N/A'}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Batches</span>
                    <span>{totalBatches}</span>
                  </div>
                  {displayTotalRuns > 1 && (
                    <>
                      <Separator />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Total Runs</span>
                        <span className="font-medium text-amber-600 dark:text-amber-400">{displayTotalRuns}</span>
                      </div>
                    </>
                  )}
                </>
              )}
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Output Format</span>
                <Badge variant="outline">{heuristicConfig?.outputFormat?.toUpperCase() || 'N/A'}</Badge>
              </div>
            </div>
          </div>

          {/* Cost Analysis */}
          <div className="rounded-lg border p-4 space-y-4">
            <h3 className="font-semibold flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-muted-foreground" />
              Cost Analysis
            </h3>
            <div className="grid gap-3 text-sm">
              {job.estimatedCost && (
                <>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Est. LLM Calls</span>
                    <span>{job.estimatedCost.total.calls.toLocaleString()}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Est. Tokens</span>
                    <span>{job.estimatedCost.total.tokens.toLocaleString()}</span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Est. Cost</span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      ${job.estimatedCost.total.cost.toFixed(4)}
                    </span>
                  </div>
                </>
              )}
              {job.actualCost?.total && job.status === 'completed' && (
                <>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Actual Calls</span>
                    <span className="text-amber-600 dark:text-amber-400">
                      {job.actualCost.total.calls.toLocaleString()}
                    </span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Actual Tokens</span>
                    <span className="text-amber-600 dark:text-amber-400">
                      {job.actualCost.total.tokens.toLocaleString()}
                    </span>
                  </div>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Actual Cost</span>
                    <span className="font-medium text-amber-600 dark:text-amber-400">
                      ${job.actualCost.total.cost.toFixed(4)}
                    </span>
                  </div>
                </>
              )}
              {!job.estimatedCost && !job.actualCost && (
                <p className="text-xs text-muted-foreground text-center py-2">
                  Cost data not available for this job
                </p>
              )}
            </div>
          </div>

          {/* Prompt Template */}
          <Collapsible open={promptExpanded} onOpenChange={setPromptExpanded}>
            <div className="rounded-lg border overflow-hidden">
              <CollapsibleTrigger asChild>
                <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                  <div className="flex items-center gap-2">
                    <FileText className="h-5 w-5" style={{ color: HEURISTIC_COLOR }} />
                    <span className="font-semibold">Prompt Template</span>
                  </div>
                  <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${promptExpanded ? 'rotate-180' : ''}`} />
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="border-t p-4">
                  <pre className="text-sm whitespace-pre-wrap font-mono bg-muted/30 p-3 rounded-lg overflow-x-auto">
                    {heuristicConfig?.prompt || 'No prompt available'}
                  </pre>
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Generation Results */}
          {(job.status === 'completed' || job.generatedCount) && (
            <div className="rounded-lg border p-4 space-y-4">
              <h3 className="font-semibold flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Generation Results
              </h3>
              <div className="grid gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Reviews Generated</span>
                  <span className="font-semibold text-lg">{job.generatedCount?.toLocaleString() || 'N/A'}</span>
                </div>
                {job.completedAt && (
                  <>
                    <Separator />
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Completed At</span>
                      <span>{new Date(job.completedAt).toLocaleString()}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* MDQA Metrics */}
          {(job.evaluationMetrics || (job.perTargetMetrics && job.perTargetMetrics.length > 0)) && (
            <Collapsible open={metricsExpanded} onOpenChange={setMetricsExpanded}>
              <div className="rounded-lg border overflow-hidden" style={{ borderColor: MDQA_COLOR }}>
                <CollapsibleTrigger asChild>
                  <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" style={{ color: MDQA_COLOR }} />
                      <span className="font-semibold">MDQA Metrics</span>
                    </div>
                    <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${metricsExpanded ? 'rotate-180' : ''}`} />
                  </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="border-t p-4 space-y-4">
                    <MetricsDisplay
                      job={job}
                      activeTarget={activeTarget}
                      isMultiTarget={isMultiTarget}
                    />
                  </div>
                </CollapsibleContent>
              </div>
            </Collapsible>
          )}

          {/* Timestamps */}
          <div className="rounded-lg border p-4 space-y-3 text-sm">
            <h3 className="font-semibold">Timeline</h3>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Created</span>
              <span>{new Date(job.createdAt).toLocaleString()}</span>
            </div>
            {job.completedAt && (
              <>
                <Separator />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Completed</span>
                  <span>{new Date(job.completedAt).toLocaleString()}</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
