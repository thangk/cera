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
  }
}

export function HeuristicJobView({ job }: HeuristicJobViewProps) {
  const [promptExpanded, setPromptExpanded] = useState(false)
  const [metricsExpanded, setMetricsExpanded] = useState(true)

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

  // Calculate expected values
  const totalReviews = heuristicConfig?.targetMode === 'reviews'
    ? heuristicConfig.targetValue
    : Math.ceil((heuristicConfig?.targetValue || 100) / parseAvgSentences(heuristicConfig?.avgSentencesPerReview || '5'))

  const totalBatches = heuristicProgress?.totalBatches || Math.ceil(totalReviews / (heuristicConfig?.reviewsPerBatch || 50))

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
                <span className="text-muted-foreground">LLM Model</span>
                <span className="font-mono text-xs">{heuristicConfig?.model || 'N/A'}</span>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target Mode</span>
                <span className="capitalize">{heuristicConfig?.targetMode || 'reviews'}</span>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target Value</span>
                <span>{heuristicConfig?.targetValue?.toLocaleString() || 'N/A'}</span>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Reviews per Batch</span>
                <span>{heuristicConfig?.reviewsPerBatch || 'N/A'}</span>
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
              {(heuristicConfig?.totalRuns && heuristicConfig.totalRuns > 1) && (
                <>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Runs</span>
                    <span className="font-medium text-amber-600 dark:text-amber-400">{heuristicConfig.totalRuns}</span>
                  </div>
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
          {job.evaluationMetrics && (
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
                    {(() => {
                      const metrics = job.evaluationMetrics
                      const fmt = (v: number | undefined) => v !== undefined ? v.toFixed(4) : '—'

                      // Metric thresholds for quality indicators
                      const METRIC_THRESHOLDS: Record<string, { poor: number; good: number; excellent: number; lowerIsBetter?: boolean }> = {
                        bleu: { poor: 0.25, good: 0.40, excellent: 0.50 },
                        rouge_l: { poor: 0.25, good: 0.40, excellent: 0.50 },
                        bertscore: { poor: 0.55, good: 0.70, excellent: 0.85 },
                        moverscore: { poor: 0.35, good: 0.50, excellent: 0.65 },
                        distinct_1: { poor: 0.30, good: 0.50, excellent: 0.70 },
                        distinct_2: { poor: 0.60, good: 0.80, excellent: 0.90 },
                        self_bleu: { poor: 0.50, good: 0.30, excellent: 0.20, lowerIsBetter: true },
                      }

                      // Get quality indicator based on value and thresholds
                      const getQualityIndicator = (metricKey: string, value: number | undefined) => {
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

                      // Metric card component
                      const MetricCard = ({ label, value, higherIsBetter, metricKey }: { label: string; value: number | undefined; higherIsBetter: boolean; metricKey: string }) => {
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
                              <p className="text-xl font-bold" style={{ color: MDQA_COLOR }}>{fmt(value)}</p>
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

                      // Ordered metric keys for consistent display
                      const METRIC_ORDER: Array<{ key: string; label: string }> = [
                        { key: 'bleu', label: 'BLEU' },
                        { key: 'rouge_l', label: 'ROUGE-L' },
                        { key: 'bertscore', label: 'BERTScore' },
                        { key: 'moverscore', label: 'MoverScore' },
                        { key: 'distinct_1', label: 'Distinct-1' },
                        { key: 'distinct_2', label: 'Distinct-2' },
                        { key: 'self_bleu', label: 'Self-BLEU' },
                      ]

                      // Check if reference metrics are available
                      const hasLexicalMetrics = metrics.bleu !== undefined || metrics.rouge_l !== undefined
                      const hasSemanticMetrics = metrics.bertscore !== undefined || metrics.moverscore !== undefined
                      const hasDiversityMetrics = metrics.distinct_1 !== undefined || metrics.distinct_2 !== undefined || metrics.self_bleu !== undefined

                      return (
                        <>
                          {/* Lexical Metrics */}
                          {hasLexicalMetrics && (
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Lexical</h4>
                              <div className="grid gap-3 sm:grid-cols-2">
                                <MetricCard label="BLEU" value={metrics.bleu} higherIsBetter={true} metricKey="bleu" />
                                <MetricCard label="ROUGE-L" value={metrics.rouge_l} higherIsBetter={true} metricKey="rouge_l" />
                              </div>
                            </div>
                          )}

                          {/* Semantic Metrics */}
                          {hasSemanticMetrics && (
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Semantic</h4>
                              <div className="grid gap-3 sm:grid-cols-2">
                                <MetricCard label="BERTScore" value={metrics.bertscore} higherIsBetter={true} metricKey="bertscore" />
                                <MetricCard label="MoverScore" value={metrics.moverscore} higherIsBetter={true} metricKey="moverscore" />
                              </div>
                            </div>
                          )}

                          {/* Diversity Metrics */}
                          {hasDiversityMetrics && (
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Diversity</h4>
                              <div className="grid gap-3 sm:grid-cols-3">
                                <MetricCard label="Distinct-1" value={metrics.distinct_1} higherIsBetter={true} metricKey="distinct_1" />
                                <MetricCard label="Distinct-2" value={metrics.distinct_2} higherIsBetter={true} metricKey="distinct_2" />
                                <MetricCard label="Self-BLEU" value={metrics.self_bleu} higherIsBetter={false} metricKey="self_bleu" />
                              </div>
                            </div>
                          )}

                          {/* Multi-run average metrics (if available) */}
                          {job.averageMetrics && Object.keys(job.averageMetrics).length > 0 && (
                            <div className="mt-4 pt-4 border-t">
                              <h4 className="text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider mb-3">
                                Average Across {job.totalRuns || job.heuristicConfig?.totalRuns || 0} Runs (with ± std)
                              </h4>
                              <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-4">
                                {METRIC_ORDER
                                  .filter(({ key }) => job.averageMetrics![key])
                                  .map(({ key, label }) => {
                                    const value = job.averageMetrics![key]
                                    return (
                                      <div key={key} className="rounded-lg border bg-amber-50 dark:bg-amber-950/20 p-2 text-center">
                                        <p className="text-[10px] text-muted-foreground uppercase">{label}</p>
                                        <p className="text-sm font-semibold text-amber-700 dark:text-amber-300">
                                          {value.mean?.toFixed(4) || '—'}
                                          {value.std !== undefined && value.std !== null && (
                                            <span className="text-[10px] font-normal text-muted-foreground">
                                              {' '}± {value.std.toFixed(4)}
                                            </span>
                                          )}
                                        </p>
                                      </div>
                                    )
                                  })}
                              </div>
                            </div>
                          )}

                          {/* Per-run metrics breakdown (if available) */}
                          {job.perRunMetrics && job.perRunMetrics.length > 0 && (
                            <div className="mt-4">
                              <details className="text-xs">
                                <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                                  View per-run breakdown ({job.perRunMetrics.length} runs)
                                </summary>
                                <div className="mt-2 space-y-2">
                                  {job.perRunMetrics.map((runData) => (
                                    <div key={runData.run} className="rounded border p-2 bg-muted/30">
                                      <div className="flex justify-between items-center mb-1">
                                        <span className="font-medium">Run {runData.run}</span>
                                        <span className="text-muted-foreground">{runData.datasetFile}</span>
                                      </div>
                                      <div className="flex flex-wrap gap-2">
                                        {METRIC_ORDER
                                          .filter(({ key }) => (runData.metrics as Record<string, number | undefined>)[key] !== undefined && (runData.metrics as Record<string, number | undefined>)[key] !== null)
                                          .map(({ key, label }) => (
                                            <span key={key} className="inline-flex items-center gap-1 text-[10px]">
                                              <span className="text-muted-foreground">{label}:</span>
                                              <span className="font-medium">{((runData.metrics as Record<string, number>)[key]).toFixed(4)}</span>
                                            </span>
                                          ))}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </details>
                            </div>
                          )}
                        </>
                      )
                    })()}
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
