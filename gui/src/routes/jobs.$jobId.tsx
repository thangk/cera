import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation, useAction } from 'convex/react'
import { api } from 'convex/_generated/api'
import { Id } from 'convex/_generated/dataModel'
import { useState, useMemo, useEffect } from 'react'
import {
  Clock,
  CheckCircle,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowLeft,
  Users,
  Sliders,
  AlertCircle,
  Play,
  Pause,
  Square,
  RefreshCcw,
  Activity,
  ChevronDown,
  BarChart3,
  ClipboardCheck,
  Network,
  Search,
  Download,
  Bot,
  Eye,
  Trash2,
  DollarSign,
} from 'lucide-react'

import { useOpenRouterModels, type ProcessedModel } from '../hooks/use-openrouter-models'
import { usePocketBaseProgress } from '../hooks/use-pocketbase'
import { PYTHON_API_URL } from '../lib/api-urls'

import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Skeleton } from '../components/ui/skeleton'
import { ScrollArea } from '../components/ui/scroll-area'
import { Separator } from '../components/ui/separator'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../components/ui/collapsible'
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
} from '../components/ui/alert-dialog'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { CompositionProgress } from '../components/job-detail/composition-progress'
import { EvaluationProgress } from '../components/job-detail/evaluation-progress'
import { GenerationProgress } from '../components/job-detail/generation-progress'
import { PhaseLogPanel } from '../components/job-detail/phase-log-panel'
import { HeuristicJobView } from '../components/heuristic'
import type { LogTab } from '../hooks/use-log-panel'

export const Route = createFileRoute('/jobs/$jobId')({
  component: JobDetailPage,
})

// Token estimates for cost calculation (matches create-job.tsx)
const TOKEN_ESTIMATES = {
  // SIL Round 1: Research subject (understand + search)
  sil_research: { input: 2500, output: 2000 },
  // SIL Round 1: Generate factual queries
  sil_generate_queries: { input: 3000, output: 1500 },
  // SIL Round 3: Answer all pooled queries (larger prompt with all queries)
  sil_answer_queries: { input: 4000, output: 3000 },
  // SIL Post-consensus: Classify verified facts
  sil_classify: { input: 2000, output: 1000 },
  // AML: Per review generation
  aml_per_review: { input: 800, output: 200 },
}

// 3-Phase colors matching create-job
const PHASES = [
  {
    id: 'composition',
    title: 'COMPOSITION',
    strongColor: '#4e95d9',
    lightColor: '#dceaf7',
    bgRgba: 'rgba(78, 149, 217, 0.15)',
    description: 'SIL, MAV - Subject intelligence gathering',
  },
  {
    id: 'generation',
    title: 'GENERATION',
    strongColor: '#f2aa84',
    lightColor: '#fbe3d6',
    bgRgba: 'rgba(242, 170, 132, 0.2)',
    description: 'AML - Review generation',
  },
  {
    id: 'evaluation',
    title: 'EVALUATION',
    strongColor: '#8ed973',
    lightColor: '#d9f2d0',
    bgRgba: 'rgba(142, 217, 115, 0.15)',
    description: 'MDQA metrics and reports',
  },
]

// Map job.currentPhase to phase index
function getPhaseIndex(currentPhase: string | undefined): number {
  if (!currentPhase) return -1
  if (currentPhase === 'composition' || currentPhase === 'sil' || currentPhase === 'mav') return 0
  if (currentPhase === 'generation' || currentPhase === 'aml') return 1
  if (currentPhase === 'evaluation' || currentPhase === 'mdqa') return 2
  return -1
}

// PhaseTab component
function PhaseTab({
  phase,
  progress,
  isActive,
  onClick,
  status,
}: {
  phase: typeof PHASES[0]
  progress: number
  isActive: boolean
  onClick: () => void
  status: 'pending' | 'ready' | 'in_progress' | 'done'
}) {
  return (
    <button
      onClick={onClick}
      className={`relative px-4 py-2 rounded-lg overflow-hidden transition-all ${
        isActive ? 'ring-2 ring-offset-2 ring-offset-background' : 'hover:opacity-80'
      } ${status === 'pending' ? 'opacity-60' : ''}`}
      style={{
        backgroundColor: phase.bgRgba,
        // @ts-expect-error CSS custom property
        '--tw-ring-color': phase.strongColor,
      }}
    >
      {/* Progress fill */}
      <div
        className="absolute inset-y-0 left-0"
        style={{
          width: `${progress}%`,
          backgroundColor: phase.strongColor,
          opacity: 0.25,
        }}
      />
      {/* Vertical layout: title on top, badge below */}
      <div className="relative flex flex-col items-center gap-1">
        <span className="font-medium text-sm" style={{ color: phase.strongColor }}>
          {phase.title}
        </span>
        {/* Status indicator - always a text badge */}
        <span
          className={`text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded ${
            status === 'pending' ? 'opacity-60' : ''
          }`}
          style={{
            backgroundColor: status === 'pending' ? 'transparent' : phase.strongColor,
            color: status === 'pending' ? phase.strongColor : 'white',
            border: status === 'pending' ? `1px solid ${phase.strongColor}` : 'none',
          }}
        >
          {status === 'done' && 'Completed'}
          {status === 'ready' && 'Ready'}
          {status === 'in_progress' && 'In Progress'}
          {status === 'pending' && 'Pending'}
        </span>
      </div>
    </button>
  )
}

function JobDetailPage() {
  const { jobId } = Route.useParams()
  const convexJob = useQuery(api.jobs.get, { id: jobId as Id<'jobs'> })
  const pbProgress = usePocketBaseProgress(jobId)
  const dataset = useQuery(api.datasets.getByJob, { jobId: jobId as Id<'jobs'> })

  // Merge PocketBase real-time progress with Convex job state.
  // PocketBase handles high-frequency updates (progress, counts) for smooth animations.
  // Convex remains source of truth for job config, status transitions, and final metrics.
  const job = useMemo(() => {
    if (!convexJob) return convexJob
    const isActive = ['composing', 'running', 'evaluating'].includes(convexJob.status)
    if (!isActive || !pbProgress) return convexJob
    return {
      ...convexJob,
      progress: pbProgress.progress ?? convexJob.progress,
      currentPhase: pbProgress.current_phase || convexJob.currentPhase,
      generatedCount: pbProgress.generated_count ?? convexJob.generatedCount,
      generatedSentences: pbProgress.generated_sentences ?? convexJob.generatedSentences,
      failedCount: pbProgress.failed_count ?? convexJob.failedCount,
      currentRun: pbProgress.current_run ?? convexJob.currentRun,
      totalRuns: pbProgress.total_runs ?? convexJob.totalRuns,
      modelProgress: pbProgress.model_progress ?? convexJob.modelProgress,
      targetProgress: pbProgress.target_progress ?? (convexJob as any).targetProgress,
      runProgress: pbProgress.run_progress ?? (convexJob as any).runProgress,
      heuristicProgress: pbProgress.heuristic_progress ?? (convexJob as any).heuristicProgress,
    }
  }, [convexJob, pbProgress])

  // OpenRouter models for rich MAV display
  const { models: rawModels, processedModels } = useOpenRouterModels()

  // Lookup model by ID
  const getModelInfo = useMemo(() => {
    const modelMap = new Map<string, ProcessedModel>()
    if (processedModels) {
      processedModels.forEach(m => modelMap.set(m.id, m))
    }
    return (modelId: string) => modelMap.get(modelId)
  }, [processedModels])

  // Helper to get model pricing from raw OpenRouter models
  // Returns pricing per 1M tokens
  const getModelPricing = useMemo(() => {
    return (modelId: string): { input: number; output: number } | null => {
      const model = rawModels.find((m) => m.id === modelId)
      if (!model) return null
      // OpenRouter pricing is per token, convert to per 1M tokens
      const inputPrice = parseFloat(model.pricing?.prompt || '0') * 1_000_000
      const outputPrice = parseFloat(model.pricing?.completion || '0') * 1_000_000
      return { input: inputPrice, output: outputPrice }
    }
  }, [rawModels])

  // Phase tab state - uses phase ID string
  const jobPhases = job?.phases || ['composition', 'generation', 'evaluation']
  const visiblePhases = useMemo(() => PHASES.filter(p => jobPhases.includes(p.id)), [jobPhases])
  const enabledPhaseIds = useMemo(() => ['all', ...visiblePhases.map(p => p.id)] as LogTab[], [visiblePhases])
  const [activeTab, setActiveTab] = useState(visiblePhases[0]?.id || 'composition')

  // Multi-target support
  const targets = (job?.config?.generation?.targets || []) as Array<{
    count_mode: 'sentences' | 'reviews'
    target_value: number
    batch_size: number
    request_size: number
    total_runs: number
    runs_mode: 'parallel' | 'sequential'
    neb_depth?: number
  }>
  const isMultiTarget = targets.length > 1
  const [activeTarget, setActiveTarget] = useState<'all' | number>('all')

  // Create a target-scoped job object for per-target progress components
  // Maps target-specific progress/status/metrics onto job-level fields so existing
  // GenerationProgress and EvaluationProgress components work without changes.
  // For single-target jobs with perTargetMetrics, maps the first target's data automatically.
  const targetScopedJob = useMemo(() => {
    if (!job) return job
    const perTargetMetrics = (job as any).perTargetMetrics as Array<Record<string, unknown>> | undefined

    // For multi-target jobs with a specific target selected
    if (isMultiTarget && typeof activeTarget === 'number') {
      const tp = (job as any).targetProgress?.find((tp: { targetIndex: number }) => tp.targetIndex === activeTarget)
      const targetConfig = targets[activeTarget]
      if (!targetConfig) return job

      const statusMap: Record<string, string> = {
        pending: 'composing',
        generating: 'running',
        evaluating: 'evaluating',
        completed: 'completed',
      }

      const ptMetrics = perTargetMetrics?.find(
        (pt: { targetIndex?: number }) => pt.targetIndex === activeTarget
      )

      return {
        ...job,
        ...(tp ? {
          status: statusMap[tp.status as string] || job.status,
          progress: tp.progress,
        } : {}),
        ...(ptMetrics?.metrics ? { evaluationMetrics: ptMetrics.metrics } : {}),
        ...(ptMetrics?.perModelMetrics ? { perModelMetrics: ptMetrics.perModelMetrics } : {}),
        ...(ptMetrics?.conformity ? { conformityReport: ptMetrics.conformity } : {}),
        ...(ptMetrics?.perRunMetrics ? { perRunMetrics: ptMetrics.perRunMetrics } : {}),
        ...(ptMetrics?.averageMetrics ? { averageMetrics: ptMetrics.averageMetrics } : {}),
        totalRuns: targetConfig.total_runs || job.totalRuns,
        config: {
          ...job.config,
          generation: {
            ...job.config.generation,
            count_mode: targetConfig.count_mode,
            target_sentences: targetConfig.count_mode === 'sentences' ? targetConfig.target_value : undefined,
            n_reviews: targetConfig.count_mode === 'reviews' ? targetConfig.target_value : undefined,
            total_runs: targetConfig.total_runs,
          },
        },
      }
    }

    // For single-target jobs that have perTargetMetrics (new pipeline path):
    // auto-map the first target's metrics onto job-level fields
    if (!isMultiTarget && perTargetMetrics && perTargetMetrics.length > 0) {
      const ptMetrics = perTargetMetrics[0]
      return {
        ...job,
        ...(ptMetrics?.metrics ? { evaluationMetrics: ptMetrics.metrics } : {}),
        ...(ptMetrics?.perModelMetrics ? { perModelMetrics: ptMetrics.perModelMetrics } : {}),
        ...(ptMetrics?.conformity ? { conformityReport: ptMetrics.conformity } : {}),
        ...(ptMetrics?.perRunMetrics ? { perRunMetrics: ptMetrics.perRunMetrics } : {}),
        ...(ptMetrics?.averageMetrics ? { averageMetrics: ptMetrics.averageMetrics } : {}),
      }
    }

    return job
  }, [job, isMultiTarget, activeTarget, targets])

  // Update activeTab when job data loads and visible phases change
  useEffect(() => {
    if (visiblePhases.length > 0 && !visiblePhases.some(p => p.id === activeTab)) {
      setActiveTab(visiblePhases[0].id)
    }
  }, [visiblePhases, activeTab])

  // Collapsible states for each phase
  const [compositionSections, setCompositionSections] = useState({
    subject: true,
    reviewer: false,
    attributes: false,
    mav: false,
  })
  const [generationSections, setGenerationSections] = useState({
    progress: true,
    conformity: false,
  })
  const [evaluationSections, setEvaluationSections] = useState({
    mdqa: true,
  })

  // Dialog states
  const [contextsDialogOpen, setContextsDialogOpen] = useState(false)
  const [rerunCompositionDialogOpen, setRerunCompositionDialogOpen] = useState(false)
  const [isRerunningComposition, setIsRerunningComposition] = useState(false)

  // Mutations for job control
  const pauseJob = useMutation(api.jobs.pause)
  const resumeJob = useMutation(api.jobs.resume)
  const terminateJob = useMutation(api.jobs.terminate)
  const rerunJob = useMutation(api.jobs.rerun)
  const deleteJob = useMutation(api.jobs.remove)
  const fullReset = useMutation(api.jobs.fullReset)
  const runComposition = useAction(api.compositionAction.runComposition)
  const runGeneration = useAction(api.generationAction.runGeneration)
  const runPipeline = useAction(api.pipelineAction.runPipeline)
  const rerunGenerationAction = useAction(api.evaluationAction.rerunGeneration)
  const runEvaluationAction = useAction(api.evaluationAction.runEvaluation)

  // Control handlers
  const handlePause = async () => {
    try {
      await pauseJob({ id: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to pause job:', error)
    }
  }

  const handleResume = async () => {
    try {
      await resumeJob({ id: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to resume job:', error)
    }
  }

  const handleTerminate = async () => {
    try {
      await terminateJob({ id: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to terminate job:', error)
    }
  }

  const handleRerun = async () => {
    try {
      await rerunJob({ id: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to rerun job:', error)
    }
  }

  // Rerun generation (and evaluation if originally included in phases)
  const handleRerunGeneration = async () => {
    try {
      await rerunGenerationAction({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to rerun generation:', error)
    }
  }

  // Rerun only the evaluation phase
  const handleRerunEvaluation = async () => {
    try {
      await runEvaluationAction({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to rerun evaluation:', error)
    }
  }

  const navigate = Route.useNavigate()
  const handleDelete = async () => {
    try {
      // Delete job directory if it exists
      if (job?.jobDir) {
        const pythonApiUrl = PYTHON_API_URL
        try {
          await fetch(`${pythonApiUrl}/api/delete-job-dir`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jobDir: job.jobDir }),
          })
        } catch (e) {
          // Log but continue with Convex deletion even if dir deletion fails
          console.warn('Failed to delete job directory:', e)
        }
      }
      await deleteJob({ id: jobId as Id<'jobs'> })
      navigate({ to: '/jobs' })
    } catch (error) {
      console.error('Failed to delete job:', error)
    }
  }

  const handleStartComposition = async () => {
    try {
      await runComposition({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to start composition:', error)
    }
  }

  const handleStartGeneration = async () => {
    try {
      await runGeneration({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to start generation:', error)
    }
  }

  const handleRunPipeline = async () => {
    try {
      await runPipeline({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to run pipeline:', error)
    }
  }

  const handleRerunComposition = async () => {
    setRerunCompositionDialogOpen(true)
    setIsRerunningComposition(true)
    try {
      // First reset the job to pending (clears contexts)
      await fullReset({ id: jobId as Id<'jobs'> })
      // Then start composition
      await runComposition({ jobId: jobId as Id<'jobs'> })
    } catch (error) {
      console.error('Failed to rerun composition:', error)
    } finally {
      setIsRerunningComposition(false)
    }
  }

  // Export report as zip file
  const handleExportReport = async (reportType: 'mav' | 'conformity' | 'metrics') => {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/jobs/${jobId}/export/${reportType}`)
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Export failed')
      }
      // Download the zip file
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${jobId}-${reportType}-report.zip`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error(`Failed to export ${reportType} report:`, error)
    }
  }

  if (job === undefined) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <Skeleton className="h-8 w-64" />
        <div className="flex gap-2">
          <Skeleton className="h-10 w-32" />
          <Skeleton className="h-10 w-32" />
          <Skeleton className="h-10 w-32" />
        </div>
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  if (job === null) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <AlertCircle className="h-12 w-12 text-muted-foreground" />
        <p className="text-lg text-muted-foreground">Job not found</p>
        <Button asChild variant="outline">
          <Link to="/jobs">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Jobs
          </Link>
        </Button>
      </div>
    )
  }

  // Heuristic jobs use a different, simpler view
  if (job.method === 'heuristic') {
    return (
      <div className="flex flex-col gap-6 p-6">
        {/* Back button */}
        <div>
          <Button asChild variant="ghost" size="sm">
            <Link to="/jobs">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Jobs
            </Link>
          </Button>
        </div>
        <HeuristicJobView job={job as any} />
      </div>
    )
  }

  const statusConfig = {
    pending: { icon: Clock, color: 'bg-yellow-500/10 text-yellow-500', label: 'Pending' },
    composing: { icon: Loader2, color: 'bg-purple-500/10 text-purple-500', label: 'Composing' },
    composed: { icon: CheckCircle, color: 'bg-indigo-500/10 text-indigo-500', label: 'Composed' },
    running: { icon: Loader2, color: 'bg-blue-500/10 text-blue-500', label: 'Running' },
    evaluating: { icon: Loader2, color: 'bg-emerald-500/10 text-emerald-500', label: 'Evaluating' },
    paused: { icon: Pause, color: 'bg-orange-500/10 text-orange-500', label: 'Paused' },
    completed: { icon: CheckCircle, color: 'bg-green-500/10 text-green-500', label: 'Completed' },
    terminated: { icon: Square, color: 'bg-gray-500/10 text-gray-500', label: 'Terminated' },
    failed: { icon: XCircle, color: 'bg-red-500/10 text-red-500', label: 'Failed' },
  }

  const baseStatus = statusConfig[job.status as keyof typeof statusConfig] || statusConfig.pending

  // Get counts from job (failedCount tracks actual LLM failures, not remaining reviews)
  const generatedCount = job.generatedCount ?? 0
  const targetCount = job.config.generation?.count ?? 0
  // Use actual failed count from job if available, otherwise 0 (not target - generated)
  const failedCount = job.failedCount ?? 0

  // Create enhanced status with descriptive label for completed/failed/terminated jobs
  const getEnhancedStatusLabel = () => {
    if (!['completed', 'failed', 'terminated'].includes(job.status)) {
      return baseStatus.label
    }
    // Show generation result suffix for finished jobs
    if (generatedCount === 0 && targetCount > 0) {
      return `${baseStatus.label} (All Failed)`
    } else if (failedCount === 0) {
      return `${baseStatus.label} (Success)`
    } else {
      // Show number of failures if any actual failures occurred
      return `${baseStatus.label} (${failedCount} Failed)`
    }
  }

  const status = {
    ...baseStatus,
    label: getEnhancedStatusLabel(),
  }
  const StatusIcon = status.icon
  const currentPhaseIndex = getPhaseIndex(job.currentPhase)

  // Calculate phase progress
  const compositionProgress = job.status === 'pending' ? 0 :
    ['composed', 'running', 'paused', 'completed', 'terminated', 'failed'].includes(job.status) ? 100 :
    job.status === 'composing' ? Math.min(job.progress, 99) : 0

  // Separate progress for rerun dialog - shows 100% when composed, actual progress when composing
  const rerunDialogProgress = job.status === 'composed' ? 100 :
    job.status === 'composing' ? Math.min(job.progress, 99) : 0

  // Actual generation progress based on generatedCount (not job.progress)
  const actualGenerationPercent = targetCount > 0
    ? Math.round((generatedCount / targetCount) * 100)
    : 0

  // Progress bar shows 100% when job is done (success, failed, or terminated)
  // This reflects "processing complete" not "all succeeded"
  const generationProgress = ['pending', 'composing', 'composed'].includes(job.status) ? 0 :
    ['completed', 'failed', 'terminated'].includes(job.status) ? 100 :
    actualGenerationPercent

  const evaluationProgress = job.status === 'completed' ? 100 : 0

  // Calculate phase statuses for tab indicators
  const getPhaseStatus = (phaseId: string): 'pending' | 'ready' | 'in_progress' | 'done' => {
    if (phaseId === 'composition') {
      if (job.status === 'pending') return 'ready'
      if (job.status === 'composing') return 'in_progress'
      return 'done' // composed, running, paused, completed, terminated, failed
    }
    if (phaseId === 'generation') {
      // If composition isn't in this job's phases, generation is ready when pending
      const hasComposition = jobPhases.includes('composition')
      if (!hasComposition) {
        if (job.status === 'pending' || job.status === 'composed') return 'ready'
      } else {
        if (['pending', 'composing'].includes(job.status)) return 'pending'
        if (job.status === 'composed') return 'ready'
      }
      if (job.status === 'running' || job.status === 'paused') return 'in_progress'
      return 'done' // completed, terminated, failed
    }
    if (phaseId === 'evaluation') {
      if (job.status === 'completed') return 'done'
      if (job.status === 'evaluating') return 'in_progress'
      // If only evaluation phase, it's ready when pending/composed
      const hasGeneration = jobPhases.includes('generation')
      if (!hasGeneration && (job.status === 'pending' || job.status === 'composed')) return 'ready'
      return 'pending'
    }
    return 'pending'
  }

  // Format elapsed time
  const formatElapsedTime = () => {
    if (!job.createdAt) return '0s'
    const end = job.completedAt || Date.now()
    const elapsed = Math.floor((end - job.createdAt) / 1000)
    const minutes = Math.floor(elapsed / 60)
    const seconds = elapsed % 60
    return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Scrollable content area - pb-[200px] accounts for fixed log panel */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="flex flex-col gap-6 p-6 pb-[220px]">
          {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link to="/jobs">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold tracking-tight">{job.name}</h1>
              <Badge className={status.color}>
                <StatusIcon className={`mr-1 h-3 w-3 ${job.status === 'running' || job.status === 'composing' || job.status === 'evaluating' ? 'animate-spin' : ''}`} />
                {status.label}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Created {new Date(job.createdAt).toLocaleString()}
            </p>
          </div>
        </div>

        {/* Job management buttons */}
        <div className="flex items-center gap-2">
          {job.status === 'completed' && dataset && (
            <Button asChild style={{ backgroundColor: PHASES[2].strongColor }}>
              <Link to="/results/$id" params={{ id: dataset._id }}>
                <Download className="mr-2 h-4 w-4" />
                View Results
              </Link>
            </Button>
          )}
          {(job.status === 'pending' || job.status === 'composed') && (
            <Button onClick={handleRunPipeline} className="bg-gradient-to-r from-[#4e95d9] via-[#f2aa84] to-[#8ed973] text-white hover:opacity-90">
              <Play className="mr-2 h-4 w-4" />
              Run Pipeline
            </Button>
          )}
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="text-destructive hover:text-destructive disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={['composing', 'running', 'evaluating'].includes(job.status)}
                title={['composing', 'running', 'evaluating'].includes(job.status) ? 'Cannot delete while pipeline is running' : 'Delete job'}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete Job?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete this job and all associated data. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDelete}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* Target selector - only for multi-target jobs */}
      {isMultiTarget && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Target:</span>
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTarget('all')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeTarget === 'all'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted'
              }`}
            >
              All
            </button>
            {targets.map((target, idx) => {
              const label = `${target.target_value} ${target.count_mode === 'sentences' ? 'sent' : 'rev'}`
              const tp = (job as any).targetProgress?.find((tp: { targetIndex: number }) => tp.targetIndex === idx)
              return (
                <button
                  key={idx}
                  onClick={() => setActiveTarget(idx)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    activeTarget === idx
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                  }`}
                >
                  {label}
                  {tp && (
                    <span className="ml-1.5 opacity-70">
                      {tp.status === 'completed' ? ' \u2713' :
                       tp.status === 'generating' || tp.status === 'evaluating' ? ` ${tp.progress}%` : ''}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Phase tabs with context-aware action buttons */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {visiblePhases.map((phase) => (
            <PhaseTab
              key={phase.id}
              phase={phase}
              progress={phase.id === 'composition' ? compositionProgress : phase.id === 'generation' ? generationProgress : evaluationProgress}
              isActive={activeTab === phase.id}
              onClick={() => setActiveTab(phase.id)}
              status={getPhaseStatus(phase.id)}
            />
          ))}
        </div>

        {/* Context-aware action buttons */}
        <div className="flex items-center gap-2">
          {/* COMPOSITION tab actions */}
          {activeTab === 'composition' && (
            <>
              {job.status === 'pending' && (
                <Button onClick={handleStartComposition} style={{ backgroundColor: PHASES[0].strongColor }}>
                  <Play className="mr-2 h-4 w-4" />
                  Start Composition
                </Button>
              )}
              {job.status === 'composing' && (
                <>
                  <Button disabled className="bg-gradient-to-r from-[#4e95d9] via-[#f2aa84] to-[#8ed973] text-white">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Composing...
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="destructive">
                        <Square className="mr-2 h-4 w-4" />
                        Terminate
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Terminate Composition?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will stop the composition process. Any partially gathered intelligence will be lost.
                          You can restart composition from the beginning later.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={handleTerminate}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Terminate
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
              {['composed', 'running', 'paused', 'completed', 'terminated', 'failed'].includes(job.status) && (
                <>
                  <Dialog open={contextsDialogOpen} onOpenChange={setContextsDialogOpen}>
                    <DialogTrigger asChild>
                      <Button variant="outline" style={{ borderColor: PHASES[0].strongColor, color: PHASES[0].strongColor }}>
                        <Eye className="mr-2 h-4 w-4" />
                        View Contexts
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-3xl max-h-[80vh]">
                      <DialogHeader>
                        <DialogTitle>Composition Contexts</DialogTitle>
                        <DialogDescription>
                          The contexts generated during the composition phase
                        </DialogDescription>
                      </DialogHeader>
                      <Tabs defaultValue="subject" className="w-full">
                        <TabsList className="grid w-full grid-cols-3">
                          <TabsTrigger value="subject">Subject Context</TabsTrigger>
                          <TabsTrigger value="reviewer">Reviewer Context</TabsTrigger>
                          <TabsTrigger value="attributes">Attributes Context</TabsTrigger>
                        </TabsList>
                        <TabsContent value="subject" className="mt-4">
                          <ScrollArea className="h-[400px] rounded-md border p-4">
                            {job.subjectContext ? (
                              <pre className="text-sm whitespace-pre-wrap font-mono">{JSON.stringify(job.subjectContext, null, 2)}</pre>
                            ) : (
                              <p className="text-muted-foreground text-center py-8">Subject context not yet generated</p>
                            )}
                          </ScrollArea>
                        </TabsContent>
                        <TabsContent value="reviewer" className="mt-4">
                          <ScrollArea className="h-[400px] rounded-md border p-4">
                            {job.reviewerContext ? (
                              <pre className="text-sm whitespace-pre-wrap font-mono">{JSON.stringify(
                                // Filter out age_range when age is disabled in ablation
                                job.config.ablation?.age_enabled === false && typeof job.reviewerContext === 'object'
                                  ? (() => {
                                      const filtered = { ...job.reviewerContext } as Record<string, unknown>
                                      delete filtered.age_range
                                      return filtered
                                    })()
                                  : job.reviewerContext,
                                null, 2
                              )}</pre>
                            ) : (
                              <p className="text-muted-foreground text-center py-8">Reviewer context not yet generated</p>
                            )}
                          </ScrollArea>
                        </TabsContent>
                        <TabsContent value="attributes" className="mt-4">
                          <ScrollArea className="h-[400px] rounded-md border p-4">
                            {job.attributesContext ? (
                              <pre className="text-sm whitespace-pre-wrap font-mono">{JSON.stringify(job.attributesContext, null, 2)}</pre>
                            ) : (
                              <p className="text-muted-foreground text-center py-8">Attributes context not yet generated</p>
                            )}
                          </ScrollArea>
                        </TabsContent>
                      </Tabs>
                    </DialogContent>
                  </Dialog>

                  {/* Rerun Composition button and dialog */}
                  <Button
                    variant="outline"
                    onClick={handleRerunComposition}
                    disabled={job.status === 'composing' || isRerunningComposition}
                    style={{ borderColor: PHASES[0].strongColor, color: PHASES[0].strongColor }}
                  >
                    <RefreshCcw className={`mr-2 h-4 w-4 ${isRerunningComposition ? 'animate-spin' : ''}`} />
                    Rerun Composition
                  </Button>

                  {/* Rerun Composition Progress Dialog */}
                  <Dialog open={rerunCompositionDialogOpen} onOpenChange={(open) => {
                    // Only allow closing if not in progress
                    if (!isRerunningComposition && job.status !== 'composing') {
                      setRerunCompositionDialogOpen(open)
                    }
                  }}>
                    <DialogContent className="sm:max-w-md">
                      <DialogHeader>
                        <DialogTitle>Rerunning Composition</DialogTitle>
                        <DialogDescription>
                          Regenerating subject, reviewer, and attributes contexts
                        </DialogDescription>
                      </DialogHeader>
                      <div className="space-y-6 py-4">
                        {/* Progress bar */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Progress</span>
                            <span className="font-medium">{rerunDialogProgress}%</span>
                          </div>
                          <Progress value={rerunDialogProgress} className="h-2" />
                        </div>

                        {/* Phase indicators */}
                        {(() => {
                          // Check if MAV is enabled with 3 models
                          const mavEnabled = job?.config?.subject_profile?.mav?.enabled &&
                            (job?.config?.subject_profile?.mav?.models?.length === 3)

                          // Define phases with progress ranges
                          const phases = mavEnabled ? [
                            { id: 'sil', name: 'SIL - Subject Intelligence Layer', description: 'Understanding the subject', progressRange: [0, 10] },
                            { id: 'mav', name: 'MAV - Multi-Agent Verification', description: 'Cross-validating facts with 3 models', progressRange: [10, 35], isSubstep: true },
                            { id: 'rgm', name: 'RGM - Reviewer Generation Module', description: 'Creating reviewer personas', progressRange: [35, 65] },
                            { id: 'acm', name: 'ACM - Attributes Composition Module', description: 'Configuring review attributes', progressRange: [65, 95] },
                          ] : [
                            { id: 'sil', name: 'SIL - Subject Intelligence Layer', description: 'Gathering factual information', progressRange: [0, 35] },
                            { id: 'rgm', name: 'RGM - Reviewer Generation Module', description: 'Creating reviewer personas', progressRange: [35, 65] },
                            { id: 'acm', name: 'ACM - Attributes Composition Module', description: 'Configuring review attributes', progressRange: [65, 95] },
                          ]

                          const getPhaseStatus = (phase: typeof phases[0]) => {
                            const currentPhaseLower = job.currentPhase?.toLowerCase() || ''
                            const isActive = job.status === 'composing' && currentPhaseLower.includes(phase.id)
                            const isComplete = rerunDialogProgress >= phase.progressRange[1]
                            return { isActive, isComplete }
                          }

                          return (
                            <div className="space-y-3">
                              {phases.map((phase) => {
                                const { isActive, isComplete } = getPhaseStatus(phase)
                                return (
                                  <div
                                    key={phase.id}
                                    className={`flex items-center gap-3 ${phase.isSubstep ? 'ml-6' : ''}`}
                                  >
                                    {isActive ? (
                                      <Loader2 className="h-4 w-4 animate-spin" style={{ color: PHASES[0].strongColor }} />
                                    ) : isComplete ? (
                                      <CheckCircle2 className="h-4 w-4" style={{ color: PHASES[0].strongColor }} />
                                    ) : (
                                      <div className="h-4 w-4 rounded-full border-2" style={{ borderColor: PHASES[0].strongColor }} />
                                    )}
                                    <div>
                                      <p className={`text-sm font-medium ${phase.isSubstep ? 'text-muted-foreground' : ''}`}>
                                        {phase.name}
                                      </p>
                                      <p className="text-xs text-muted-foreground">{phase.description}</p>
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          )
                        })()}

                        {/* Current status */}
                        {job.currentPhase && job.status === 'composing' && (
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Activity className="h-4 w-4" />
                            <span>{job.currentPhase}</span>
                          </div>
                        )}

                        {/* Completion state */}
                        {job.status === 'composed' && (
                          <div className="flex items-center gap-2 text-sm" style={{ color: PHASES[0].strongColor }}>
                            <CheckCircle2 className="h-4 w-4" />
                            <span>Composition complete! You can now close this dialog.</span>
                          </div>
                        )}

                        {/* Error state */}
                        {job.status === 'failed' && job.error && (
                          <div className="flex items-center gap-2 text-sm text-destructive">
                            <XCircle className="h-4 w-4" />
                            <span>{job.error}</span>
                          </div>
                        )}
                      </div>
                    </DialogContent>
                  </Dialog>
                </>
              )}
            </>
          )}

          {/* GENERATION tab actions */}
          {activeTab === 'generation' && (
            <>
              {job.status === 'composed' && (
                <Button onClick={handleStartGeneration} style={{ backgroundColor: PHASES[1].strongColor }}>
                  <Play className="mr-2 h-4 w-4" />
                  Start Generation
                </Button>
              )}
              {job.status === 'running' && (
                <>
                  <Button variant="outline" onClick={handlePause}>
                    <Pause className="mr-2 h-4 w-4" />
                    Pause
                  </Button>
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
                          This will stop the generation process and keep all reviews generated so far.
                          You can rerun the job later, which will overwrite any existing reviews.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={handleTerminate}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Terminate
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
              {job.status === 'paused' && (
                <>
                  <Button onClick={handleResume} style={{ backgroundColor: PHASES[1].strongColor }}>
                    <Play className="mr-2 h-4 w-4" />
                    Resume
                  </Button>
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
                          This will stop the generation process and keep all reviews generated so far.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={handleTerminate}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Terminate
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
              {(job.status === 'terminated' || job.status === 'failed' || job.status === 'completed') && (
                <Button variant="outline" onClick={handleRerunGeneration} style={{ borderColor: PHASES[1].strongColor, color: PHASES[1].strongColor }}>
                  <RefreshCcw className="mr-2 h-4 w-4" />
                  Rerun Generation
                </Button>
              )}
            </>
          )}

          {/* EVALUATION tab actions */}
          {activeTab === 'evaluation' && (
            <>
              {job.status === 'evaluating' && (
                <>
                  <Button disabled className="bg-gradient-to-r from-[#8ed973] to-[#6bc050] text-white">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Evaluating...
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="destructive">
                        <Square className="mr-2 h-4 w-4" />
                        Terminate
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Terminate Evaluation?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will stop the evaluation process. Partial metrics may not be saved.
                          You can rerun evaluation later.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={handleTerminate}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Terminate
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </>
              )}
              {job.status === 'completed' && dataset && (
                <Button asChild style={{ backgroundColor: PHASES[2].strongColor }}>
                  <Link to="/results/$id" params={{ id: dataset._id }}>
                    <Download className="mr-2 h-4 w-4" />
                    Export Dataset
                  </Link>
                </Button>
              )}
              {(job.status === 'completed' || job.status === 'failed' || job.status === 'terminated') && (
                <Button
                  variant="outline"
                  onClick={handleRerunEvaluation}
                  style={{ borderColor: PHASES[2].strongColor, color: PHASES[2].strongColor }}
                >
                  <RefreshCcw className="mr-2 h-4 w-4" />
                  Rerun Evaluation
                </Button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Phase content */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Main content area */}
        <div className="lg:col-span-3 space-y-4">
          {/* COMPOSITION Tab */}
          {activeTab === 'composition' && job.config.subject_profile && (
            <div className="space-y-4">
              {/* Composition Progress */}
              <CompositionProgress job={job} />

              {/* Subject Profile */}
              <Collapsible
                open={compositionSections.subject}
                onOpenChange={() => setCompositionSections(prev => ({ ...prev, subject: !prev.subject }))}
              >
                <div
                  className="rounded-lg border overflow-hidden"
                  style={{ borderColor: compositionSections.subject ? PHASES[0].strongColor : undefined }}
                >
                  <CollapsibleTrigger asChild>
                    <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[0].lightColor }}>
                          <Search className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                        </div>
                        <div className="text-left">
                          <h3 className="font-semibold">Subject Profile</h3>
                          <p className="text-sm text-muted-foreground">{job.config.subject_profile?.query}</p>
                        </div>
                      </div>
                      <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${compositionSections.subject ? 'rotate-180' : ''}`} />
                    </button>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="border-t p-4 space-y-3 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Query</span>
                        <span className="font-medium">{job.config.subject_profile?.query}</span>
                      </div>
                      <Separator />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Domain</span>
                        <span className="font-medium">{job.config.subject_profile?.domain || job.config.subject_profile?.category}</span>
                      </div>
                      <Separator />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Region</span>
                        <span className="font-medium">{job.config.subject_profile?.region}</span>
                      </div>
                      <Separator />
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">SIL (Web Search Grounding)</span>
                        <span className="font-medium">
                          {job.config.subject_profile?.mav?.enabled ? (
                            <Badge variant="outline" className="bg-green-500/10 text-green-600">Enabled</Badge>
                          ) : (
                            <Badge variant="outline" className="text-muted-foreground">Disabled</Badge>
                          )}
                        </span>
                      </div>
                      {job.config.subject_profile?.mav && (
                        <>
                          <Separator />
                          <div className="space-y-3">
                            <span className="text-muted-foreground">MAV Models ({job.config.subject_profile.mav.models.length})</span>
                            <div className="grid grid-cols-3 gap-2">
                              {job.config.subject_profile.mav.models.map((modelId: string, i: number) => {
                                const modelInfo = getModelInfo(modelId)
                                const [provider, ...nameParts] = modelId.split('/')
                                const modelName = nameParts.join('/')
                                // Clean model name: remove provider prefix and "(free)" suffix
                                const cleanName = (modelInfo?.name || modelName)
                                  .replace(/\s*\(free\)\s*$/i, '')
                                  .replace(/^[^:]+:\s*/, '') // Remove "Provider: " prefix
                                return (
                                  <div
                                    key={i}
                                    className="flex flex-col gap-1 p-2 rounded-lg border bg-muted/30"
                                    title={modelId}
                                  >
                                    <div className="flex items-center gap-1.5">
                                      <Bot className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                                      <span className="text-[10px] text-muted-foreground capitalize truncate">{provider}</span>
                                      {modelInfo?.isFree && (
                                        <Badge variant="secondary" className="bg-green-500/20 text-green-600 text-[9px] px-1 py-0 shrink-0">Free</Badge>
                                      )}
                                      {modelInfo?.isOpenSource && (
                                        <Badge variant="outline" className="text-[9px] px-1 py-0 shrink-0">OSS</Badge>
                                      )}
                                    </div>
                                    <p className="font-medium text-xs truncate">{cleanName}</p>
                                    {modelInfo && (
                                      <span className="text-[10px] text-muted-foreground">
                                        {modelInfo.contextLength.toLocaleString()} ctx
                                      </span>
                                    )}
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  </CollapsibleContent>
                </div>
              </Collapsible>

              {/* Reviewer Profile */}
              {job.config.reviewer_profile && (
                <Collapsible
                  open={compositionSections.reviewer}
                  onOpenChange={() => setCompositionSections(prev => ({ ...prev, reviewer: !prev.reviewer }))}
                >
                  <div
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: compositionSections.reviewer ? PHASES[0].strongColor : undefined }}
                  >
                    <CollapsibleTrigger asChild>
                      <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[0].lightColor }}>
                            <Users className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                          </div>
                          <div className="text-left">
                            <h3 className="font-semibold">Reviewer Profile</h3>
                            <p className="text-sm text-muted-foreground">
                              {job.config.ablation?.age_enabled !== false && job.config.reviewer_profile?.age_range
                                ? `Age ${job.config.reviewer_profile.age_range[0]}-${job.config.reviewer_profile.age_range[1]}`
                                : 'Demographics settings'}
                            </p>
                          </div>
                        </div>
                        <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${compositionSections.reviewer ? 'rotate-180' : ''}`} />
                      </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="border-t p-4 space-y-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Age Range</span>
                          <span className={`font-medium ${job.config.ablation?.age_enabled === false || !job.config.reviewer_profile?.age_range ? 'text-muted-foreground italic' : ''}`}>
                            {job.config.ablation?.age_enabled === false
                              ? 'Disabled (ablation)'
                              : job.config.reviewer_profile?.age_range
                                ? `${job.config.reviewer_profile.age_range[0]}-${job.config.reviewer_profile.age_range[1]}`
                                : 'Not set'}
                          </span>
                        </div>
                        <Separator />
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Sex Distribution</span>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="bg-blue-500/10 text-blue-600 border-blue-200">M: {Math.round((job.config.reviewer_profile?.sex_distribution?.male ?? 0) * 100)}%</Badge>
                            <Badge variant="outline" className="bg-pink-500/10 text-pink-600 border-pink-200">F: {Math.round((job.config.reviewer_profile?.sex_distribution?.female ?? 0) * 100)}%</Badge>
                            <Badge variant="outline">U: {Math.round((job.config.reviewer_profile?.sex_distribution?.unspecified ?? 0) * 100)}%</Badge>
                          </div>
                        </div>
                        {job.config.reviewer_profile?.additional_context && (
                          <>
                            <Separator />
                            <div className="flex justify-between items-start gap-4">
                              <span className="text-muted-foreground shrink-0">Additional Context</span>
                              <span className="text-sm text-right">{job.config.reviewer_profile.additional_context}</span>
                            </div>
                          </>
                        )}
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}

              {/* Attributes Profile */}
              {job.config.attributes_profile && (
                <Collapsible
                  open={compositionSections.attributes}
                  onOpenChange={() => setCompositionSections(prev => ({ ...prev, attributes: !prev.attributes }))}
                >
                  <div
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: compositionSections.attributes ? PHASES[0].strongColor : undefined }}
                  >
                    <CollapsibleTrigger asChild>
                      <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[0].lightColor }}>
                            <Sliders className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                          </div>
                          <div className="text-left">
                            <h3 className="font-semibold">Attributes Profile</h3>
                            <p className="text-sm text-muted-foreground">
                              {Math.round((job.config.attributes_profile?.polarity?.positive ?? 0) * 100)}% positive
                            </p>
                          </div>
                        </div>
                        <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${compositionSections.attributes ? 'rotate-180' : ''}`} />
                      </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="border-t p-4 space-y-3 text-sm">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Polarity Distribution</span>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="bg-green-500/10 text-green-500">
                              +{Math.round((job.config.attributes_profile?.polarity?.positive ?? 0) * 100)}%
                            </Badge>
                            <Badge variant="outline" className="bg-gray-500/10">
                              ~{Math.round((job.config.attributes_profile?.polarity?.neutral ?? 0) * 100)}%
                            </Badge>
                            <Badge variant="outline" className="bg-red-500/10 text-red-500">
                              -{Math.round((job.config.attributes_profile?.polarity?.negative ?? 0) * 100)}%
                            </Badge>
                          </div>
                        </div>
                        <Separator />
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Length Range</span>
                          <span className="font-medium">
                            {job.config.attributes_profile?.length_range?.[0]}-{job.config.attributes_profile?.length_range?.[1]} sentences
                          </span>
                        </div>
                        <Separator />
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Noise Preset</span>
                          <span className="font-medium capitalize">
                            {(() => {
                              const preset = job.config.attributes_profile?.noise?.preset || 'moderate'
                              const typoRates: Record<string, string> = {
                                none: '0%',
                                light: '0.5%',
                                moderate: '1%',
                                heavy: '3%'
                              }
                              return `${preset} (${typoRates[preset] || '1%'} typo rate)`
                            })()}
                          </span>
                        </div>
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}

              {/* MAV Report */}
              {job.status !== 'pending' && (
                <Collapsible
                  open={compositionSections.mav}
                  onOpenChange={() => setCompositionSections(prev => ({ ...prev, mav: !prev.mav }))}
                >
                  <div
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: compositionSections.mav ? PHASES[0].strongColor : undefined }}
                  >
                    <CollapsibleTrigger asChild>
                      <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[0].lightColor }}>
                            <Network className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                          </div>
                          <div className="text-left">
                            <h3 className="font-semibold">MAV Report</h3>
                            <p className="text-sm text-muted-foreground">Multi-agent verification</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 text-xs"
                            style={{ borderColor: PHASES[0].strongColor, color: PHASES[0].strongColor }}
                            onClick={(e) => { e.stopPropagation(); handleExportReport('mav') }}
                          >
                            <Download className="mr-1 h-3 w-3" />
                            Export
                          </Button>
                          <CheckCircle className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                          <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${compositionSections.mav ? 'rotate-180' : ''}`} />
                        </div>
                      </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="border-t p-4 space-y-3">
                        {(() => {
                          const mavStats = (job.subjectContext as any)?.mav_stats
                          const mavConfig = job.config?.subject_profile?.mav
                          const factsExtracted = mavStats?.total_facts_extracted ?? 0
                          const factsVerified = mavStats?.facts_verified ?? 0
                          const verificationRate = mavStats?.verification_rate ?? 0
                          const modelsUsed = mavConfig?.models?.length ?? 0
                          const threshold = mavConfig?.similarity_threshold ?? 0.85
                          return (
                            <>
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Facts Extracted</span>
                                <span className="font-medium">{factsExtracted}</span>
                              </div>
                              <Separator />
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Facts Verified</span>
                                <span className="font-medium text-green-600">
                                  {factsVerified} ({Math.round(verificationRate * 100)}%)
                                </span>
                              </div>
                              <Separator />
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Models Used</span>
                                <span className="font-medium">{modelsUsed}</span>
                              </div>
                              <Separator />
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Similarity Threshold</span>
                                <span className="font-medium">{threshold}</span>
                              </div>
                            </>
                          )
                        })()}
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}
            </div>
          )}

          {/* GENERATION Tab */}
          {activeTab === 'generation' && (
            <div className="space-y-4">
              {/* Multi-target: All overview with target cards */}
              {isMultiTarget && activeTarget === 'all' ? (
                <div className="space-y-4">
                  <div
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: PHASES[1].strongColor }}
                  >
                    <div
                      className="flex items-center justify-between p-4 border-b"
                      style={{ backgroundColor: PHASES[1].bgRgba }}
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[1].lightColor }}>
                          <Activity className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                        </div>
                        <div>
                          <h3 className="font-semibold text-foreground">Multi-Target Overview</h3>
                          <p className="text-sm text-muted-foreground">{targets.length} target datasets</p>
                        </div>
                      </div>
                    </div>
                    <div className="p-4">
                      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                        {targets.map((target, idx) => {
                          const tp = (job as any).targetProgress?.find((tp: { targetIndex: number }) => tp.targetIndex === idx)
                          const statusColors: Record<string, string> = {
                            pending: 'text-muted-foreground',
                            generating: 'text-orange-500',
                            evaluating: 'text-green-500',
                            completed: 'text-green-600',
                            failed: 'text-destructive',
                          }
                          return (
                            <button
                              key={idx}
                              onClick={() => setActiveTarget(idx)}
                              className="rounded-lg border p-4 text-left hover:border-[#f2aa84] transition-colors"
                            >
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-semibold text-lg">{target.target_value}</span>
                                <Badge
                                  variant="outline"
                                  className={statusColors[tp?.status || 'pending'] || 'text-muted-foreground'}
                                >
                                  {tp?.status || 'pending'}
                                </Badge>
                              </div>
                              <span className="text-xs text-muted-foreground">
                                {target.count_mode === 'sentences' ? 'sentences' : 'reviews'}
                                {target.total_runs > 1 ? ` · ${target.total_runs} runs` : ''}
                              </span>
                              <Progress
                                value={tp?.progress || 0}
                                className="h-1.5 mt-2"
                                indicatorColor={PHASES[1].strongColor}
                                trackColor={PHASES[1].lightColor}
                              />
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Single-target or specific target selected: show GenerationProgress */
                <GenerationProgress job={targetScopedJob || job} />
              )}

              {/* Legacy Collapsible - hidden, will be removed */}
              {false && <Collapsible
                open={generationSections.progress}
                onOpenChange={() => setGenerationSections(prev => ({ ...prev, progress: !prev.progress }))}
              >
                <div
                  className="rounded-lg border overflow-hidden"
                  style={{ borderColor: generationSections.progress ? PHASES[1].strongColor : undefined }}
                >
                  <CollapsibleTrigger asChild>
                    <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[1].lightColor }}>
                          {job.status === 'running' ? (
                            <Loader2 className="h-5 w-5 animate-spin" style={{ color: PHASES[1].strongColor }} />
                          ) : job.status === 'paused' ? (
                            <Pause className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                          ) : job.status === 'completed' ? (
                            <CheckCircle2 className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                          ) : job.status === 'terminated' || job.status === 'failed' ? (
                            <XCircle className="h-5 w-5 text-destructive" />
                          ) : (
                            <Activity className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                          )}
                        </div>
                        <div className="text-left">
                          <h3 className="font-semibold">Generation Progress</h3>
                          <p className="text-sm text-muted-foreground">
                            {job.status === 'pending' && 'Waiting to start'}
                            {job.status === 'composing' && 'Running composition...'}
                            {job.status === 'composed' && 'Ready to start'}
                            {job.status === 'running' && `${job.progress}% complete`}
                            {job.status === 'paused' && `Paused at ${job.progress}%`}
                            {job.status === 'completed' && 'Completed'}
                            {job.status === 'terminated' && `Terminated at ${job.progress}%`}
                            {job.status === 'failed' && 'Failed'}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {job.status !== 'pending' && (
                          <Badge
                            className={status.color}
                          >
                            {status.label}
                          </Badge>
                        )}
                        <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${generationSections.progress ? 'rotate-180' : ''}`} />
                      </div>
                    </button>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="border-t p-4 space-y-4">
                      {/* Progress Bar */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Progress</span>
                          <span className="font-medium">{job.progress}%</span>
                        </div>
                        <div className="h-3 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${job.progress}%`,
                              backgroundColor: job.status === 'terminated' || job.status === 'failed' ? 'hsl(var(--destructive))' :
                                              PHASES[1].strongColor,
                            }}
                          />
                        </div>
                      </div>

                      {/* Stats Grid */}
                      <div className="grid gap-3 sm:grid-cols-5">
                        <div className="rounded-lg border bg-muted/30 p-3">
                          <div className="flex items-center gap-2 text-muted-foreground mb-1">
                            <CheckCircle2 className="h-4 w-4 text-blue-500" />
                            <span className="text-xs">Target</span>
                          </div>
                          <p className="text-lg font-semibold">{targetCount.toLocaleString()}</p>
                        </div>
                        <div className="rounded-lg border bg-muted/30 p-3">
                          <div className="flex items-center gap-2 text-muted-foreground mb-1">
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                            <span className="text-xs">Generated</span>
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
                          <p className="text-sm font-semibold truncate" title={job.config.generation?.model}>
                            {job.config.generation?.model?.split('/').pop()}
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

                      {/* Summary when completed/terminated */}
                      {(job.status === 'completed' || job.status === 'terminated' || job.status === 'failed') && (
                        <div className="rounded-lg border p-4 space-y-3">
                          <div className="flex items-center gap-2">
                            <div className={`p-1.5 rounded-full ${job.status === 'completed' ? 'bg-green-500/20' : 'bg-destructive/20'}`}>
                              {job.status === 'completed' ? (
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                              ) : (
                                <XCircle className="h-4 w-4 text-destructive" />
                              )}
                            </div>
                            <h4 className="font-semibold">
                              Generation {job.status === 'completed' ? 'Completed' : job.status === 'terminated' ? 'Terminated' : 'Failed'}
                            </h4>
                          </div>
                          <Separator />
                          <div className="grid gap-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Status:</span>
                              <Badge className={status.color}>{status.label.toUpperCase()}</Badge>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Reviews Requested:</span>
                              <span className="font-medium">{targetCount.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Reviews Generated:</span>
                              <span className="font-medium text-green-600">
                                {generatedCount.toLocaleString()}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Reviews Failed:</span>
                              <span className={`font-medium ${failedCount > 0 ? 'text-red-500' : 'text-muted-foreground'}`}>
                                {failedCount.toLocaleString()}
                              </span>
                            </div>
                            {job.error && (
                              <>
                                <Separator />
                                <div className="text-destructive">
                                  <span className="font-medium">Error:</span> {job.error}
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </CollapsibleContent>
                </div>
              </Collapsible>}

              {/* Conformity Report */}
              {(job.status === 'completed' || job.status === 'terminated' || job.status === 'failed') && (
                <Collapsible
                  open={generationSections.conformity}
                  onOpenChange={() => setGenerationSections(prev => ({ ...prev, conformity: !prev.conformity }))}
                >
                  <div
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: generationSections.conformity ? PHASES[1].strongColor : undefined }}
                  >
                    <CollapsibleTrigger asChild>
                      <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[1].lightColor }}>
                            <ClipboardCheck className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                          </div>
                          <div className="text-left">
                            <h3 className="font-semibold">Conformity Report</h3>
                            <p className="text-sm text-muted-foreground">Distribution analysis</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 text-xs"
                            style={{ borderColor: PHASES[1].strongColor, color: PHASES[1].strongColor }}
                            onClick={(e) => { e.stopPropagation(); handleExportReport('conformity') }}
                          >
                            <Download className="mr-1 h-3 w-3" />
                            Export
                          </Button>
                          <CheckCircle className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                          <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${generationSections.conformity ? 'rotate-180' : ''}`} />
                        </div>
                      </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="border-t p-4 space-y-3">
                        {(() => {
                          const report = ((targetScopedJob || job) as any).conformityReport
                          const polarity = report ? Math.round(report.polarity * 100) : null
                          const length = report ? Math.round(report.length * 100) : null
                          const noise = report ? Math.round(report.noise * 100) : null
                          const validation = report?.validation != null ? Math.round(report.validation * 100) : null
                          if (!report) {
                            return (
                              <p className="text-xs text-muted-foreground text-center py-2">
                                Conformity data will be available after generation completes.
                              </p>
                            )
                          }
                          return (
                            <>
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Polarity Conformity</span>
                                <span className="font-medium" style={{ color: PHASES[1].strongColor }}>{polarity}%</span>
                              </div>
                              <div className="h-2 rounded-full bg-muted overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${polarity}%`, backgroundColor: PHASES[1].strongColor }} />
                              </div>
                              <Separator />
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Length Conformity</span>
                                <span className="font-medium" style={{ color: PHASES[1].strongColor }}>{length}%</span>
                              </div>
                              <div className="h-2 rounded-full bg-muted overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${length}%`, backgroundColor: PHASES[1].strongColor }} />
                              </div>
                              <Separator />
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">Noise Conformity</span>
                                <span className="font-medium" style={{ color: PHASES[1].strongColor }}>{noise}%</span>
                              </div>
                              <div className="h-2 rounded-full bg-muted overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${noise}%`, backgroundColor: PHASES[1].strongColor }} />
                              </div>
                              {validation != null && (
                                <>
                                  <Separator />
                                  <div className="flex justify-between text-sm">
                                    <span className="text-muted-foreground">JSON Validation</span>
                                    <span className="font-medium" style={{ color: validation === 100 ? '#22c55e' : validation >= 95 ? PHASES[1].strongColor : '#f59e0b' }}>{validation}%</span>
                                  </div>
                                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                                    <div className="h-full rounded-full" style={{ width: `${validation}%`, backgroundColor: validation === 100 ? '#22c55e' : validation >= 95 ? PHASES[1].strongColor : '#f59e0b' }} />
                                  </div>
                                  {validation < 100 && (
                                    <p className="text-xs text-muted-foreground">
                                      {100 - validation}% of reviews fell back to plain text due to JSON validation failures
                                    </p>
                                  )}
                                </>
                              )}
                            </>
                          )
                        })()}
                      </div>
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              )}
            </div>
          )}

          {/* EVALUATION Tab */}
          {activeTab === 'evaluation' && (
            <div className="space-y-4">
              {/* Multi-target: All overview - prompt to select a target */}
              {isMultiTarget && activeTarget === 'all' ? (
                <div className="space-y-4">
                  <EvaluationProgress job={job} />
                  <div className="text-center py-12">
                    <div
                      className="mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4"
                      style={{ backgroundColor: PHASES[2].lightColor }}
                    >
                      <BarChart3 className="h-8 w-8" style={{ color: PHASES[2].strongColor }} />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">Per-Target Evaluation</h3>
                    <p className="text-muted-foreground">
                      Select a target dataset above to view its MDQA evaluation metrics and cross-run results.
                    </p>
                  </div>
                </div>
              ) : (
                <>
              {/* Single-target or specific target selected: show full evaluation */}
              <EvaluationProgress job={targetScopedJob || job} />

              {(() => {
                const scopedJob = targetScopedJob || job
                const scopedStatus = scopedJob.status
                const hasMetrics = !!(scopedJob as any).evaluationMetrics || (scopedJob as any).perModelMetrics?.length > 0
                return scopedStatus !== 'completed' && scopedStatus !== 'evaluating' && !hasMetrics
              })() ? (
                <div className="text-center py-12">
                  <div
                    className="mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4"
                    style={{ backgroundColor: PHASES[2].lightColor }}
                  >
                    <BarChart3 className="h-8 w-8" style={{ color: PHASES[2].strongColor }} />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Evaluation Pending</h3>
                  <p className="text-muted-foreground">
                    Evaluation metrics will be available after generation completes.
                  </p>
                </div>
              ) : (
                <>
                  {/* MDQA Metrics */}
                  <Collapsible
                    open={evaluationSections.mdqa}
                    onOpenChange={() => setEvaluationSections(prev => ({ ...prev, mdqa: !prev.mdqa }))}
                  >
                    <div
                      className="rounded-lg border overflow-hidden"
                      style={{ borderColor: evaluationSections.mdqa ? PHASES[2].strongColor : undefined }}
                    >
                      <CollapsibleTrigger asChild>
                        <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
                          <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg" style={{ backgroundColor: PHASES[2].lightColor }}>
                              <BarChart3 className="h-5 w-5" style={{ color: PHASES[2].strongColor }} />
                            </div>
                            <div className="text-left">
                              <h3 className="font-semibold">MDQA Metrics</h3>
                              <p className="text-sm text-muted-foreground">
                                Lexical, Semantic, Diversity
                                {(job.referenceDataset?.useForEvaluation || job.evaluationConfig?.reference_file) && (
                                  <span className="ml-1 text-green-600 dark:text-green-400">• Reference dataset used</span>
                                )}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 text-xs"
                              style={{ borderColor: PHASES[2].strongColor, color: PHASES[2].strongColor }}
                              onClick={(e) => { e.stopPropagation(); handleExportReport('metrics') }}
                            >
                              <Download className="mr-1 h-3 w-3" />
                              Export
                            </Button>
                            <CheckCircle className="h-5 w-5" style={{ color: PHASES[2].strongColor }} />
                            <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${evaluationSections.mdqa ? 'rotate-180' : ''}`} />
                          </div>
                        </button>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="border-t p-4 space-y-4">
                          {(() => {
                            // Use targetScopedJob for per-target metrics when a specific target is selected
                            const metricsSource = targetScopedJob || job
                            const metrics = (metricsSource as any).evaluationMetrics
                            const perRunMetrics = (metricsSource as any).perRunMetrics as Array<{ run: number; datasetFile: string; metrics: Record<string, number> }> | undefined
                            const averageMetrics = (metricsSource as any).averageMetrics as Record<string, { mean: number; std?: number }> | undefined
                            const totalRuns = (metricsSource as any).totalRuns || (metricsSource as any).config?.generation?.total_runs || job.totalRuns || job.config.generation?.total_runs || 1
                            const hasMultipleRuns = perRunMetrics && perRunMetrics.length > 1
                            const perModelMetrics = (metricsSource as any).perModelMetrics as Array<{
                              model: string
                              modelSlug: string
                              metrics: Record<string, number>
                              conformity?: { polarity: number; length: number; noise: number; validation?: number; temperature?: number }
                              perRunMetrics?: Array<{ run: number; datasetFile: string; metrics: Record<string, number> }>
                            }> | undefined
                            const isMultiModelMetrics = perModelMetrics && perModelMetrics.length > 1
                            const modelProgressEntries = (metricsSource as any).modelProgress as Array<{ model: string; modelSlug: string; status: string }> | undefined

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

                            const getQualityIndicator = (metricKey: string, value: number | undefined) => {
                              if (value === undefined) return null
                              const threshold = METRIC_THRESHOLDS[metricKey]
                              if (!threshold) return null
                              if (threshold.lowerIsBetter) {
                                if (value <= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-50 dark:bg-green-950/30' }
                                if (value <= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-50 dark:bg-blue-950/30' }
                                if (value <= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-950/30' }
                                return { label: 'Poor', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-950/30' }
                              } else {
                                if (value >= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-50 dark:bg-green-950/30' }
                                if (value >= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-50 dark:bg-blue-950/30' }
                                if (value >= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-950/30' }
                                return { label: 'Poor', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-950/30' }
                              }
                            }

                            const METRIC_COLS = [
                              { key: 'bleu', label: 'BLEU', higher: true },
                              { key: 'rouge_l', label: 'ROUGE-L', higher: true },
                              { key: 'bertscore', label: 'BERTScore', higher: true },
                              { key: 'moverscore', label: 'MoverScore', higher: true },
                              { key: 'distinct_1', label: 'Dist-1', higher: true },
                              { key: 'distinct_2', label: 'Dist-2', higher: true },
                              { key: 'self_bleu', label: 'Self-BLEU', higher: false },
                            ]

                            // ── Multi-Model Comparison Table ──
                            if (isMultiModelMetrics) {

                              // Find best value per metric for highlighting
                              const bestValues: Record<string, number> = {}
                              for (const col of METRIC_COLS) {
                                const values = perModelMetrics!
                                  .map(pm => pm.metrics[col.key])
                                  .filter((v): v is number => v !== undefined)
                                if (values.length > 0) {
                                  bestValues[col.key] = col.higher
                                    ? Math.max(...values)
                                    : Math.min(...values)
                                }
                              }

                              // Determine which models are still in progress
                              const modelStatusMap = new Map(
                                (modelProgressEntries || []).map(mp => [mp.model, mp.status])
                              )

                              return (
                                <div className="space-y-4">
                                  <p className="text-xs text-muted-foreground text-center">
                                    Comparison across {perModelMetrics!.length} models
                                    {totalRuns > 1 && ` (${totalRuns} runs each)`}
                                  </p>
                                  <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                      <thead>
                                        <tr className="border-b">
                                          <th className="text-left py-2 px-2 text-xs font-semibold text-muted-foreground">Model</th>
                                          {METRIC_COLS.map(col => (
                                            <th key={col.key} className="text-center py-2 px-2 text-xs font-semibold text-muted-foreground">
                                              {col.label}
                                              <span className="block text-[9px] font-normal opacity-60">
                                                {col.higher ? '↑' : '↓'}
                                              </span>
                                            </th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {perModelMetrics!.map((pm) => {
                                          const modelStatus = modelStatusMap.get(pm.model)
                                          const isIncomplete = modelStatus && modelStatus !== 'completed'
                                          return (
                                            <tr key={pm.model} className={`border-b last:border-0 ${isIncomplete ? 'opacity-50' : ''}`}>
                                              <td className="py-2.5 px-2">
                                                <div className="flex items-center gap-2">
                                                  <span className="font-medium truncate max-w-[160px]" title={pm.model}>
                                                    {pm.modelSlug}
                                                  </span>
                                                  {isIncomplete && (
                                                    <Loader2 className="h-3 w-3 animate-spin text-muted-foreground shrink-0" />
                                                  )}
                                                </div>
                                              </td>
                                              {METRIC_COLS.map(col => {
                                                const value = pm.metrics[col.key]
                                                const quality = getQualityIndicator(col.key, value)
                                                const isBest = value !== undefined && bestValues[col.key] === value && perModelMetrics!.length > 1
                                                return (
                                                  <td key={col.key} className={`text-center py-2.5 px-2 ${isBest ? 'font-bold' : ''}`}>
                                                    {isIncomplete && value === undefined ? (
                                                      <span className="text-muted-foreground">—</span>
                                                    ) : (
                                                      <div className="flex flex-col items-center">
                                                        <span style={{ color: PHASES[2].strongColor }}>
                                                          {fmt(value)}
                                                        </span>
                                                        {quality && (
                                                          <span className={`text-[9px] ${quality.color}`}>{quality.label}</span>
                                                        )}
                                                      </div>
                                                    )}
                                                  </td>
                                                )
                                              })}
                                            </tr>
                                          )
                                        })}
                                      </tbody>
                                    </table>
                                  </div>

                                  {/* Per-model expandable per-run details */}
                                  {totalRuns > 1 && perModelMetrics!.some(pm => pm.perRunMetrics && pm.perRunMetrics.length > 1) && (
                                    <div className="space-y-2 pt-2">
                                      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Per-Run Breakdown</h4>
                                      {perModelMetrics!.map((pm) => {
                                        if (!pm.perRunMetrics || pm.perRunMetrics.length <= 1) return null
                                        return (
                                          <Collapsible key={pm.model}>
                                            <CollapsibleTrigger asChild>
                                              <button className="flex items-center gap-2 w-full text-left text-sm font-medium hover:bg-muted/50 rounded px-2 py-1.5 transition-colors">
                                                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                                                {pm.modelSlug}
                                                <span className="text-xs text-muted-foreground">({pm.perRunMetrics.length} runs)</span>
                                              </button>
                                            </CollapsibleTrigger>
                                            <CollapsibleContent>
                                              <div className="pl-6 space-y-2 py-2">
                                                {pm.perRunMetrics.map((runData) => (
                                                  <div key={runData.run} className="rounded border p-2 bg-muted/20">
                                                    <div className="flex justify-between items-center mb-1.5">
                                                      <span className="text-xs font-medium">Run {runData.run}</span>
                                                      <span className="text-[10px] text-muted-foreground">{runData.datasetFile}</span>
                                                    </div>
                                                    <div className="grid gap-1.5 grid-cols-2 sm:grid-cols-4 lg:grid-cols-7">
                                                      {METRIC_COLS.map(col => {
                                                        const value = runData.metrics[col.key]
                                                        if (value === undefined) return null
                                                        const quality = getQualityIndicator(col.key, value)
                                                        return (
                                                          <div key={col.key} className="text-center p-1.5 rounded bg-background">
                                                            <p className="text-[9px] text-muted-foreground uppercase">{col.label}</p>
                                                            <p className="text-xs font-semibold" style={{ color: PHASES[2].strongColor }}>{value.toFixed(4)}</p>
                                                            {quality && <span className={`text-[8px] ${quality.color}`}>{quality.label}</span>}
                                                          </div>
                                                        )
                                                      })}
                                                    </div>
                                                  </div>
                                                ))}
                                              </div>
                                            </CollapsibleContent>
                                          </Collapsible>
                                        )
                                      })}
                                    </div>
                                  )}
                                </div>
                              )
                            }

                            // ── Single-Model View (original) ──
                            if (!metrics) {
                              return (
                                <p className="text-xs text-muted-foreground text-center py-2">
                                  Metrics will be available after evaluation completes.
                                </p>
                              )
                            }

                            const MetricCard = ({ label, value, higherIsBetter, metricKey, std }: { label: string; value: number | undefined; higherIsBetter: boolean; metricKey: string; std?: number }) => {
                              const quality = getQualityIndicator(metricKey, value)
                              return (
                                <div className="rounded-lg border p-3 text-center">
                                  <p className="text-xs text-muted-foreground mb-1">{label}</p>
                                  <div className="flex items-center justify-center gap-2">
                                    <p className="text-xl font-bold" style={{ color: PHASES[2].strongColor }}>
                                      {fmt(value)}
                                      {std !== undefined && (
                                        <span className="text-xs font-normal text-muted-foreground"> ± {std.toFixed(4)}</span>
                                      )}
                                    </p>
                                    {quality && (
                                      <span className={`text-[10px] font-medium ${quality.color}`}>{quality.label}</span>
                                    )}
                                  </div>
                                  <p className="text-[10px] text-muted-foreground/70 mt-1">
                                    {higherIsBetter ? '↑ higher is better' : '↓ lower is better'}
                                  </p>
                                </div>
                              )
                            }

                            // Content for Summary view (average metrics with std)
                            const SummaryContent = () => {
                              const displayMetrics = averageMetrics || metrics
                              const isAverage = !!averageMetrics
                              return (
                                <div className="space-y-4">
                                  {isAverage && (
                                    <p className="text-xs text-muted-foreground text-center">
                                      Average across {totalRuns} runs (with ± standard deviation)
                                    </p>
                                  )}
                                  {/* Lexical */}
                                  <div>
                                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Lexical</h4>
                                    <div className="grid gap-3 sm:grid-cols-2">
                                      <MetricCard label="BLEU" value={isAverage ? (displayMetrics as any).bleu?.mean : displayMetrics.bleu} higherIsBetter={true} metricKey="bleu" std={isAverage ? (displayMetrics as any).bleu?.std : undefined} />
                                      <MetricCard label="ROUGE-L" value={isAverage ? (displayMetrics as any).rouge_l?.mean : displayMetrics.rouge_l} higherIsBetter={true} metricKey="rouge_l" std={isAverage ? (displayMetrics as any).rouge_l?.std : undefined} />
                                    </div>
                                  </div>
                                  {/* Semantic */}
                                  <div>
                                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Semantic</h4>
                                    <div className="grid gap-3 sm:grid-cols-2">
                                      <MetricCard label="BERTScore" value={isAverage ? (displayMetrics as any).bertscore?.mean : displayMetrics.bertscore} higherIsBetter={true} metricKey="bertscore" std={isAverage ? (displayMetrics as any).bertscore?.std : undefined} />
                                      <MetricCard label="MoverScore" value={isAverage ? (displayMetrics as any).moverscore?.mean : displayMetrics.moverscore} higherIsBetter={true} metricKey="moverscore" std={isAverage ? (displayMetrics as any).moverscore?.std : undefined} />
                                    </div>
                                  </div>
                                  {/* Diversity */}
                                  <div>
                                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Diversity</h4>
                                    <div className="grid gap-3 sm:grid-cols-3">
                                      <MetricCard label="Distinct-1" value={isAverage ? (displayMetrics as any).distinct_1?.mean : displayMetrics.distinct_1} higherIsBetter={true} metricKey="distinct_1" std={isAverage ? (displayMetrics as any).distinct_1?.std : undefined} />
                                      <MetricCard label="Distinct-2" value={isAverage ? (displayMetrics as any).distinct_2?.mean : displayMetrics.distinct_2} higherIsBetter={true} metricKey="distinct_2" std={isAverage ? (displayMetrics as any).distinct_2?.std : undefined} />
                                      <MetricCard label="Self-BLEU" value={isAverage ? (displayMetrics as any).self_bleu?.mean : displayMetrics.self_bleu} higherIsBetter={false} metricKey="self_bleu" std={isAverage ? (displayMetrics as any).self_bleu?.std : undefined} />
                                    </div>
                                  </div>
                                </div>
                              )
                            }

                            // Content for Per Run view
                            const PerRunContent = () => {
                              if (!perRunMetrics || perRunMetrics.length === 0) {
                                return <p className="text-xs text-muted-foreground text-center py-4">No per-run metrics available.</p>
                              }
                              return (
                                <div className="space-y-3">
                                  {perRunMetrics.map((runData) => (
                                    <div key={runData.run} className="rounded-lg border p-3 bg-muted/30">
                                      <div className="flex justify-between items-center mb-2">
                                        <span className="font-semibold text-sm">Run {runData.run}</span>
                                        <span className="text-xs text-muted-foreground">{runData.datasetFile}</span>
                                      </div>
                                      <div className="grid gap-2 grid-cols-2 sm:grid-cols-4 lg:grid-cols-7">
                                        {METRIC_COLS.map(col => {
                                          const value = runData.metrics[col.key]
                                          if (value === undefined || value === null) return null
                                          const quality = getQualityIndicator(col.key, value)
                                          return (
                                            <div key={col.key} className="text-center p-2 rounded bg-background">
                                              <p className="text-[10px] text-muted-foreground uppercase">{col.label}</p>
                                              <p className="text-sm font-semibold" style={{ color: PHASES[2].strongColor }}>{value.toFixed(4)}</p>
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

                            // For single run or no per-run data, just show summary
                            if (!hasMultipleRuns) {
                              return <SummaryContent />
                            }

                            // For multiple runs, show tabs
                            return (
                              <Tabs defaultValue="summary" className="w-full">
                                <TabsList className="grid w-full grid-cols-2 mb-4">
                                  <TabsTrigger value="summary">Summary</TabsTrigger>
                                  <TabsTrigger value="per-run">Per Run ({perRunMetrics?.length || 0})</TabsTrigger>
                                </TabsList>
                                <TabsContent value="summary">
                                  <SummaryContent />
                                </TabsContent>
                                <TabsContent value="per-run">
                                  <PerRunContent />
                                </TabsContent>
                              </Tabs>
                            )
                          })()}
                        </div>
                      </CollapsibleContent>
                    </div>
                  </Collapsible>
                </>
              )}
              </>
              )}
            </div>
          )}
        </div>

        {/* Sidebar - Quick Info */}
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-lg border p-4">
            <h4 className="font-semibold mb-3">Job Info</h4>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Status</span>
                <Badge className={status.color}>{status.label}</Badge>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-medium">{job.progress}%</span>
              </div>
              <Separator />
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Phase</span>
                {(() => {
                  // Map sub-phases to parent phase
                  const currentPhase = job.currentPhase?.toLowerCase()
                  let phaseInfo = null
                  if (currentPhase === 'composition' || currentPhase === 'sil' || currentPhase === 'mav') {
                    phaseInfo = PHASES.find(p => p.id === 'composition')
                  } else if (currentPhase === 'generation' || currentPhase === 'aml') {
                    phaseInfo = PHASES.find(p => p.id === 'generation')
                  } else if (currentPhase === 'evaluation' || currentPhase === 'mdqa') {
                    phaseInfo = PHASES.find(p => p.id === 'evaluation')
                  }
                  return phaseInfo ? (
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0" style={{ borderColor: phaseInfo.strongColor, color: phaseInfo.strongColor }}>
                      {phaseInfo.title}
                    </Badge>
                  ) : (
                    <span className="text-muted-foreground text-xs">Pending</span>
                  )
                })()}
              </div>
              {/* Run progress for multi-run jobs */}
              {((job.totalRuns && job.totalRuns > 1) || (job.config.generation?.total_runs && job.config.generation.total_runs > 1)) && (
                <>
                  <Separator />
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Run</span>
                    <span className="font-medium text-amber-600 dark:text-amber-400">
                      {job.currentRun || 1} / {job.totalRuns || job.config.generation?.total_runs || 1}
                    </span>
                  </div>
                </>
              )}
              <Separator />
              <div className="flex justify-between items-center">
                <span className="text-muted-foreground">Phases</span>
                <div className="flex gap-1">
                  {jobPhases.map((p: string) => {
                    const phaseInfo = PHASES.find(ph => ph.id === p)
                    return phaseInfo ? (
                      <Badge key={p} variant="outline" className="text-[10px] px-1.5 py-0" style={{ borderColor: phaseInfo.strongColor, color: phaseInfo.strongColor }}>
                        {phaseInfo.title}
                      </Badge>
                    ) : null
                  })}
                </div>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Created</span>
                <span className="font-medium text-xs">{new Date(job.createdAt).toLocaleString()}</span>
              </div>
              {job.completedAt && (
                <>
                  <Separator />
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">Completed</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px] font-mono">
                        {(() => {
                          const durationMs = job.completedAt - job.createdAt
                          const totalSecs = Math.floor(durationMs / 1000)
                          const hours = Math.floor(totalSecs / 3600)
                          const minutes = Math.floor((totalSecs % 3600) / 60)
                          const seconds = totalSecs % 60
                          if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`
                          if (minutes > 0) return `${minutes}m ${seconds}s`
                          return `${seconds}s`
                        })()}
                      </Badge>
                      <span className="font-medium text-xs">{new Date(job.completedAt).toLocaleString()}</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {job.config.generation && (
            <div className="rounded-lg border p-4">
              <h4 className="font-semibold mb-3">Generation Config</h4>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Reviews</span>
                  <span className="font-medium">{job.config.generation.count.toLocaleString()}</span>
                </div>
                <Separator />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Batch Size</span>
                  <span className="font-medium">{job.config.generation.batch_size}</span>
                </div>
                {job.config.generation.total_runs && job.config.generation.total_runs > 1 && (
                  <>
                    <Separator />
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Total Runs</span>
                      <span className="font-medium text-amber-600 dark:text-amber-400">{job.config.generation.total_runs}</span>
                    </div>
                  </>
                )}
                {job.config.generation.request_size && (
                  <>
                    <Separator />
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Request Size</span>
                      <span className="font-medium">{job.config.generation.request_size}</span>
                    </div>
                  </>
                )}
                <Separator />
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground shrink-0">Generation Model</span>
                  <span className="font-medium text-xs truncate" title={job.config.generation.model}>
                    {job.config.generation.model.split('/').pop()}
                  </span>
                </div>
                {job.config.subject_profile?.mav?.models && job.config.subject_profile?.mav?.models?.length > 0 && (
                  <>
                    <Separator />
                    <div className="flex justify-between gap-2">
                      <span className="text-muted-foreground shrink-0">MAV Models</span>
                      <div className="flex flex-col items-end gap-0.5">
                        {job.config.subject_profile?.mav?.models?.map((modelId: string, i: number) => (
                          <span key={i} className="font-medium text-xs truncate" title={modelId}>
                            {modelId.split('/').pop()}
                          </span>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Costs Analysis - 3-column phase breakdown */}
          {(() => {
            // Get stored estimates or calculate fallback for old jobs
            const storedEstimates = job.estimatedCost
            const jobPhases = job.phases || ['composition', 'generation', 'evaluation']
            const isReusing = !!job.reusedFrom

            // Fallback calculation for old jobs without stored estimates
            const calculateFallbackEstimates = () => {
              const defaultPricing = { input: 2.0, output: 8.0 }

              const mavModels = job.config.subject_profile?.mav?.models?.filter((m: string) => m) || []
              const M = mavModels.length || 1

              // Calculate review count
              const sentencesPerReview = job.config.attributes_profile?.length_range
                ? (job.config.attributes_profile.length_range[0] + job.config.attributes_profile.length_range[1]) / 2
                : 3.5

              let N: number
              if (job.config.generation?.count_mode === 'sentences') {
                const targetSentences = job.config.generation?.target_sentences || 1000
                N = Math.ceil(targetSentences / sentencesPerReview)
              } else {
                N = job.config.generation?.count || 0
              }

              // Composition cost: 3M + 1 calls
              const compositionCalls = !isReusing && jobPhases.includes('composition') ? 3 * M + 1 : 0
              const compositionTokens = compositionCalls > 0 ? M * (
                TOKEN_ESTIMATES.sil_research.input + TOKEN_ESTIMATES.sil_research.output +
                TOKEN_ESTIMATES.sil_generate_queries.input + TOKEN_ESTIMATES.sil_generate_queries.output +
                TOKEN_ESTIMATES.sil_answer_queries.input + TOKEN_ESTIMATES.sil_answer_queries.output
              ) + TOKEN_ESTIMATES.sil_classify.input + TOKEN_ESTIMATES.sil_classify.output : 0
              const compositionCost = (compositionTokens / 1_000_000) * (defaultPricing.input + defaultPricing.output) / 2

              // Generation cost: N calls
              const generationCalls = jobPhases.includes('generation') ? N : 0
              const generationTokens = generationCalls * (TOKEN_ESTIMATES.aml_per_review.input + TOKEN_ESTIMATES.aml_per_review.output)
              const generationCost = (generationTokens / 1_000_000) * (defaultPricing.input + defaultPricing.output) / 2

              return {
                composition: { cost: compositionCost, calls: compositionCalls },
                generation: { cost: generationCost, calls: generationCalls },
                evaluation: { cost: 0, calls: 0 },
                total: {
                  cost: compositionCost + generationCost,
                  calls: compositionCalls + generationCalls,
                  tokens: compositionTokens + generationTokens,
                },
              }
            }

            const estimates = storedEstimates || calculateFallbackEstimates()
            const actualCosts = job.actualCost

            // Format helpers
            const formatCost = (cost: number) => {
              if (cost === 0) return '$0.00'
              if (cost < 0.01) return '< $0.01'
              return `$${cost.toFixed(2)}`
            }
            const formatNumber = (n: number) => n.toLocaleString()

            return (
              <div className="rounded-lg border p-4">
                <div className="flex items-center gap-2 mb-4">
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                  <h4 className="font-semibold">Cost Analysis</h4>
                  {!storedEstimates && (
                    <Badge variant="outline" className="text-[9px] px-1.5 py-0">
                      Estimated
                    </Badge>
                  )}
                </div>

                <div className="space-y-4">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="rounded-lg bg-muted/50 p-2">
                      <div className="text-sm font-semibold">{formatNumber(estimates.total.calls)}</div>
                      <div className="text-[9px] text-muted-foreground">LLM Calls</div>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-2">
                      <div className="text-sm font-semibold">{formatNumber(estimates.total.tokens)}</div>
                      <div className="text-[9px] text-muted-foreground">Est. Tokens</div>
                    </div>
                    <div className="rounded-lg p-2" style={{ backgroundColor: `${PHASES[1].strongColor}15` }}>
                      <div className="text-sm font-semibold" style={{ color: PHASES[1].strongColor }}>
                        ~{formatCost(estimates.total.cost)}
                      </div>
                      <div className="text-[9px] text-muted-foreground">Est. Total</div>
                    </div>
                  </div>

                  {/* Phase Breakdown - 3 columns */}
                  <div className="grid grid-cols-3 gap-2">
                    {/* Composition Phase */}
                    {jobPhases.includes('composition') && (
                      <div className="rounded border p-2">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PHASES[0].strongColor }} />
                          <span className="text-[10px] font-medium">Composition</span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-[10px]">
                            <span className="text-muted-foreground">Est:</span>
                            <span className={isReusing ? 'text-green-600' : ''}>
                              {isReusing ? '$0 (reused)' : formatCost(estimates.composition.cost)}
                            </span>
                          </div>
                          {job.status === 'completed' && (
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Act:</span>
                              <span className="text-amber-600">
                                {actualCosts?.composition ? formatCost(actualCosts.composition.cost) : '—'}
                              </span>
                            </div>
                          )}
                          {!isReusing && (
                            <div className="text-[9px] text-muted-foreground mt-1">
                              {estimates.composition.calls} calls
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Generation Phase */}
                    {jobPhases.includes('generation') && (
                      <div className="rounded border p-2">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PHASES[1].strongColor }} />
                          <span className="text-[10px] font-medium">Generation</span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-[10px]">
                            <span className="text-muted-foreground">Est:</span>
                            <span>{formatCost(estimates.generation.cost)}</span>
                          </div>
                          {job.status === 'completed' && (
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Act:</span>
                              <span className="text-amber-600">
                                {actualCosts?.generation ? formatCost(actualCosts.generation.cost) : '—'}
                              </span>
                            </div>
                          )}
                          <div className="text-[9px] text-muted-foreground mt-1">
                            {formatNumber(estimates.generation.calls)} calls
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Evaluation Phase */}
                    {jobPhases.includes('evaluation') && (
                      <div className="rounded border p-2">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: PHASES[2].strongColor }} />
                          <span className="text-[10px] font-medium">Evaluation</span>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-[10px]">
                            <span className="text-muted-foreground">Est:</span>
                            <span className="text-green-600">$0.00</span>
                          </div>
                          {job.status === 'completed' && (
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Act:</span>
                              <span className="text-green-600">$0.00</span>
                            </div>
                          )}
                          <div className="text-[9px] text-muted-foreground mt-1">
                            Local metrics
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Footer note */}
                  <div className="text-[9px] text-muted-foreground pt-2 border-t text-center">
                    {storedEstimates
                      ? 'Estimates saved at job creation'
                      : 'Estimates calculated using default pricing'
                    }
                    {!actualCosts && job.status === 'completed' && (
                      <span className="block mt-0.5">Actual costs not tracked (pre-update job)</span>
                    )}
                  </div>
                </div>
              </div>
            )
          })()}
        </div>
      </div>
        </div>
      </div>

      {/* Phase Log Panel - docked at bottom */}
      <PhaseLogPanel
        jobId={jobId as Id<'jobs'>}
        activePhaseTab={activeTab as LogTab}
        enabledPhases={enabledPhaseIds}
      />
    </div>
  )
}
