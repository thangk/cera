import { createFileRoute, useNavigate, useSearch, Link } from '@tanstack/react-router'
import { useMutation, useQuery, useAction } from 'convex/react'
import { api } from 'convex/_generated/api'
import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { toast } from 'sonner'
import { PYTHON_API_URL } from '../lib/api-urls'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  Users,
  Sliders,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Loader2,
  AlertTriangle,
  Info,
  Settings,
  RotateCcw,
  HelpCircle,
  Eye,
  Lock,
  Unlock,
  Play,
  Cog,
  Activity,
  Copy,
  X,
  Upload,
  FileText,
  ShieldCheck,
  FileUp,
  Sparkles,
  Trash2,
  Layers,
  DollarSign,
  RefreshCw,
  XCircle,
  Database,
  Plus,
  BarChart3,
} from 'lucide-react'
import type { Id } from 'convex/_generated/dataModel'
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Textarea } from '../components/ui/textarea'
import { Slider } from '../components/ui/slider'
import { Switch } from '../components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog'
import { Progress } from '../components/ui/progress'
import { Badge } from '../components/ui/badge'
import { Checkbox } from '../components/ui/checkbox'
import { Separator } from '../components/ui/separator'
import { FileDropZone } from '../components/ui/file-drop-zone'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../components/ui/collapsible'
import domainPatternsDefault from '../config/domain-patterns.json'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../components/ui/tooltip'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../components/ui/popover'
import { useOpenRouterModels } from '../hooks/use-openrouter-models'
import { useUIConstraints, type UIConstraints } from '../hooks/use-ui-constraints'
import { LLMSelector, type ValidationStatus } from '../components/llm-selector'
import { PresetSelector, type LLMPreset } from '../components/preset-selector'
import { HeuristicWizard } from '../components/heuristic'
import { CeraTargetRow, DEFAULT_CERA_TARGET, type CeraTarget } from '../components/target-dataset-row'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '../components/ui/alert-dialog'

// Model validation state type
interface ModelValidationState {
  status: ValidationStatus
  error?: string
  modelId?: string  // Track which model was validated
  note?: string     // Optional note for specialized models
}

// Wrapper for ablation-controlled sections - keeps layout stable when disabled
function AblationSection({
  enabled,
  effect,
  children
}: {
  enabled: boolean
  effect: string
  children: React.ReactNode
}) {
  return (
    <div className="relative">
      <div className={enabled ? '' : 'opacity-25 pointer-events-none select-none'}>
        {children}
      </div>
      {!enabled && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="rounded-md bg-background/95 border border-dashed border-muted-foreground/30 px-3 py-2 shadow-sm">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">Disabled:</span> {effect}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// Ablation Settings Dialog Component
function AblationSettingsDialog({ config, updateConfig }: { config: any; updateConfig: (path: string, value: any) => void }) {
  const disabledCount = [
    !config.ablation.sil_enabled,
    !config.ablation.mav_enabled,
    !config.ablation.polarity_enabled,
    !config.ablation.noise_enabled,
    !config.ablation.age_enabled,
    !config.ablation.sex_enabled,
  ].filter(Boolean).length

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="text-muted-foreground">
          <Settings className="h-4 w-4 mr-1" />
          Components
          {disabledCount > 0 && (
            <Badge variant="secondary" className="ml-2 px-1.5 py-0 text-[10px]">
              {disabledCount} off
            </Badge>
          )}
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Pipeline Components</DialogTitle>
          <DialogDescription>
            Toggle components on/off for ablation studies to measure their individual impact on generation quality
          </DialogDescription>
        </DialogHeader>

        {/* Core Innovations - SIL & MAV */}
        <div className="space-y-2 mt-2">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold">Core Innovations</h3>
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">Key contributions</Badge>
          </div>
          <div className="rounded-lg border border-primary/30 bg-primary/5">
            <table className="w-full text-sm table-fixed">
              <thead>
                <tr className="border-b border-primary/20 bg-primary/10">
                  <th className="text-left p-3 font-medium w-24">Component</th>
                  <th className="text-left p-3 font-medium">Description</th>
                  <th className="text-left p-3 font-medium">Effect When Off</th>
                  <th className="text-center p-3 font-medium w-20">Enabled</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-primary/20">
                  <td className="p-3 font-mono text-xs font-semibold text-primary">SIL</td>
                  <td className="p-3 text-muted-foreground">Subject Intelligence Layer - web search grounding</td>
                  <td className="p-3 text-muted-foreground">No factual grounding</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.sil_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.sil_enabled', v)}
                    />
                  </td>
                </tr>
                <tr>
                  <td className="p-3 font-mono text-xs font-semibold text-primary">MAV</td>
                  <td className="p-3 text-muted-foreground">Multi-Agent Verification - 2/3 consensus voting</td>
                  <td className="p-3 text-muted-foreground">Single LLM (no consensus)</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.mav_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.mav_enabled', v)}
                    />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Generation Controls */}
        <div className="space-y-2 mt-4">
          <h3 className="text-sm font-semibold text-muted-foreground">Generation Controls</h3>
          <div className="rounded-lg border">
            <table className="w-full text-sm table-fixed">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="text-left p-3 font-medium w-24">Component</th>
                  <th className="text-left p-3 font-medium">Description</th>
                  <th className="text-left p-3 font-medium">Effect When Off</th>
                  <th className="text-center p-3 font-medium w-20">Enabled</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-3 font-mono text-xs">Polarity</td>
                  <td className="p-3 text-muted-foreground">Sentiment distribution control</td>
                  <td className="p-3 text-muted-foreground">Let LLM decide sentiment</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.polarity_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.polarity_enabled', v)}
                    />
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-3 font-mono text-xs">Noise</td>
                  <td className="p-3 text-muted-foreground">Naturalness injection</td>
                  <td className="p-3 text-muted-foreground">Clean text only</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.noise_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.noise_enabled', v)}
                    />
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-3 font-mono text-xs">Age</td>
                  <td className="p-3 text-muted-foreground">Reviewer age range</td>
                  <td className="p-3 text-muted-foreground">Random ages (13-80)</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.age_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.age_enabled', v)}
                    />
                  </td>
                </tr>
                <tr>
                  <td className="p-3 font-mono text-xs">Sex</td>
                  <td className="p-3 text-muted-foreground">Reviewer sex distribution</td>
                  <td className="p-3 text-muted-foreground">All unspecified sex</td>
                  <td className="p-3 text-center">
                    <Switch
                      checked={config.ablation.sex_enabled}
                      onCheckedChange={(v) => updateConfig('ablation.sex_enabled', v)}
                    />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick info about current ablation state */}
        {disabledCount > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            <span className="text-xs text-muted-foreground">Disabled components:</span>
            {!config.ablation.sil_enabled && <Badge variant="outline" className="text-xs">SIL</Badge>}
            {!config.ablation.mav_enabled && <Badge variant="outline" className="text-xs">MAV</Badge>}
            {!config.ablation.polarity_enabled && <Badge variant="outline" className="text-xs">Polarity</Badge>}
            {!config.ablation.noise_enabled && <Badge variant="outline" className="text-xs">Noise</Badge>}
            {!config.ablation.age_enabled && <Badge variant="outline" className="text-xs">Age</Badge>}
            {!config.ablation.sex_enabled && <Badge variant="outline" className="text-xs">Sex</Badge>}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

export const Route = createFileRoute('/create-job')({
  component: GenerateWizard,
  validateSearch: (search: Record<string, unknown>) => {
    return {
      reset: search.reset === true || search.reset === 'true',
    }
  },
})

// 3-Phase Pipeline matching CERA architecture
const PHASES = [
  {
    id: 'composition',
    title: 'COMPOSITION',
    strongColor: '#4e95d9',
    lightColor: '#dceaf7',
    description: 'Gather subject intelligence with SIL and verify with MAV',
  },
  {
    id: 'generation',
    title: 'GENERATION',
    strongColor: '#f2aa84',
    lightColor: '#fbe3d6',
    description: 'Generate synthetic reviews using AML with LLM batch processing',
  },
  {
    id: 'evaluation',
    title: 'EVALUATION',
    strongColor: '#8ed973',
    lightColor: '#d9f2d0',
    description: 'Assess quality with MDQA multi-dimensional metrics',
  },
]

// All available MDQA metrics
const ALL_METRICS = ['bertscore', 'bleu', 'rouge_l', 'moverscore', 'distinct_1', 'distinct_2', 'self_bleu']

// PhaseTab component with progress indicator
function PhaseTab({
  phase,
  progress,
  isActive,
  onClick,
  disabled,
  locked,
}: {
  phase: typeof PHASES[0]
  progress: number
  isActive: boolean
  onClick: () => void
  disabled?: boolean
  locked?: boolean
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`relative px-4 py-2 rounded-lg overflow-hidden transition-all font-medium text-sm ${
        isActive ? 'ring-2 ring-offset-2' : ''
      } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
      style={{
        backgroundColor: phase.lightColor,
        // @ts-ignore - CSS custom property for ring color
        '--tw-ring-color': phase.strongColor,
      }}
    >
      {/* Progress fill */}
      <div
        className="absolute inset-y-0 left-0 transition-all duration-500"
        style={{
          width: `${progress}%`,
          backgroundColor: phase.strongColor,
          opacity: 0.3,
        }}
      />

      {/* Label */}
      <span className="relative flex items-center gap-2" style={{ color: phase.strongColor }}>
        {phase.title}
        {/* Lock icon when locked (reusing) */}
        {locked && (
          <Lock className="h-4 w-4" />
        )}
        {/* Progress percentage */}
        {!locked && progress > 0 && progress < 100 && (
          <span className="text-xs opacity-70">{progress}%</span>
        )}
        {/* Checkmark when complete (and not locked) */}
        {!locked && progress === 100 && (
          <CheckCircle className="h-4 w-4" />
        )}
      </span>
    </button>
  )
}


const NOISE_PRESETS = [
  { value: 'light', label: 'Light', description: '0.5% typos, minor grammar' },
  { value: 'moderate', label: 'Moderate', description: '1% typos, colloquialisms' },
  { value: 'heavy', label: 'Heavy', description: '3% typos, very casual' },
]

const STORAGE_KEY = 'cera-generate-config'
const UI_STATE_KEY = 'cera-generate-ui-state'

const DEFAULT_CONFIG = {
  name: '',
  // Selected pipeline phases
  selectedPhases: [] as string[],
  subject_profile: {
    query: '',
    additional_context: '',
    region: 'united states',
    domain: 'general',
    aspect_categories: [] as string[],
    aspect_category_mode: 'infer' as 'preset' | 'infer' | 'import' | 'ref_dataset',
    sentiment_depth: 'praise and complain',
    mav: {
      models: ['', '', ''],
      similarity_threshold: 0.75,
      max_queries: 30,
    },
  },
  reviewer_profile: {
    age_range: [18, 65],
    sex_distribution: {
      male: 0.5,
      female: 0.5,
      unknown: 0.0,
    },
    additional_context: '',
    persona_ratio: 0.9,
  },
  attributes_profile: {
    polarity: { positive: 0.65, neutral: 0.15, negative: 0.2 },
    noise: {
      typo_rate: 0.01,
      colloquialism: true,
      grammar_errors: true,
      preset: 'moderate' as string | undefined,
    },
    length_range: [2, 5],
    edge_lengths: { min_length: 1, min_chance: 0.15, max_length: 15, max_chance: 0.05 },
    temperature_range: [0.7, 0.9], // LLM generation temperature range
    cap_weights: { standard: 0.55, lowercase: 0.20, mixed: 0.15, emphasis: 0.10 },
  },
  generation: {
    // Multi-target dataset support
    target_prefix: '' as string, // File naming prefix (defaults to sanitized job name if empty)
    targets: [{
      count_mode: 'sentences' as 'sentences' | 'reviews',
      target_value: 100,
      batch_size: 1,
      request_size: 25,
      total_runs: 1,
      runs_mode: 'parallel' as 'parallel' | 'sequential',
      neb_depth: 0,
    }],
    parallel_targets: true,
    // Legacy fields (derived from targets[0] at submission time for backward compat)
    count: 100,
    count_mode: 'sentences' as 'reviews' | 'sentences',
    target_sentences: 100 as number | undefined,
    batch_size: 1,
    request_size: 25,
    provider: '',
    model: '',
    output_formats: ['jsonl', 'semeval_xml'] as string[],  // JSONL always included
    dataset_mode: 'both' as string,
    total_runs: 1,
    neb_enabled: false,
    neb_depth: 0,
    models: [] as string[], // Multi-model comparison (overrides model when populated)
    parallel_models: true, // Run models concurrently (default ON)
  },
  // Evaluation configuration
  evaluation: {
    metrics: [...ALL_METRICS] as string[],
    reference_metrics_enabled: false, // Toggle for lexical/semantic metrics that need reference data
  },
  // Pipeline component toggles for ablation studies
  ablation: {
    sil_enabled: true,
    mav_enabled: true,
    polarity_enabled: true,
    noise_enabled: true,
    age_enabled: true,
    sex_enabled: true,
  },
}


function GenerateWizard() {
  const navigate = useNavigate()
  const { reset: shouldReset } = useSearch({ from: '/create-job' })
  const createJob = useMutation(api.jobs.create)
  const runContextsOnly = useAction(api.compositionAction.runContextsOnly)
  const runPipeline = useAction(api.pipelineAction.runPipeline)
  const settings = useQuery(api.settings.get)
  const completedJobs = useQuery(api.jobs.listForReuse)
  const { constraints } = useUIConstraints()
  const { models: rawModels, processedModels, providers, groupedModels, loading: modelsLoading } = useOpenRouterModels()

  // LLM Presets
  const presets = useQuery(api.llmPresets.list)
  const defaultPreset = useQuery(api.llmPresets.getDefault)
  const [selectedPresetId, setSelectedPresetId] = useState<Id<"llm_presets"> | null>(null)
  const [presetApplied, setPresetApplied] = useState(false)

  // Generation method: "cera" (default), "heuristic" (RQ1 baseline), or "real" (eval-only)
  const [method, setMethod] = useState<'cera' | 'heuristic' | 'real'>(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.method ?? 'cera'
        } catch { /* ignore */ }
      }
    }
    return 'cera'
  })

  // Phase selection vs wizard view - load from localStorage if available
  const [showWizard, setShowWizard] = useState(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.showWizard ?? false
        } catch { /* ignore */ }
      }
    }
    return false
  })
  // Heuristic wizard view state
  const [showHeuristicWizard, setShowHeuristicWizard] = useState(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.showHeuristicWizard ?? false
        } catch { /* ignore */ }
      }
    }
    return false
  })
  const [activeTab, setActiveTab] = useState(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.activeTab ?? 0
        } catch { /* ignore */ }
      }
    }
    return 0
  })
  const [submitting, setSubmitting] = useState(false)
  const [validatingModels, setValidatingModels] = useState(false)

  // Real-time model validation state
  const [modelValidations, setModelValidations] = useState<Record<string, ModelValidationState>>({})
  const [showInvalidModelsAlert, setShowInvalidModelsAlert] = useState(false)
  const [showValidationErrors, setShowValidationErrors] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  // Job reuse state (for GENERATION without COMPOSITION)
  const [reusedFrom, setReusedFrom] = useState<Id<"jobs"> | null>(null)
  const [checkingContexts, setCheckingContexts] = useState(false)
  const [contextsValid, setContextsValid] = useState<boolean | null>(null)
  const [sourceConfig, setSourceConfig] = useState<{
    subject_profile?: {
      query: string
      region: string
      domain?: string
      sentiment_depth: string | number
      aspect_categories?: string[]
      aspect_category_mode?: string
      additional_context?: string
      mav?: {
        enabled: boolean
        models: string[]
        similarity_threshold?: number
        max_queries?: number
      }
    }
    reviewer_profile?: {
      age_range: number[]
      sex_distribution: { male: number; female: number; unspecified: number }
      additional_context?: string
    }
    attributes_profile?: {
      polarity: { positive: number; neutral: number; negative: number }
      noise: {
        typo_rate: number
        preset?: string
        colloquialism?: boolean
        grammar_errors?: boolean
      }
      length_range: number[]
      temperature_range?: number[]
    }
    ablation?: {
      sil_enabled: boolean
      mav_enabled: boolean
      polarity_enabled: boolean
      noise_enabled: boolean
      age_enabled: boolean
      sex_enabled: boolean
    }
    referenceDataset?: {
      fileName?: string
      useForEvaluation: boolean
      extractedSubjectContext?: boolean
      extractedReviewerContext?: boolean
    }
    reference_dataset?: {
      fileName?: string
      useForEvaluation?: boolean
      extractedSubjectContext?: boolean
      extractedReviewerContext?: boolean
    }
  } | null>(null)

  // Dataset upload state (for EVALUATION only)
  const [datasetFile, setDatasetFile] = useState<File | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  // Reference dataset for lexical/semantic metrics comparison
  const [referenceDatasetFile, setReferenceDatasetFile] = useState<File | null>(null)

  // Quick stats from parsing reference dataset (before LLM extraction)
  interface QuickStats {
    reviewCount: number
    sentenceCount: number
    opinionCount: number
    avgSentencesPerReview: number
    polarityCounts: Record<string, number>
    categoryCount: number
    inferredDomain: string
    status: 'idle' | 'parsing' | 'success' | 'error'
    error?: string
  }
  const [referenceQuickStats, setReferenceQuickStats] = useState<QuickStats | null>(null)

  // Domain patterns for Quick Stats inference (fetched from API, fallback to bundled defaults)
  const [domainPatterns, setDomainPatterns] = useState<{ domains: Array<{ name: string; keywords: string[] }> }>(domainPatternsDefault)

  // Reference dataset options for context extraction and MDQA comparison - load from localStorage
  const [referenceOptions, setReferenceOptions] = useState(() => {
    const defaultOptions = {
      useForEvaluation: true,       // Use for MDQA comparison (Dreal vs Dgen)
      extractSubjectContext: false, // Extract subject context from reference
      extractReviewerContext: false, // Extract reviewer context from reference
      sampleCount: 25,              // Number of reviews to sample for context extraction (10-50)
    }
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return { ...defaultOptions, ...parsed.referenceOptions }
        } catch { /* ignore */ }
      }
    }
    return defaultOptions
  })

  // Self Test state (split dataset into two halves for self-referencing metrics)
  const [selfTest, setSelfTest] = useState({ enabled: false, splitMode: 'random' as 'random' | 'sequential' })

  // Real dataset target sizes (method="real" eval-only: subsample at each size)
  const [realTargets, setRealTargets] = useState<Array<{ count_mode: string; target_value: number }>>([
    { count_mode: 'sentences', target_value: 100 },
    { count_mode: 'sentences', target_value: 500 },
    { count_mode: 'sentences', target_value: 1000 },
    { count_mode: 'sentences', target_value: 1500 },
    { count_mode: 'sentences', target_value: 2000 },
  ])

  // RDE (Reference Dataset Extraction) state
  const [rdeModel, setRdeModel] = useState<string>('')
  const [rdeExtractionState, setRdeExtractionState] = useState<{
    status: 'idle' | 'extracting' | 'success' | 'error'
    progress: string
    step: number
    totalSteps: number
    error?: string
  }>({ status: 'idle', progress: '', step: 0, totalSteps: 6 })

  // Extracted reference context (cached by file hash)
  interface ExtractedRefContext {
    subject_query: string | null
    additional_context: string | null
    reviewer_context: string | null
    domain: { value: string | null; confidence: number; reason?: string } | null
    region: { value: string | null; confidence: number; reason?: string } | null
    polarity: { positive: number; neutral: number; negative: number } | null
    sex_distribution: { male: number; female: number; unknown: number; detected_count: number } | null
    noise: { typo_rate: number; has_colloquialisms: boolean; sample_size: number } | null
    review_length: { avg_sentences: number; min_sentences: number; max_sentences: number; suggested_range: number[] } | null
    aspect_categories: string[]
    sample_count: number
    total_reviews: number
  }
  const [extractedRefContext, setExtractedRefContext] = useState<ExtractedRefContext | null>(null)
  // RDE token usage records (forwarded to pipeline for tokens.json)
  const [rdeUsage, setRdeUsage] = useState<Array<Record<string, unknown>> | null>(null)

  // Local storage key for ref-context cache
  const REF_CONTEXT_CACHE_KEY = 'cera-ref-context-cache'

  // Helper to compute simple hash of file for caching
  const computeFileHash = async (file: File): Promise<string> => {
    const buffer = await file.slice(0, 10000).arrayBuffer() // First 10KB
    const hashArray = Array.from(new Uint8Array(buffer))
    const hashStr = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    return `${file.name}-${file.size}-${hashStr.substring(0, 32)}`
  }

  // Fetch domain patterns from API on mount (fallback to bundled defaults)
  useEffect(() => {
    const fetchDomainPatterns = async () => {
      try {
        const pythonApiUrl = PYTHON_API_URL
        const response = await fetch(`${pythonApiUrl}/api/domain-patterns`)
        if (response.ok) {
          const data = await response.json()
          if (data.domains?.length > 0) {
            setDomainPatterns(data)
          }
        }
      } catch {
        // Silently fallback to bundled defaults
      }
    }
    fetchDomainPatterns()
  }, [])

  // Load cached context when file changes
  useEffect(() => {
    const loadCachedContext = async () => {
      if (!referenceDatasetFile) {
        // Don't clear extractedRefContext when file is removed - keep it cached
        return
      }
      const hash = await computeFileHash(referenceDatasetFile)
      const cached = localStorage.getItem(REF_CONTEXT_CACHE_KEY)
      if (cached) {
        try {
          const cacheData = JSON.parse(cached)
          if (cacheData.datasets?.[hash]) {
            setExtractedRefContext(cacheData.datasets[hash].context)
            setRdeExtractionState({ status: 'success', progress: 'Loaded from cache', step: 6, totalSteps: 6 })
            return
          }
        } catch { /* ignore */ }
      }
      // No cache found - reset extraction state
      setRdeExtractionState({ status: 'idle', progress: '', step: 0, totalSteps: 6 })
    }
    loadCachedContext()
  }, [referenceDatasetFile])

  // Form state - load from localStorage if available (unless reset param is set)
  const [config, setConfig] = useState(() => {
    // If reset param is set, don't load from localStorage
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          // Deep merge to preserve new fields added to DEFAULT_CONFIG (like total_runs)
          // that might be missing from older localStorage data
          const deepMerge = (target: any, source: any): any => {
            const result = { ...target }
            for (const key of Object.keys(source)) {
              if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = deepMerge(target[key] || {}, source[key])
              } else {
                result[key] = source[key]
              }
            }
            return result
          }
          return deepMerge(DEFAULT_CONFIG, parsed)
        } catch {
          // Invalid JSON, use default
        }
      }
    }
    return DEFAULT_CONFIG
  })

  // Handle reset parameter - clear localStorage and remove query param
  useEffect(() => {
    if (shouldReset && typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY)
      localStorage.removeItem(UI_STATE_KEY)
      // Remove the reset param from URL to prevent re-triggering on refresh
      navigate({ to: '/create-job', search: {}, replace: true })
    }
  }, [shouldReset, navigate])

  // Validate model IDs when models are loaded - clear invalid ones from localStorage
  useEffect(() => {
    if (modelsLoading || processedModels.length === 0) return

    const validModelIds = new Set(processedModels.map(m => m.id))
    let hasInvalidModels = false
    const newConfig = { ...config }

    // Check MAV models
    const mavModels = config.subject_profile?.mav?.models || []
    const validatedMavModels = mavModels.map((modelId: string) => {
      if (modelId && !validModelIds.has(modelId)) {
        hasInvalidModels = true
        console.warn(`Invalid MAV model ID found: ${modelId}`)
        return ''
      }
      return modelId
    })

    // Check generation model
    const genModel = config.generation?.model || ''
    let validatedGenModel = genModel
    if (genModel && !validModelIds.has(genModel)) {
      hasInvalidModels = true
      console.warn(`Invalid generation model ID found: ${genModel}`)
      validatedGenModel = ''
    }

    if (hasInvalidModels) {
      newConfig.subject_profile = {
        ...newConfig.subject_profile,
        mav: {
          ...newConfig.subject_profile.mav,
          models: validatedMavModels,
        },
      }
      newConfig.generation = {
        ...newConfig.generation,
        model: validatedGenModel,
      }
      setConfig(newConfig)
      toast.warning('Some cached model selections were invalid and have been cleared. Please re-select your models.')
    }
  }, [modelsLoading, processedModels]) // Don't include config to avoid infinite loop

  // Save config to localStorage when it changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
  }, [config])

  // Save UI state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem(UI_STATE_KEY, JSON.stringify({
      method,
      showWizard,
      showHeuristicWizard,
      activeTab,
      referenceOptions,
      // Store file names for reminder on reload (files themselves can't be persisted)
      referenceDatasetFileName: referenceDatasetFile?.name || null,
      datasetFileName: datasetFile?.name || null,
    }))
  }, [method, showWizard, showHeuristicWizard, activeTab, referenceOptions, referenceDatasetFile, datasetFile])

  // Track previously uploaded file names (shown as reminder after page refresh)
  const [previousReferenceFileName, setPreviousReferenceFileName] = useState<string | null>(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.referenceDatasetFileName || null
        } catch { /* ignore */ }
      }
    }
    return null
  })
  const [previousDatasetFileName, setPreviousDatasetFileName] = useState<string | null>(() => {
    if (typeof window !== 'undefined' && !shouldReset) {
      const saved = localStorage.getItem(UI_STATE_KEY)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          return parsed.datasetFileName || null
        } catch { /* ignore */ }
      }
    }
    return null
  })

  // Clear previous file name reminder when file is uploaded
  useEffect(() => {
    if (referenceDatasetFile) {
      setPreviousReferenceFileName(null)
    }
  }, [referenceDatasetFile])

  // Calculate quick stats when reference dataset file is uploaded
  useEffect(() => {
    if (!referenceDatasetFile) {
      setReferenceQuickStats(null)
      return
    }

    const calculateQuickStats = async () => {
      setReferenceQuickStats({ reviewCount: 0, sentenceCount: 0, opinionCount: 0, avgSentencesPerReview: 0, polarityCounts: {}, categoryCount: 0, inferredDomain: '', status: 'parsing' })
      try {
        const reviews = await parseFullReviewsFromFile(referenceDatasetFile)
        if (reviews.length === 0) {
          setReferenceQuickStats({ reviewCount: 0, sentenceCount: 0, opinionCount: 0, avgSentencesPerReview: 0, polarityCounts: {}, categoryCount: 0, inferredDomain: '', status: 'error', error: 'No reviews found' })
          return
        }

        // Calculate stats
        const sentenceCount = reviews.reduce((sum, r) => sum + r.sentences.length, 0)
        const opinions = reviews.flatMap(r => r.sentences.flatMap(s => s.opinions))
        const opinionCount = opinions.length

        // Polarity counts
        const polarityCounts: Record<string, number> = {}
        opinions.forEach(o => {
          const pol = (o.polarity || 'neutral').toLowerCase()
          polarityCounts[pol] = (polarityCounts[pol] || 0) + 1
        })

        // Unique categories
        const categories = new Set(opinions.map(o => o.category).filter(Boolean))

        // Infer domain from categories using config file (empty string if no match - won't display)
        // Use scoring system: domain with most keyword matches wins
        const categoryStr = Array.from(categories).join(' ').toUpperCase()
        let inferredDomain = ''
        let bestScore = 0
        for (const domain of domainPatterns.domains) {
          const score = domain.keywords.filter(keyword => categoryStr.includes(keyword)).length
          if (score > bestScore) {
            bestScore = score
            inferredDomain = domain.name
          }
        }

        setReferenceQuickStats({
          reviewCount: reviews.length,
          sentenceCount,
          opinionCount,
          avgSentencesPerReview: sentenceCount / reviews.length,
          polarityCounts,
          categoryCount: categories.size,
          inferredDomain,
          status: 'success',
        })
      } catch (e) {
        setReferenceQuickStats({ reviewCount: 0, sentenceCount: 0, opinionCount: 0, avgSentencesPerReview: 0, polarityCounts: {}, categoryCount: 0, inferredDomain: '', status: 'error', error: e instanceof Error ? e.message : 'Failed to parse' })
      }
    }

    calculateQuickStats()
  }, [referenceDatasetFile])

  useEffect(() => {
    if (datasetFile) {
      setPreviousDatasetFileName(null)
    }
  }, [datasetFile])

  // Compute dynamic wizard tabs based on selected phases
  const selectedPhases = config.selectedPhases as string[]

  // Reset "real" method if user leaves eval-only mode
  const isEvalOnly = selectedPhases.length === 1 && selectedPhases.includes('evaluation')
  useEffect(() => {
    if (method === 'real' && !isEvalOnly) {
      setMethod('cera')
    }
  }, [isEvalOnly, method])

  const wizardTabs = useMemo(() => {
    const tabs: Array<{ id: string; label: string; color?: string; lightColor?: string }> = [
      { id: 'input', label: 'Input' },
    ]
    if (selectedPhases.includes('composition')) {
      tabs.push({ id: 'composition', label: 'COMPOSITION', color: PHASES[0].strongColor, lightColor: PHASES[0].lightColor })
    }
    if (selectedPhases.includes('generation')) {
      tabs.push({ id: 'generation', label: 'GENERATION', color: PHASES[1].strongColor, lightColor: PHASES[1].lightColor })
    }
    if (selectedPhases.includes('evaluation')) {
      tabs.push({ id: 'evaluation', label: 'EVALUATION', color: PHASES[2].strongColor, lightColor: PHASES[2].lightColor })
    }
    tabs.push({ id: 'output', label: 'Output' })
    return tabs
  }, [selectedPhases])

  const currentTabId = wizardTabs[activeTab]?.id || 'input'

  // Determine if Next button should be disabled due to RDE extraction requirement
  // When reference dataset is uploaded and composition is selected, user must extract context first
  const isNextDisabledByRde = useMemo(() => {
    // Only applies when on INPUT tab
    if (currentTabId !== 'input') return false
    // Only applies when composition phase is selected
    if (!selectedPhases.includes('composition')) return false
    // Only applies when a reference dataset file is uploaded
    if (!referenceDatasetFile) return false
    // Disabled unless extraction is complete
    return rdeExtractionState.status !== 'success'
  }, [currentTabId, selectedPhases, referenceDatasetFile, rdeExtractionState.status])

  // Reset form to defaults
  const resetConfig = () => {
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem(UI_STATE_KEY)
    localStorage.removeItem(REF_CONTEXT_CACHE_KEY) // Clear extraction cache
    setConfig(DEFAULT_CONFIG)
    setSelectedPresetId(null)
    setPresetApplied(false)
    setReusedFrom(null)
    setShowWizard(false)
    setActiveTab(0)
    setDatasetFile(null)
    setReferenceDatasetFile(null)
    setReferenceOptions({
      useForEvaluation: true,
      extractSubjectContext: false,
      extractReviewerContext: false,
      sampleCount: 25,
    })
    setPreviousReferenceFileName(null)
    setPreviousDatasetFileName(null)
    // Reset extraction state
    setRdeExtractionState({ status: 'idle', progress: '', step: 0, totalSteps: 0, error: undefined })
    setExtractedRefContext(null)
    toast.success('Form reset to defaults')
  }

  // Handle selecting a job to reuse its contexts
  const handleReuseJob = useCallback(async (jobId: Id<"jobs">) => {
    const job = completedJobs?.find(j => j._id === jobId)
    if (!job) return

    setReusedFrom(jobId)
    setContextsValid(null)
    setSourceConfig(null)

    const pythonApiUrl = PYTHON_API_URL

    // Check contexts and load config in parallel
    if (job.jobDir) {
      setCheckingContexts(true)
      try {
        const [contextsRes, configRes] = await Promise.all([
          fetch(`${pythonApiUrl}/api/check-contexts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jobDir: job.jobDir }),
          }),
          fetch(`${pythonApiUrl}/api/load-job-config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jobDir: job.jobDir }),
          }),
        ])

        if (contextsRes.ok) {
          const contextsResult = await contextsRes.json()
          setContextsValid(contextsResult.valid)
        } else {
          setContextsValid(false)
        }

        if (configRes.ok) {
          const configResult = await configRes.json()
          if (configResult.config) {
            // Include referenceDataset from the job (stored in Convex, not in config file)
            setSourceConfig({
              ...configResult.config,
              referenceDataset: job.referenceDataset || undefined,
            })
          }
        }
      } catch {
        setContextsValid(false)
      } finally {
        setCheckingContexts(false)
      }
    }

    toast.success(`Reusing contexts from "${job.name}"`)
  }, [completedJobs])

  // Clear job reuse
  const clearReuse = () => {
    setReusedFrom(null)
    setContextsValid(null)
    setSourceConfig(null)
    toast.info('Reuse cleared')
  }

  const updateConfig = (path: string, value: any) => {
    setConfig((prev: typeof DEFAULT_CONFIG) => {
      const parts = path.split('.')
      const newConfig = { ...prev }
      let obj: any = newConfig
      for (let i = 0; i < parts.length - 1; i++) {
        obj[parts[i]] = { ...obj[parts[i]] }
        obj = obj[parts[i]]
      }
      obj[parts[parts.length - 1]] = value
      return newConfig
    })
  }

  // Apply preset to config
  const applyPreset = useCallback((preset: LLMPreset) => {
    const validModelIds = new Set(processedModels.map(m => m.id))
    let appliedCount = 0

    // Apply RDE model if valid
    if (preset.rdeModel && validModelIds.has(preset.rdeModel)) {
      setRdeModel(preset.rdeModel)
      appliedCount++
    }

    // Apply MAV models if valid (cap at 3 to match UI slots)
    if (preset.mavModels && preset.mavModels.length > 0) {
      const validatedMavModels = preset.mavModels
        .slice(0, 3)
        .map(m => m && validModelIds.has(m) ? m : '')
      // Ensure we have 3 slots (pad with empty strings)
      while (validatedMavModels.length < 3) {
        validatedMavModels.push('')
      }
      updateConfig('subject_profile.mav.models', validatedMavModels)
      appliedCount += validatedMavModels.filter(m => m).length
    }

    // Apply SAV model (for when MAV is disabled - uses first MAV slot)
    if (preset.savModel && validModelIds.has(preset.savModel)) {
      // SAV model goes into the first MAV model slot when MAV is disabled
      const currentMavModels = config.subject_profile?.mav?.models || ['', '', '']
      // Only apply if no MAV models are already set
      if (!preset.mavModels || preset.mavModels.every(m => !m)) {
        updateConfig('subject_profile.mav.models', [preset.savModel, currentMavModels[1] || '', currentMavModels[2] || ''])
        appliedCount++
      }
    }

    // Apply generation model if valid
    if (preset.genModel && validModelIds.has(preset.genModel)) {
      updateConfig('generation.model', preset.genModel)
      updateConfig('generation.provider', preset.genModel.split('/')[0])
      // Also update first slot in models array if multi-model is active
      if (config.generation.models?.length > 0) {
        const updated = [...config.generation.models]
        updated[0] = preset.genModel
        updateConfig('generation.models', updated)
      }
      appliedCount++
    }

    if (appliedCount === 0) {
      toast.warning(`Preset "${preset.name}" has no valid models`)
    }
  }, [processedModels, config.subject_profile?.mav?.models, updateConfig])

  // Handler for preset selection
  const handlePresetSelect = useCallback((preset: LLMPreset) => {
    setSelectedPresetId(preset._id)
    applyPreset(preset)
  }, [applyPreset])

  const handlePresetClear = useCallback(() => {
    setSelectedPresetId(null)
  }, [])

  // Auto-apply default preset on mount (only once when data loads)
  useEffect(() => {
    if (!presetApplied && defaultPreset && processedModels.length > 0 && !modelsLoading) {
      applyPreset(defaultPreset)
      setSelectedPresetId(defaultPreset._id)
      setPresetApplied(true)
    }
  }, [defaultPreset, processedModels.length, modelsLoading, presetApplied, applyPreset])

  // Real-time model validation with debounce
  const validateModelRealtime = useCallback(async (modelKey: string, modelId: string) => {
    if (!modelId || !settings?.openrouterApiKey) {
      setModelValidations(prev => {
        const next = { ...prev }
        delete next[modelKey]
        return next
      })
      return
    }

    // Skip if already validated for this exact model ID
    setModelValidations(prev => {
      const current = prev[modelKey]
      if (current?.modelId === modelId && (current.status === 'valid' || current.status === 'checking')) {
        return prev // Already validated or validating this model
      }
      return {
        ...prev,
        [modelKey]: { status: 'checking', modelId }
      }
    })

    // Check current state - if we didn't update to checking, skip the API call
    // (This is a workaround since we can't access the updated state directly)

    try {
      // Check if this is a specialized model that shouldn't be validated with text
      const modelLower = modelId.toLowerCase()
      const isAudioOnly = modelLower.includes('audio') && !modelLower.includes('omni')
      const isImageGen = modelLower.includes('dall-e') || modelLower.includes('image-gen')
      const isSpeechModel = modelLower.includes('tts') || modelLower.includes('speech')

      if (isAudioOnly || isImageGen || isSpeechModel) {
        // Mark as error - these models can't be used for text review generation
        setModelValidations(prev => ({
          ...prev,
          [modelKey]: {
            status: 'error',
            error: 'This model only supports audio/image - select a text model for review generation',
            modelId
          }
        }))
        return
      }

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${settings.openrouterApiKey}`,
        },
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: 'Say OK' }],
          // GPT-5 models require min 16 tokens
          max_tokens: 16,
        }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        // Extract the most useful error message
        let errorMsg = 'Model unavailable'

        if (data?.error?.message) {
          errorMsg = data.error.message
        } else if (data?.error?.metadata?.raw) {
          // Try to parse raw error for more details
          try {
            const rawError = typeof data.error.metadata.raw === 'string'
              ? JSON.parse(data.error.metadata.raw)
              : data.error.metadata.raw
            errorMsg = rawError?.error?.message || rawError?.message || String(data.error.metadata.raw).substring(0, 100)
          } catch {
            errorMsg = String(data.error.metadata.raw).substring(0, 100)
          }
        } else if (response.status === 402) {
          errorMsg = 'Insufficient credits or payment required'
        } else if (response.status === 429) {
          errorMsg = 'Rate limited - try again later'
        } else if (response.status === 503) {
          errorMsg = 'Model temporarily unavailable'
        }

        setModelValidations(prev => ({
          ...prev,
          [modelKey]: { status: 'error', error: errorMsg, modelId }
        }))
      } else {
        setModelValidations(prev => ({
          ...prev,
          [modelKey]: { status: 'valid', modelId }
        }))
      }
    } catch (err) {
      setModelValidations(prev => ({
        ...prev,
        [modelKey]: { status: 'error', error: err instanceof Error ? err.message : 'Validation failed', modelId }
      }))
    }
  }, [settings?.openrouterApiKey])

  // Debounced validation triggers for each model
  useEffect(() => {
    if (!settings?.openrouterApiKey) return

    // Get models to validate
    const modelsToValidate: Array<{ key: string; id: string }> = []

    // MAV models (if not reusing and MAV/SIL enabled)
    if (!reusedFrom && config.ablation.sil_enabled) {
      if (config.ablation.mav_enabled) {
        config.subject_profile.mav.models.forEach((modelId: string, index: number) => {
          if (modelId) {
            modelsToValidate.push({ key: `mav-${index}`, id: modelId })
          }
        })
      } else {
        // Just the first model when MAV disabled
        if (config.subject_profile.mav.models[0]) {
          modelsToValidate.push({ key: 'mav-0', id: config.subject_profile.mav.models[0] })
        }
      }
    }

    // Generation model(s) — validate all in multi-model mode
    if (config.generation.models?.length > 0) {
      config.generation.models.forEach((modelId: string, index: number) => {
        if (modelId) {
          modelsToValidate.push({ key: `generation-${index}`, id: modelId })
        }
      })
    } else if (config.generation.model) {
      modelsToValidate.push({ key: 'generation', id: config.generation.model })
    }

    // Debounce validation - wait 1.5 seconds after last change
    // Only validate models that actually changed
    const timeoutIds: NodeJS.Timeout[] = []

    modelsToValidate.forEach(({ key, id }) => {
      const current = modelValidations[key]

      // Skip if already checking this model
      if (current?.status === 'checking' && current?.modelId === id) return

      // Skip if already validated this exact model (valid or error)
      if (current?.modelId === id && (current.status === 'valid' || current.status === 'error')) return

      // Model changed or never validated - schedule validation
      const timeoutId = setTimeout(() => {
        validateModelRealtime(key, id)
      }, 1500)
      timeoutIds.push(timeoutId)
    })

    // Cleanup unused validations (models that were removed)
    const activeKeys = new Set(modelsToValidate.map(m => m.key))
    setModelValidations(prev => {
      const next = { ...prev }
      let changed = false
      for (const key of Object.keys(next)) {
        if (!activeKeys.has(key)) {
          delete next[key]
          changed = true
        }
      }
      return changed ? next : prev
    })

    return () => {
      timeoutIds.forEach(id => clearTimeout(id))
    }
  }, [
    config.subject_profile.mav.models,
    config.generation.model,
    config.generation.models,
    config.ablation.sil_enabled,
    config.ablation.mav_enabled,
    reusedFrom,
    settings?.openrouterApiKey,
    validateModelRealtime,
    modelValidations, // Need to include this to check current state
  ])

  // Check if any models have validation errors
  const hasInvalidModels = Object.values(modelValidations).some(v => v.status === 'error')
  const hasCheckingModels = Object.values(modelValidations).some(v => v.status === 'checking')
  const invalidModelsList = Object.entries(modelValidations)
    .filter(([, v]) => v.status === 'error')
    .map(([key, v]) => {
      const label = key.startsWith('mav-') ? `MAV Model ${parseInt(key.split('-')[1]) + 1}`
        : key.startsWith('generation-') ? `Generation Model ${parseInt(key.split('-')[1]) + 1}`
        : 'Generation Model'
      return { label, error: v.error || 'Model unavailable' }
    })

  // Validate a single model by making a test API call
  const validateModel = async (modelId: string, apiKey: string): Promise<{ model: string; success: boolean; error?: string }> => {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: modelId,
          messages: [{ role: 'user', content: 'Say OK' }],
          // GPT-5 models require min 16 tokens
          max_tokens: 16,
        }),
      })

      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        const errorMsg = data?.error?.message || data?.error?.metadata?.raw || `HTTP ${response.status}`
        return { model: modelId, success: false, error: errorMsg }
      }

      return { model: modelId, success: true }
    } catch (err) {
      return { model: modelId, success: false, error: err instanceof Error ? err.message : 'Unknown error' }
    }
  }

  // Validate all models that will be used (MAV models + generation model)
  const validateAllModels = async (): Promise<{ valid: boolean; failures: Array<{ model: string; error: string }> }> => {
    if (!settings?.openrouterApiKey) {
      return { valid: false, failures: [{ model: 'API Key', error: 'OpenRouter API key not configured' }] }
    }

    const modelsToValidate: string[] = []

    // Add MAV models if enabled and not reusing
    if (!reusedFrom && config.ablation.mav_enabled) {
      modelsToValidate.push(...config.subject_profile.mav.models.filter((m: string) => m))
    } else if (!reusedFrom && config.ablation.sil_enabled) {
      // Single model for SIL when MAV disabled
      if (config.subject_profile.mav.models[0]) {
        modelsToValidate.push(config.subject_profile.mav.models[0])
      }
    }

    // Add generation model
    if (config.generation.model) {
      modelsToValidate.push(config.generation.model)
    }

    // Remove duplicates
    const uniqueModels = [...new Set(modelsToValidate)]

    if (uniqueModels.length === 0) {
      return { valid: true, failures: [] }
    }

    // Validate all models in parallel
    const results = await Promise.all(
      uniqueModels.map(model => validateModel(model, settings.openrouterApiKey!))
    )

    const failures = results
      .filter(r => !r.success)
      .map(r => ({ model: r.model, error: r.error || 'Unknown error' }))

    return { valid: failures.length === 0, failures }
  }

  // Build config for job creation
  const buildTransformedConfig = () => {
    // Use 3-way sex distribution values (male/female/unknown -> male/female/unspecified for schema)
    const sexDist = config.reviewer_profile.sex_distribution

    // For GEN+EVAL only mode (reusing composition), include subject_profile from source job
    // This ensures aspect_categories are preserved in the new job's config
    const includeSubjectProfile = selectedPhases.includes('composition') ||
      (reusedFrom && sourceConfig?.subject_profile)

    return {
      subject_profile: includeSubjectProfile ? (
        selectedPhases.includes('composition') ? {
          query: config.subject_profile.query,
          additional_context: config.subject_profile.additional_context || undefined,
          region: config.subject_profile.region,
          domain: config.subject_profile.domain,
          aspect_category_mode: config.subject_profile.aspect_category_mode,
          aspect_categories: (config.subject_profile.aspect_category_mode === 'preset' || config.subject_profile.aspect_category_mode === 'ref_dataset') && config.subject_profile.aspect_categories.length > 0 ? config.subject_profile.aspect_categories : undefined,
          sentiment_depth: config.subject_profile.sentiment_depth,
          mav: {
            enabled: config.ablation.mav_enabled,
            models: config.subject_profile.mav.models.filter((m: string) => m),
            similarity_threshold: config.subject_profile.mav.similarity_threshold,
            max_queries: config.subject_profile.mav.max_queries,
          },
        } : {
          // GEN+EVAL only mode: use sourceConfig's subject_profile
          query: sourceConfig?.subject_profile?.query || '',
          additional_context: sourceConfig?.subject_profile?.additional_context || undefined,
          region: sourceConfig?.subject_profile?.region || '',
          domain: sourceConfig?.subject_profile?.domain || undefined,
          aspect_category_mode: sourceConfig?.subject_profile?.aspect_category_mode || 'infer',
          aspect_categories: sourceConfig?.subject_profile?.aspect_categories?.length ? sourceConfig.subject_profile.aspect_categories : undefined,
          sentiment_depth: sourceConfig?.subject_profile?.sentiment_depth || 'praise and complain',
          mav: sourceConfig?.subject_profile?.mav || undefined,
        }
      ) : undefined,
      reviewer_profile: (selectedPhases.includes('composition') || selectedPhases.includes('generation')) ? {
        age_range: config.ablation.age_enabled ? config.reviewer_profile.age_range : null,
        sex_distribution: {
          male: sexDist.male ?? 0.5,
          female: sexDist.female ?? 0.5,
          unspecified: sexDist.unknown ?? 0,
        },
        additional_context: config.reviewer_profile.additional_context || undefined,
        persona_ratio: config.reviewer_profile.persona_ratio ?? 0.9,
      } : undefined,
      attributes_profile: (selectedPhases.includes('composition') || selectedPhases.includes('generation')) ? {
        polarity: config.attributes_profile.polarity,
        noise: {
          typo_rate: config.attributes_profile.noise.typo_rate,
          colloquialism: config.attributes_profile.noise.colloquialism,
          grammar_errors: config.attributes_profile.noise.grammar_errors,
          preset: config.attributes_profile.noise.preset,
        },
        length_range: config.attributes_profile.length_range,
        temperature_range: config.attributes_profile.temperature_range || undefined,
        cap_weights: config.attributes_profile.cap_weights,
        edge_lengths: config.attributes_profile.edge_lengths,
      } : undefined,
      generation: selectedPhases.includes('generation') ? (() => {
        const firstTarget = config.generation.targets[0]
        return {
          ...config.generation,
          // Filter empty slots from multi-model array
          models: config.generation.models?.filter(Boolean)?.length > 0
            ? config.generation.models.filter(Boolean)
            : undefined,
          // Legacy fields derived from targets[0] for backward compat
          count_mode: firstTarget.count_mode,
          count: firstTarget.count_mode === 'reviews' ? firstTarget.target_value : Math.ceil(firstTarget.target_value / 5),
          target_sentences: firstTarget.count_mode === 'sentences' ? firstTarget.target_value : undefined,
          batch_size: firstTarget.batch_size,
          request_size: firstTarget.request_size,
          total_runs: firstTarget.total_runs,
          neb_enabled: firstTarget.neb_depth > 0,
          neb_depth: firstTarget.neb_depth,
        }
      })() : undefined,
      // Ablation settings for reproducibility
      ablation: {
        sil_enabled: config.ablation.sil_enabled,
        mav_enabled: config.ablation.mav_enabled,
        polarity_enabled: config.ablation.polarity_enabled,
        noise_enabled: config.ablation.noise_enabled,
        age_enabled: config.ablation.age_enabled,
        sex_enabled: config.ablation.sex_enabled,
      },
    }
  }

  // Helper to parse reviews from a file (JSONL, CSV, or SemEval XML)
  const parseReviewsFromFile = async (file: File): Promise<string[]> => {
    const text = await file.text()
    const ext = file.name.toLowerCase().split('.').pop()
    const reviews: string[] = []

    if (ext === 'jsonl') {
      // JSONL: Each line is a JSON object with a "review" or "text" field
      const lines = text.split('\n').filter(line => line.trim())
      for (const line of lines) {
        try {
          const obj = JSON.parse(line)
          const review = obj.review || obj.text || obj.content || obj.body
          if (review && typeof review === 'string') {
            reviews.push(review)
          }
        } catch {
          // Skip invalid lines
        }
      }
    } else if (ext === 'csv') {
      // CSV: First column or column named "review"/"text"
      const lines = text.split('\n')
      const header = lines[0]?.split(',').map(h => h.trim().toLowerCase().replace(/"/g, ''))
      const reviewIdx = header?.findIndex(h => ['review', 'text', 'content', 'body'].includes(h)) ?? 0
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',')
        const review = cols[reviewIdx]?.trim().replace(/^"|"$/g, '')
        if (review) {
          reviews.push(review)
        }
      }
    } else if (ext === 'xml') {
      // SemEval XML: Parse <text> elements from <sentence> or <Review> structures
      const parser = new DOMParser()
      const doc = parser.parseFromString(text, 'text/xml')

      // Try SemEval 2014/2015/2016 format: <sentences><sentence><text>...</text></sentence></sentences>
      const sentenceTexts = doc.querySelectorAll('sentence > text')
      if (sentenceTexts.length > 0) {
        sentenceTexts.forEach(el => {
          const review = el.textContent?.trim()
          if (review) reviews.push(review)
        })
      } else {
        // Try alternative format: <Reviews><Review><sentences>...</sentences></Review></Reviews>
        // Concatenate all sentences within a Review into one review text
        const reviewElements = doc.querySelectorAll('Review')
        if (reviewElements.length > 0) {
          reviewElements.forEach(reviewEl => {
            const texts = reviewEl.querySelectorAll('sentence > text')
            const combinedText = Array.from(texts)
              .map(el => el.textContent?.trim())
              .filter(Boolean)
              .join(' ')
            if (combinedText) reviews.push(combinedText)
          })
        } else {
          // Fallback: Just get all <text> elements
          const allTexts = doc.querySelectorAll('text')
          allTexts.forEach(el => {
            const review = el.textContent?.trim()
            if (review) reviews.push(review)
          })
        }
      }
    }

    return reviews
  }

  // Helper to parse full reviews with sentences and opinions for RDE extraction
  const parseFullReviewsFromFile = async (file: File): Promise<Array<{ id: string; sentences: Array<{ id: string; text: string; opinions: Array<{ category: string; polarity: string; target: string }> }> }>> => {
    const text = await file.text()
    const ext = file.name.toLowerCase().split('.').pop()
    const reviews: Array<{ id: string; sentences: Array<{ id: string; text: string; opinions: Array<{ category: string; polarity: string; target: string }> }> }> = []

    if (ext === 'xml') {
      const parser = new DOMParser()
      const doc = parser.parseFromString(text, 'text/xml')

      // Try SemEval format: <Reviews><Review><sentences><sentence>...</sentence></sentences></Review></Reviews>
      const reviewElements = doc.querySelectorAll('Review')
      if (reviewElements.length > 0) {
        reviewElements.forEach((reviewEl, idx) => {
          const sentences: Array<{ id: string; text: string; opinions: Array<{ category: string; polarity: string; target: string }> }> = []
          const sentenceEls = reviewEl.querySelectorAll('sentence')
          sentenceEls.forEach((sentenceEl, sIdx) => {
            const textEl = sentenceEl.querySelector('text')
            const opinionEls = sentenceEl.querySelectorAll('Opinion')
            const opinions = Array.from(opinionEls).map(opEl => ({
              category: opEl.getAttribute('category') || '',
              polarity: opEl.getAttribute('polarity') || '',
              target: opEl.getAttribute('target') || '',
            }))
            sentences.push({
              id: sentenceEl.getAttribute('id') || `s${sIdx}`,
              text: textEl?.textContent?.trim() || '',
              opinions,
            })
          })
          reviews.push({
            id: reviewEl.getAttribute('rid') || `r${idx}`,
            sentences,
          })
        })
      } else {
        // Fallback: <sentences><sentence>...</sentence></sentences> (flat structure)
        const sentenceEls = doc.querySelectorAll('sentence')
        sentenceEls.forEach((sentenceEl, idx) => {
          const textEl = sentenceEl.querySelector('text')
          const opinionEls = sentenceEl.querySelectorAll('Opinion')
          const opinions = Array.from(opinionEls).map(opEl => ({
            category: opEl.getAttribute('category') || '',
            polarity: opEl.getAttribute('polarity') || '',
            target: opEl.getAttribute('target') || '',
          }))
          reviews.push({
            id: sentenceEl.getAttribute('id') || `r${idx}`,
            sentences: [{
              id: `s0`,
              text: textEl?.textContent?.trim() || '',
              opinions,
            }],
          })
        })
      }
    } else if (ext === 'jsonl') {
      const lines = text.split('\n').filter(line => line.trim())
      lines.forEach((line, idx) => {
        try {
          const obj = JSON.parse(line)
          if (obj.sentences && Array.isArray(obj.sentences)) {
            reviews.push({
              id: obj.id || `r${idx}`,
              sentences: obj.sentences.map((s: any, sIdx: number) => ({
                id: s.id || `s${sIdx}`,
                text: s.text || '',
                opinions: (s.opinions || []).map((o: any) => ({
                  category: o.category || '',
                  polarity: o.polarity || '',
                  target: o.target || '',
                })),
              })),
            })
          } else {
            // Simple format with just text
            const reviewText = obj.review || obj.text || obj.content || ''
            reviews.push({
              id: obj.id || `r${idx}`,
              sentences: [{ id: 's0', text: reviewText, opinions: [] }],
            })
          }
        } catch { /* skip */ }
      })
    }

    return reviews
  }

  // Handle RDE extraction
  const handleRdeExtraction = async () => {
    if (!referenceDatasetFile || !rdeModel) {
      toast.error('Please select an RDE model and upload a reference dataset')
      return
    }

    if (!settings?.openrouterApiKey) {
      toast.error('OpenRouter API key required. Please set it in Settings.')
      return
    }

    setRdeExtractionState({ status: 'extracting', progress: 'Parsing dataset...', step: 0, totalSteps: 6 })

    try {
      // Parse the full reviews
      const reviews = await parseFullReviewsFromFile(referenceDatasetFile)
      if (reviews.length === 0) {
        throw new Error('No reviews found in the dataset')
      }

      setRdeExtractionState({ status: 'extracting', progress: 'Sending to extraction API...', step: 1, totalSteps: 6 })

      // Call the extract-ref-context endpoint
      const pythonApiUrl = PYTHON_API_URL
      const response = await fetch(`${pythonApiUrl}/api/extract-ref-context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviews,
          model: rdeModel,
          apiKey: settings.openrouterApiKey,
          sampleCount: referenceOptions.sampleCount,
        }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const result = await response.json()

      if (result.error) {
        throw new Error(result.error)
      }

      // Cache the result
      const hash = await computeFileHash(referenceDatasetFile)
      const existingCache = localStorage.getItem(REF_CONTEXT_CACHE_KEY)
      const cacheData = existingCache ? JSON.parse(existingCache) : { datasets: {} }
      cacheData.datasets[hash] = {
        filename: referenceDatasetFile.name,
        fileSize: referenceDatasetFile.size,
        extractedAt: new Date().toISOString(),
        model: rdeModel,
        context: result,
      }
      cacheData.current = hash
      localStorage.setItem(REF_CONTEXT_CACHE_KEY, JSON.stringify(cacheData))

      // Set the extracted context
      setExtractedRefContext(result)
      // Store RDE token usage for forwarding to pipeline
      if (result.rde_usage) {
        setRdeUsage(result.rde_usage)
      }
      setRdeExtractionState({ status: 'success', progress: 'Extraction complete', step: 6, totalSteps: 6 })

      // Auto-apply extracted values to config (don't rely on child effects)
      if (result.aspect_categories?.length > 0) {
        updateConfig('subject_profile.aspect_category_mode', 'ref_dataset')
        updateConfig('subject_profile.aspect_categories', result.aspect_categories)
      }

      // Auto-apply extracted sex distribution
      if (result.sex_distribution) {
        updateConfig('reviewer_profile.sex_distribution', {
          male: result.sex_distribution.male,
          female: result.sex_distribution.female,
          unknown: result.sex_distribution.unknown,
        })
      }

      // Disable age since age detection from text is unreliable
      updateConfig('ablation.age_enabled', false)

      toast.success('Reference context extracted successfully')
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setRdeExtractionState({
        status: 'error',
        progress: errorMessage,
        step: 0,
        totalSteps: 6,
        error: errorMessage,
      })
      toast.error(`Extraction failed: ${errorMessage}`)
    }
  }

  // Helper to calculate cost estimates for job creation
  const calculateCostEstimates = () => {
    // Default pricing if model not found (conservative estimate)
    const defaultPricing = { input: 2.0, output: 8.0 }

    // Get pricing for selected generation model
    const getModelPricingLocal = (modelId: string): { input: number; output: number } => {
      // Local vLLM models have zero cost
      if (modelId.startsWith('local/')) return { input: 0, output: 0 }
      const model = rawModels.find((m) => m.id === modelId)
      if (!model) return defaultPricing
      const inputPrice = parseFloat(model.pricing?.prompt || '0') * 1_000_000
      const outputPrice = parseFloat(model.pricing?.completion || '0') * 1_000_000
      return { input: inputPrice, output: outputPrice }
    }

    // Count MAV models
    const mavModels = config.subject_profile.mav.models.filter((m: string) => m)
    const M = config.ablation.mav_enabled ? mavModels.length : 1

    // Calculate review count based on mode
    const sentencesPerReview = config.attributes_profile?.length_range
      ? (config.attributes_profile.length_range[0] + config.attributes_profile.length_range[1]) / 2
      : 3.5

    let N: number
    if (config.generation.count_mode === 'sentences') {
      const targetSentences = config.generation.target_sentences || 1000
      N = Math.ceil(targetSentences / sentencesPerReview)
    } else {
      N = config.generation.count
    }

    // Calculate SIL cost (composition phase) - 3M + 1 calls
    let compositionCost = 0
    let compositionCalls = 0
    let compositionTokens = 0

    if (!reusedFrom && selectedPhases.includes('composition')) {
      for (const mavModelId of (mavModels.length > 0 ? mavModels : [config.generation.model])) {
        const pricing = getModelPricingLocal(mavModelId)

        // Round 1: Research
        const researchCost = (TOKEN_ESTIMATES.sil_research.input * pricing.input + TOKEN_ESTIMATES.sil_research.output * pricing.output) / 1_000_000
        // Round 1: Query Generation
        const queryGenCost = (TOKEN_ESTIMATES.sil_generate_queries.input * pricing.input + TOKEN_ESTIMATES.sil_generate_queries.output * pricing.output) / 1_000_000
        // Round 3: Answer Queries
        const answerCost = (TOKEN_ESTIMATES.sil_answer_queries.input * pricing.input + TOKEN_ESTIMATES.sil_answer_queries.output * pricing.output) / 1_000_000

        compositionCost += researchCost + queryGenCost + answerCost
        compositionTokens += TOKEN_ESTIMATES.sil_research.input + TOKEN_ESTIMATES.sil_research.output
        compositionTokens += TOKEN_ESTIMATES.sil_generate_queries.input + TOKEN_ESTIMATES.sil_generate_queries.output
        compositionTokens += TOKEN_ESTIMATES.sil_answer_queries.input + TOKEN_ESTIMATES.sil_answer_queries.output
      }

      // Classification (1 call using first model)
      const classifyModel = mavModels[0] || config.generation.model
      const classifyPricing = getModelPricingLocal(classifyModel)
      const classifyCost = (TOKEN_ESTIMATES.sil_classify.input * classifyPricing.input + TOKEN_ESTIMATES.sil_classify.output * classifyPricing.output) / 1_000_000
      compositionCost += classifyCost
      compositionTokens += TOKEN_ESTIMATES.sil_classify.input + TOKEN_ESTIMATES.sil_classify.output

      compositionCalls = 3 * M + 1
    }

    // Calculate AML cost (generation phase) - N calls × R runs
    let generationCost = 0
    let generationCalls = 0
    let generationTokens = 0
    const totalRuns = config.generation.total_runs || 1

    if (selectedPhases.includes('generation')) {
      const modelPricing = getModelPricingLocal(config.generation.model)
      const amlInputTokens = N * TOKEN_ESTIMATES.aml_per_review.input * totalRuns
      const amlOutputTokens = N * TOKEN_ESTIMATES.aml_per_review.output * totalRuns
      generationCost = (amlInputTokens * modelPricing.input + amlOutputTokens * modelPricing.output) / 1_000_000
      generationCalls = N * totalRuns
      generationTokens = amlInputTokens + amlOutputTokens
    }

    // Calculate RDE cost (one-time extraction from reference dataset)
    let rdeCost = 0
    let rdeCalls = 0
    let rdeTokens = 0

    if (rdeModel && rdeExtractionState.status === 'success') {
      // RDE extraction cost - uses one API call
      const rdePricing = getModelPricingLocal(rdeModel)
      const rdeInputTokens = 3000 // Estimated: system prompt + sample reviews
      const rdeOutputTokens = 2000 // Estimated: extraction response
      rdeCost = (rdeInputTokens * rdePricing.input + rdeOutputTokens * rdePricing.output) / 1_000_000
      rdeCalls = 1
      rdeTokens = rdeInputTokens + rdeOutputTokens
    }

    // Calculate prompt/completion token breakdown for generation
    const genPromptTokens = selectedPhases.includes('generation')
      ? N * TOKEN_ESTIMATES.aml_per_review.input * totalRuns
      : 0
    const genCompletionTokens = selectedPhases.includes('generation')
      ? N * TOKEN_ESTIMATES.aml_per_review.output * totalRuns
      : 0

    return {
      rde: {
        cost: rdeCost,
        calls: rdeCalls,
      },
      composition: {
        cost: compositionCost,
        calls: compositionCalls,
        tokens: compositionTokens,
      },
      generation: {
        cost: generationCost,
        calls: generationCalls,
        tokens: generationTokens,
        promptTokens: genPromptTokens,
        completionTokens: genCompletionTokens,
      },
      evaluation: {
        cost: 0, // Evaluation uses local metrics, no API calls
        calls: 0,
      },
      total: {
        cost: rdeCost + compositionCost + generationCost,
        calls: rdeCalls + compositionCalls + generationCalls,
        tokens: rdeTokens + compositionTokens + generationTokens,
        promptTokens: genPromptTokens,
        completionTokens: genCompletionTokens,
      },
    }
  }

  // Handle "Create Job" - only runs RGM+ACM (instant, no LLM)
  const handleCreateJob = async (andRunPipeline = false) => {
    // Run validation checks
    const errors = getValidationErrors()
    if (errors.length > 0) {
      setValidationErrors(errors)
      setShowValidationErrors(true)
      return
    }

    setSubmitting(true)
    try {
      const transformedConfig = buildTransformedConfig()
      const pythonApiUrl = PYTHON_API_URL

      // Context extraction: Skip if already extracted via RDE, otherwise extract if enabled
      let extractedSubjectContext: string | null = null
      let extractedReviewerContext: string | null = null

      // Only run legacy extraction if RDE context is NOT available and extraction is enabled
      const hasRdeContext = extractedRefContext && (extractedRefContext.additional_context || extractedRefContext.reviewer_context)

      if (!hasRdeContext && referenceDatasetFile && selectedPhases.includes('composition')) {
        const shouldExtractSubject = referenceOptions.extractSubjectContext
        const shouldExtractReviewer = referenceOptions.extractReviewerContext

        if (shouldExtractSubject || shouldExtractReviewer) {
          try {
            toast.info('Extracting context from reference dataset...')

            // Parse reviews from the file
            const reviews = await parseReviewsFromFile(referenceDatasetFile)

            if (reviews.length === 0) {
              toast.warning('No reviews found in reference file. Skipping context extraction.')
            } else {
              // Get the first MAV model for extraction
              const extractionModel = config.subject_profile?.mav?.models?.[0] || 'anthropic/claude-sonnet-4'

              // Call the extraction API
              const response = await fetch(`${pythonApiUrl}/api/extract-context`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  reviews,
                  extractSubject: shouldExtractSubject,
                  extractReviewer: shouldExtractReviewer,
                  model: extractionModel,
                  apiKey: settings?.openrouterApiKey || '',
                  sampleCount: referenceOptions.sampleCount,
                }),
              })

              const result = await response.json()

              if (result.error) {
                toast.error(`Context extraction failed: ${result.error}`)
              } else {
                extractedSubjectContext = result.subject_context
                extractedReviewerContext = result.reviewer_context

                // Append extracted context to additional_context fields
                if (extractedSubjectContext && transformedConfig.subject_profile) {
                  const existingContext = transformedConfig.subject_profile.additional_context || ''
                  const separator = existingContext ? '\n\n--- Extracted from reference ---\n' : ''
                  transformedConfig.subject_profile.additional_context = existingContext + separator + extractedSubjectContext
                }

                if (extractedReviewerContext && transformedConfig.reviewer_profile) {
                  const existingContext = transformedConfig.reviewer_profile.additional_context || ''
                  const separator = existingContext ? '\n\n--- Extracted from reference ---\n' : ''
                  transformedConfig.reviewer_profile.additional_context = existingContext + separator + extractedReviewerContext
                }

                // Store RDE usage from legacy extraction too
                if (result.rde_usage) {
                  setRdeUsage(result.rde_usage)
                }
                toast.success(`Context extracted from ${result.sample_count} reviews`)
              }
            }
          } catch (e) {
            console.error('Context extraction error:', e)
            toast.warning('Context extraction failed. Continuing without extracted context.')
          }
        }
      }

      // Build referenceDataset configuration for storage
      const referenceDatasetConfig = referenceDatasetFile && selectedPhases.includes('composition') ? {
        fileName: referenceDatasetFile.name,
        useForEvaluation: referenceOptions.useForEvaluation,
        extractedSubjectContext: !!extractedSubjectContext,
        extractedReviewerContext: !!extractedReviewerContext,
      } : (!selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation) ? {
        ...sourceConfig.referenceDataset,
      } : undefined

      // Create the job in Convex
      // Calculate cost estimates for storage
      const estimatedCost = calculateCostEstimates()

      // Debug logging for multi-run and multi-model
      console.log('[CreateJob] DEBUG: transformedConfig.generation =', transformedConfig.generation)
      console.log('[CreateJob] DEBUG: total_runs =', transformedConfig.generation?.total_runs)
      console.log('[CreateJob] DEBUG: models =', transformedConfig.generation?.models)
      console.log('[CreateJob] DEBUG: parallel_models =', transformedConfig.generation?.parallel_models)

      const jobId = await createJob({
        name: config.name,
        config: transformedConfig,
        phases: selectedPhases,
        evaluationConfig: selectedPhases.includes('evaluation') ? {
          metrics: config.evaluation.metrics,
          reference_metrics_enabled: selfTest.enabled || (referenceOptions.useForEvaluation && !!referenceDatasetFile) || (!selectedPhases.includes('composition') && reusedFrom && !!sourceConfig?.referenceDataset?.useForEvaluation),
          reference_file: selfTest.enabled ? undefined : (referenceDatasetFile && referenceOptions.useForEvaluation ? referenceDatasetFile.name : (!selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation ? sourceConfig.referenceDataset.fileName : undefined)),
          self_test: selfTest.enabled ? { enabled: true, split_mode: selfTest.splitMode } : undefined,
          targets: method === 'real' && isEvalOnly ? realTargets : undefined,
        } : undefined,
        datasetFile: datasetFile ? datasetFile.name : undefined,
        reusedFrom: reusedFrom || undefined,
        referenceDataset: referenceDatasetConfig,
        estimatedCost,
        method,
        rdeUsage: rdeUsage || undefined,
      })

      // Upload dataset file if present (EVAL-only jobs)
      if (datasetFile && selectedPhases.includes('evaluation')) {
        try {
          const formData = new FormData()
          formData.append('file', datasetFile)
          formData.append('jobId', jobId)
          formData.append('jobName', config.name)
          await fetch(`${pythonApiUrl}/api/upload-dataset`, {
            method: 'POST',
            body: formData,
          })
        } catch {
          console.warn('Dataset upload failed - may need manual upload')
        }
      }

      // Upload reference dataset file if present (for Lexical/Semantic metrics) — skip during self test
      if (referenceDatasetFile && selectedPhases.includes('evaluation') && !selfTest.enabled && referenceOptions.useForEvaluation) {
        try {
          const formData = new FormData()
          formData.append('file', referenceDatasetFile)
          formData.append('jobId', jobId)
          formData.append('jobName', config.name)
          formData.append('fileType', 'reference')
          await fetch(`${pythonApiUrl}/api/upload-dataset`, {
            method: 'POST',
            body: formData,
          })
        } catch {
          console.warn('Reference dataset upload failed - may need manual upload')
        }
      }

      // Run RGM+ACM to create contexts (instant, no LLM)
      // Only if composition or generation phases are selected
      if (selectedPhases.includes('composition') || selectedPhases.includes('generation')) {
        try {
          await runContextsOnly({ jobId: jobId as Id<"jobs"> })
        } catch (e) {
          console.warn('Context creation failed:', e)
        }
      }

      // If "Create Job & Run Pipeline", trigger full pipeline
      if (andRunPipeline) {
        runPipeline({ jobId: jobId as Id<"jobs"> }).catch((error) => {
          console.error('Pipeline failed:', error)
        })
        toast.success('Job created and pipeline started!')
      } else {
        toast.success('Job created successfully!')
      }

      localStorage.removeItem(STORAGE_KEY)
      localStorage.removeItem(UI_STATE_KEY)
      setReusedFrom(null)
      navigate({ to: '/jobs/$jobId', params: { jobId } })
    } catch (error) {
      toast.error('Failed to create job')
      console.error(error)
    } finally {
      setSubmitting(false)
    }
  }

  const getValidationErrors = (): string[] => {
    const errors: string[] = []
    if (!config.name) {
      errors.push('Job name is required')
    }

    // COMPOSITION phase validation
    if (selectedPhases.includes('composition')) {
      if (!config.subject_profile.query) {
        errors.push('Subject Query is not set in COMPOSITION phase')
      }
      const mavEnabled = config.ablation.mav_enabled
      if (mavEnabled) {
        const selectedModels = config.subject_profile.mav.models.filter((m: string) => m).length
        if (selectedModels < 2) {
          errors.push(`At least 2 MAV models required (${selectedModels} selected) in Intelligence & Verification`)
        }
      } else {
        if (!config.subject_profile.mav.models[0]) {
          errors.push('Intelligence model is not selected in Intelligence & Verification')
        }
      }
    }

    // GENERATION phase validation
    if (selectedPhases.includes('generation')) {
      if (!settings?.openrouterApiKey) {
        errors.push('OpenRouter API key is not configured (check Settings)')
      }
      if (config.generation.models?.length > 0) {
        // Multi-model mode: check that at least one model is selected
        const selectedModels = config.generation.models.filter(Boolean).length
        if (selectedModels === 0) {
          errors.push('At least one generation model must be selected')
        }
      } else if (!config.generation.model) {
        errors.push('Generation LLM is not selected in GENERATION phase')
      }
      // Need composition source if not running composition phase
      if (!selectedPhases.includes('composition') && !reusedFrom) {
        errors.push('Select a job to reuse composition data from')
      }
    }

    // EVALUATION-only validation (no composition, no generation)
    if (selectedPhases.includes('evaluation') &&
        !selectedPhases.includes('generation') &&
        !selectedPhases.includes('composition')) {
      if (!datasetFile) {
        errors.push('Upload a dataset file for evaluation-only mode')
      }
    }

    return errors
  }

  // Get the reused job info for display
  const reusedJobInfo = reusedFrom ? completedJobs?.find(j => j._id === reusedFrom) : null

  // --- HEURISTIC WIZARD ---
  // If heuristic method is selected AND user clicked Next, render the heuristic wizard
  if (method === 'heuristic' && showHeuristicWizard) {
    return (
      <HeuristicWizard
        onBack={() => setShowHeuristicWizard(false)}
        onReset={() => {
          setMethod('cera')
          setShowHeuristicWizard(false)
          resetConfig()
        }}
      />
    )
  }

  // --- METHOD SELECTION + PHASE SELECTION SCREEN ---
  if (!showWizard) {
    const togglePhase = (phaseId: string) => {
      const current = config.selectedPhases as string[]
      if (current.includes(phaseId)) {
        updateConfig('selectedPhases', current.filter((p: string) => p !== phaseId))
      } else {
        updateConfig('selectedPhases', [...current, phaseId])
      }
    }

    const selectAll = () => {
      updateConfig('selectedPhases', PHASES.map(p => p.id))
    }

    return (
      <div className="flex flex-col gap-6 p-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Create a job</h1>
            <p className="text-muted-foreground">
              Choose a generation method and configure your synthetic ABSA dataset
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetConfig}
            className="text-muted-foreground"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>

        {/* Method Selection */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Generation Method</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {/* CERA Method */}
            <button
              onClick={() => setMethod('cera')}
              className={`relative p-6 rounded-xl border-2 text-left transition-all ${
                method === 'cera' ? 'ring-2 ring-offset-2 scale-[1.02]' : 'hover:scale-[1.01]'
              }`}
              style={{
                borderColor: method === 'cera' ? '#4e95d9' : 'transparent',
                backgroundColor: method === 'cera' ? '#e8f4fd' : '#f5f5f5',
                // @ts-ignore
                '--tw-ring-color': '#4e95d9',
              }}
            >
              <div className={`absolute top-3 right-3 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                method === 'cera' ? 'border-transparent bg-[#4e95d9]' : 'border-gray-300'
              }`}>
                {method === 'cera' && <CheckCircle className="h-4 w-4 text-white" />}
              </div>
              <h3 className="font-bold text-lg mb-1" style={{ color: '#4e95d9' }}>
                CERA Pipeline
              </h3>
              <p className="text-sm text-muted-foreground">
                Full 3-phase pipeline with SIL, MAV, RGM, and AML for context-engineered review generation
              </p>
            </button>

            {/* Heuristic Method */}
            <button
              onClick={() => setMethod('heuristic')}
              className={`relative p-6 rounded-xl border-2 text-left transition-all ${
                method === 'heuristic' ? 'ring-2 ring-offset-2 scale-[1.02]' : 'hover:scale-[1.01]'
              }`}
              style={{
                borderColor: method === 'heuristic' ? '#8b5cf6' : 'transparent',
                backgroundColor: method === 'heuristic' ? '#f3e8ff' : '#f5f5f5',
                // @ts-ignore
                '--tw-ring-color': '#8b5cf6',
              }}
            >
              <div className={`absolute top-3 right-3 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                method === 'heuristic' ? 'border-transparent bg-[#8b5cf6]' : 'border-gray-300'
              }`}>
                {method === 'heuristic' && <CheckCircle className="h-4 w-4 text-white" />}
              </div>
              <h3 className="font-bold text-lg mb-1" style={{ color: '#8b5cf6' }}>
                Heuristic Prompting
              </h3>
              <p className="text-sm text-muted-foreground">
                Simplified baseline for comparison — direct LLM prompting without CERA innovations
              </p>
            </button>
          </div>
        </div>

        {/* Phase Selection (always shown, but disabled when heuristic is selected) */}
        <Separator />
        <div className={method === 'heuristic' ? 'opacity-40 pointer-events-none' : ''}>
          <div className="space-y-3">
            <h2 className="text-lg font-semibold">Pipeline Phases</h2>
            <p className="text-sm text-muted-foreground">
              Select which phases to include in your CERA pipeline
            </p>
          </div>

          {/* Phase Selection Boxes */}
          <div className="grid gap-6 sm:grid-cols-3 mt-4">
            {PHASES.map((phase) => {
              const isSelected = (config.selectedPhases as string[]).includes(phase.id)
              return (
                <button
                  key={phase.id}
                  onClick={() => togglePhase(phase.id)}
                  disabled={method === 'heuristic'}
                  className={`relative p-6 rounded-xl border-2 text-left transition-all ${
                    isSelected ? 'ring-2 ring-offset-2 scale-[1.02]' : 'hover:scale-[1.01]'
                  }`}
                  style={{
                    borderColor: isSelected ? phase.strongColor : 'transparent',
                    backgroundColor: phase.lightColor,
                    // @ts-ignore
                    '--tw-ring-color': phase.strongColor,
                  }}
                >
                  {/* Selection indicator */}
                  <div className={`absolute top-3 right-3 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                    isSelected ? 'border-transparent' : 'border-gray-300'
                  }`} style={{ backgroundColor: isSelected ? phase.strongColor : 'transparent' }}>
                    {isSelected && <CheckCircle className="h-4 w-4 text-white" />}
                  </div>

                  <h3 className="font-bold text-lg mb-1" style={{ color: phase.strongColor }}>
                    {phase.title}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {phase.description}
                  </p>
                </button>
              )
            })}
          </div>
        </div>

        {/* Eval-only method selector — shown when only evaluation is selected */}
        {method !== 'heuristic' &&
          (config.selectedPhases as string[]).length === 1 &&
          (config.selectedPhases as string[]).includes('evaluation') && (
          <>
            <Separator />
            <div className="space-y-3">
              <h2 className="text-lg font-semibold">Dataset Method</h2>
              <p className="text-sm text-muted-foreground">
                What generated this dataset? This labels the job in the jobs list.
              </p>
              <div className="grid gap-3 sm:grid-cols-3">
                {([
                  { id: 'cera' as const, label: 'CERA', color: '#4e95d9', bg: '#e8f4fd', desc: 'Generated by CERA pipeline' },
                  { id: 'heuristic' as const, label: 'Heuristic', color: '#8b5cf6', bg: '#f3e8ff', desc: 'Generated by heuristic prompting' },
                  { id: 'real' as const, label: 'Real', color: '#16a34a', bg: '#f0fdf4', desc: 'Real-world dataset (ceiling)' },
                ] as const).map(opt => (
                  <button
                    key={opt.id}
                    onClick={() => setMethod(opt.id)}
                    className={`relative p-4 rounded-xl border-2 text-left transition-all ${
                      method === opt.id ? 'ring-2 ring-offset-2 scale-[1.02]' : 'hover:scale-[1.01]'
                    }`}
                    style={{
                      borderColor: method === opt.id ? opt.color : 'transparent',
                      backgroundColor: method === opt.id ? opt.bg : '#f5f5f5',
                      // @ts-ignore
                      '--tw-ring-color': opt.color,
                    }}
                  >
                    <div className={`absolute top-2 right-2 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-colors ${
                      method === opt.id ? 'border-transparent' : 'border-gray-300'
                    }`} style={{ backgroundColor: method === opt.id ? opt.color : 'transparent' }}>
                      {method === opt.id && <CheckCircle className="h-3 w-3 text-white" />}
                    </div>
                    <h3 className="font-bold mb-0.5" style={{ color: opt.color }}>{opt.label}</h3>
                    <p className="text-xs text-muted-foreground">{opt.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Select All + Continue buttons */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            onClick={selectAll}
            disabled={method === 'heuristic' || (config.selectedPhases as string[]).length === PHASES.length}
            className={method === 'heuristic' ? 'opacity-40' : ''}
          >
            Select All
          </Button>
          <Button
            onClick={() => {
              if (method === 'heuristic') {
                setShowHeuristicWizard(true)
              } else {
                setShowWizard(true)
                setActiveTab(0)
              }
            }}
            disabled={method !== 'heuristic' && (config.selectedPhases as string[]).length === 0}
          >
            Continue
            <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </div>
    )
  }

  // --- WIZARD VIEW ---
  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Create a job</h1>
          <p className="text-muted-foreground">
            Configure your synthetic ABSA review generation
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selectedPhases.includes('composition') && (
            <AblationSettingsDialog config={config} updateConfig={updateConfig} />
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { setShowWizard(false); setActiveTab(0) }}
            className="text-muted-foreground"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Phases
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetConfig}
            className="text-muted-foreground"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>
      </div>

      {/* API Key Warning */}
      {(selectedPhases.includes('composition') || selectedPhases.includes('generation')) && !settings?.openrouterApiKey && (
        <Alert variant="destructive" className="border-2 border-destructive bg-destructive/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>OpenRouter API Key Required</AlertTitle>
          <AlertDescription className="flex items-center justify-between">
            <span>You need to configure an OpenRouter API key.</span>
            <Button variant="destructive" size="sm" asChild className="ml-4 shrink-0">
              <Link to="/settings">
                <Settings className="mr-2 h-4 w-4" />
                Go to Settings
              </Link>
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Dynamic Tab Navigation */}
      <div className="flex items-center gap-3 overflow-x-auto py-2 px-1">
        {wizardTabs.map((tab, i) => {
          const isActive = activeTab === i
          const isPhaseTab = !!tab.color
          // Disable tabs beyond INPUT when RDE extraction is required but not complete
          const isTabDisabled = isNextDisabledByRde && i > 0
          return (
            <button
              key={tab.id}
              onClick={() => !isTabDisabled && setActiveTab(i)}
              disabled={isTabDisabled}
              title={isTabDisabled ? 'Extract reference context before continuing' : undefined}
              className={`px-4 py-2 text-sm font-medium transition-all whitespace-nowrap ${
                isTabDisabled
                  ? 'opacity-50 cursor-not-allowed'
                  : isPhaseTab
                    ? `rounded-lg ${isActive ? 'ring-2 ring-offset-2 scale-[1.02]' : 'hover:opacity-80'}`
                    : `rounded-lg ${isActive ? 'bg-muted border border-border' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'}`
              }`}
              style={isPhaseTab ? {
                backgroundColor: tab.lightColor,
                color: tab.color,
                // @ts-expect-error CSS custom property
                '--tw-ring-color': tab.color,
              } : undefined}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentTabId}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
        >
          {/* INPUT TAB */}
          {currentTabId === 'input' && (
            <div className="space-y-6">
              {/* Job Name - always shown */}
              <div className="space-y-2">
                <Label className="text-base">Job Name *</Label>
                <Input
                  placeholder="e.g., iPhone 15 Pro Reviews"
                  value={config.name}
                  onChange={(e) => updateConfig('name', e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  A descriptive name for this generation job
                </p>
              </div>

              {/* LLM Selection Preset */}
              {presets && presets.length > 0 && (
                <div className="p-4 rounded-lg bg-muted/30 border">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-sm font-medium">LLM Selection Preset</Label>
                      <p className="text-xs text-muted-foreground">
                        Quickly apply saved model configurations. <a href="/settings" className="underline hover:text-foreground">Manage presets in Settings</a>
                      </p>
                    </div>
                    <PresetSelector
                      presets={presets}
                      selectedPresetId={selectedPresetId}
                      onSelect={handlePresetSelect}
                      onClear={handlePresetClear}
                      processedModels={processedModels}
                      loading={modelsLoading}
                    />
                  </div>
                </div>
              )}

              {/* Self Test Toggle - shown in any mode with evaluation */}
              {selectedPhases.includes('evaluation') && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Activity className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <Label className="text-sm font-medium">Self Test</Label>
                        <p className="text-xs text-muted-foreground">
                          {selectedPhases.includes('generation')
                            ? 'Splits generated reviews into two halves and compares them against each other (no external reference needed)'
                            : 'Splits the uploaded dataset into two halves and compares them against each other'}
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={selfTest.enabled}
                      onCheckedChange={(checked) => {
                        setSelfTest(prev => ({ ...prev, enabled: checked }))
                        if (checked) {
                          // In EVAL-only mode, clear the reference file (it's replaced by self-test)
                          if (!selectedPhases.includes('generation') && !selectedPhases.includes('composition')) {
                            setReferenceDatasetFile(null)
                          }
                          // In all modes: uncheck "Use for MDQA evaluation comparison" (mutual exclusivity)
                          setReferenceOptions(prev => ({ ...prev, useForEvaluation: false }))
                        }
                      }}
                    />
                  </div>

                  {selfTest.enabled && (
                    <div className="space-y-2 pt-2 border-t">
                      <Label className="text-xs text-muted-foreground">Split Mode</Label>
                      <div className="flex gap-2">
                        <Button
                          type="button"
                          size="sm"
                          variant={selfTest.splitMode === 'random' ? 'default' : 'outline'}
                          className="flex-1"
                          onClick={() => setSelfTest(prev => ({ ...prev, splitMode: 'random' }))}
                        >
                          Random 50% Split
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant={selfTest.splitMode === 'sequential' ? 'default' : 'outline'}
                          className="flex-1"
                          onClick={() => setSelfTest(prev => ({ ...prev, splitMode: 'sequential' }))}
                        >
                          Normal 50% Split
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {selfTest.splitMode === 'random'
                          ? 'Randomly assigns each review to Set A or Set B (~50/50). Recommended for unbiased self-referencing metrics.'
                          : 'First half of reviews → Set A, second half → Set B. Useful when review order matters.'}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Real Dataset Target Sizes - shown for method="real" eval-only */}
              {method === 'real' && isEvalOnly && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <Label className="text-sm font-medium">Target Sizes</Label>
                      <p className="text-xs text-muted-foreground">
                        The source dataset will be subsampled at each target size (greedy review accumulation)
                      </p>
                    </div>
                  </div>
                  <div className="space-y-2 pt-2 border-t">
                    {realTargets.map((target, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <Input
                          type="number"
                          value={target.target_value}
                          onChange={(e) => {
                            const updated = [...realTargets]
                            updated[idx] = { ...updated[idx], target_value: parseInt(e.target.value) || 100 }
                            setRealTargets(updated)
                          }}
                          className="w-32"
                          min={10}
                          step={10}
                          placeholder="Sentences"
                        />
                        <span className="text-xs text-muted-foreground">sentences</span>
                        {realTargets.length > 1 && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={() => setRealTargets(realTargets.filter((_, i) => i !== idx))}
                          >
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    ))}
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const last = realTargets[realTargets.length - 1]
                        setRealTargets([...realTargets, {
                          count_mode: 'sentences',
                          target_value: last.target_value + 500,
                        }])
                      }}
                    >
                      <Plus className="h-3.5 w-3.5 mr-1.5" />
                      Add target size
                    </Button>
                  </div>
                </div>
              )}

              {/* Reference Dataset - shown when COMPOSITION selected OR (EVAL-only without Self Test) */}
              {(selectedPhases.includes('composition') ||
                (selectedPhases.includes('evaluation') && !selectedPhases.includes('generation') && !selfTest.enabled)) && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <FileUp className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <Label className="text-sm font-medium">Reference Dataset</Label>
                        <p className="text-xs text-muted-foreground">
                          Upload real reviews for context extraction and MDQA comparison (required for lexical/semantic metrics)
                        </p>
                      </div>
                    </div>
                    <Badge variant="secondary" className="text-xs">Optional</Badge>
                  </div>

                  {/* File Drop Zone */}
                  {!referenceDatasetFile ? (
                    <label
                      className={`flex flex-col items-center justify-center w-full h-28 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                        isDragOver ? 'bg-primary/10 border-primary' : 'hover:bg-muted/50'
                      }`}
                      onDragOver={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(true)
                      }}
                      onDragEnter={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(true)
                      }}
                      onDragLeave={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(false)
                      }}
                      onDrop={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(false)
                        const file = e.dataTransfer.files?.[0]
                        if (file) {
                          const ext = file.name.toLowerCase().split('.').pop()
                          if (['jsonl', 'csv', 'xml'].includes(ext || '')) {
                            setReferenceDatasetFile(file)
                            // Pre-check extraction options when file is loaded
                            setReferenceOptions(prev => ({ ...prev, extractSubjectContext: true, extractReviewerContext: true }))
                          } else {
                            toast.error('Invalid file type. Please upload .jsonl, .csv, or .xml files.')
                          }
                        }
                      }}
                    >
                      <div className="flex flex-col items-center pointer-events-none">
                        <Upload className={`h-6 w-6 mb-2 ${isDragOver ? 'text-primary' : 'text-muted-foreground'}`} />
                        <p className={`text-sm ${isDragOver ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                          {isDragOver ? 'Drop file here' : 'Click to upload or drag & drop'}
                        </p>
                        <p className="text-xs text-muted-foreground">JSONL, CSV, or SemEval XML with real reviews</p>
                        {previousReferenceFileName && !referenceDatasetFile && (
                          <p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
                            ⚠️ Previously: <span className="font-medium">{previousReferenceFileName}</span> — please re-upload
                          </p>
                        )}
                      </div>
                      <input
                        type="file"
                        className="hidden"
                        accept=".jsonl,.csv,.xml"
                        onChange={(e) => {
                          const file = e.target.files?.[0]
                          if (file) {
                            setReferenceDatasetFile(file)
                            // Pre-check extraction options when file is loaded
                            setReferenceOptions(prev => ({ ...prev, extractSubjectContext: true, extractReviewerContext: true }))
                          }
                        }}
                      />
                    </label>
                  ) : (
                    <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">{referenceDatasetFile.name}</span>
                        <span className="text-xs text-muted-foreground">({(referenceDatasetFile.size / 1024).toFixed(1)} KB)</span>
                      </div>
                      <Button variant="ghost" size="sm" onClick={() => {
                        setReferenceDatasetFile(null)
                        setReferenceOptions({ useForEvaluation: true, extractSubjectContext: false, extractReviewerContext: false, sampleCount: 25 })
                      }}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  )}

                  {/* Quick Stats - shown after parsing reference dataset */}
                  {referenceQuickStats && referenceDatasetFile && (
                    <div className="mt-3 p-3 bg-muted/30 rounded-md space-y-2">
                      <div className="text-xs font-medium text-muted-foreground">Quick Stats</div>
                      {referenceQuickStats.status === 'parsing' && (
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Parsing dataset...
                        </div>
                      )}
                      {referenceQuickStats.status === 'error' && (
                        <div className="text-xs text-red-500">
                          Error: {referenceQuickStats.error}
                        </div>
                      )}
                      {referenceQuickStats.status === 'success' && (
                        <>
                          <div className="flex flex-wrap gap-2">
                            <Badge variant="outline">{referenceQuickStats.reviewCount} reviews</Badge>
                            <Badge variant="outline">{referenceQuickStats.sentenceCount} sentences</Badge>
                            <Badge variant="outline">avg. {referenceQuickStats.avgSentencesPerReview.toFixed(1)} sents/review</Badge>
                            <Badge variant="outline">{referenceQuickStats.categoryCount} categories</Badge>
                            {referenceQuickStats.inferredDomain && (
                              <Badge variant="secondary">Domain: {referenceQuickStats.inferredDomain}</Badge>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {(() => {
                              const totalOpinions = Object.values(referenceQuickStats.polarityCounts).reduce((a, b) => a + b, 0)
                              return Object.entries(referenceQuickStats.polarityCounts).map(([pol, count]) => (
                                <Badge
                                  key={pol}
                                  className={
                                    pol === 'positive' ? 'bg-green-500/20 text-green-700 dark:text-green-400 border-green-500/30' :
                                    pol === 'negative' ? 'bg-red-500/20 text-red-700 dark:text-red-400 border-red-500/30' :
                                    'bg-gray-500/20 text-gray-700 dark:text-gray-400 border-gray-500/30'
                                  }
                                >
                                  {pol}: {Math.round(count / totalOpinions * 100)}%
                                </Badge>
                              ))
                            })()}
                          </div>
                        </>
                      )}
                    </div>
                  )}

                  {/* Reference Options - shown when file is uploaded */}
                  {referenceDatasetFile && (
                    <div className="space-y-3 pt-2 border-t">
                      {/* Use for MDQA - only when EVALUATION selected */}
                      {selectedPhases.includes('evaluation') && (
                        <div className="flex items-start gap-3">
                          <Checkbox
                            id="use-for-evaluation"
                            checked={referenceOptions.useForEvaluation}
                            onCheckedChange={(checked) => {
                              setReferenceOptions(prev => ({ ...prev, useForEvaluation: !!checked }))
                              if (checked) {
                                // Mutual exclusivity: disable Self Test when using reference for eval
                                setSelfTest(prev => ({ ...prev, enabled: false }))
                              }
                            }}
                          />
                          <div className="space-y-1">
                            <label htmlFor="use-for-evaluation" className="text-sm font-medium cursor-pointer">
                              Use for MDQA evaluation comparison
                            </label>
                            <p className="text-xs text-muted-foreground">
                              Compare generated reviews (Dgen) against this reference dataset (Dreal) for Lexical/Semantic metrics
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Extract Context - only when COMPOSITION selected */}
                      {selectedPhases.includes('composition') && (
                        <>
                          {/* Info note explaining why context extraction matters */}
                          <div className="flex items-start gap-2 rounded-md bg-blue-500/10 border border-blue-500/20 p-3 mb-1">
                            <Info className="h-4 w-4 mt-0.5 text-blue-500 shrink-0" />
                            <div className="space-y-1">
                              <p className="text-xs font-medium text-blue-700 dark:text-blue-300">Why extract context?</p>
                              <p className="text-xs text-muted-foreground">
                                Context extraction domain-matches the generated reviews with your reference dataset.
                                This ensures MDQA's Lexical and Semantic metrics provide meaningful comparisons—without
                                matching context, even two real datasets from different domains would score poorly.
                              </p>
                            </div>
                          </div>

                          <div className="flex items-start gap-3">
                            <Checkbox
                              id="extract-subject"
                              checked={referenceOptions.extractSubjectContext}
                              onCheckedChange={(checked) =>
                                setReferenceOptions(prev => ({ ...prev, extractSubjectContext: !!checked }))
                              }
                            />
                            <div className="space-y-1">
                              <label htmlFor="extract-subject" className="text-sm font-medium cursor-pointer flex items-center gap-2">
                                Extract Subject Context
                                <Sparkles className="h-3 w-3 text-amber-500" />
                              </label>
                              <p className="text-xs text-muted-foreground">
                                Analyze reviews to infer what type of product/service/entity is being reviewed
                              </p>
                            </div>
                          </div>

                          <div className="flex items-start gap-3">
                            <Checkbox
                              id="extract-reviewer"
                              checked={referenceOptions.extractReviewerContext}
                              onCheckedChange={(checked) =>
                                setReferenceOptions(prev => ({ ...prev, extractReviewerContext: !!checked }))
                              }
                            />
                            <div className="space-y-1">
                              <label htmlFor="extract-reviewer" className="text-sm font-medium cursor-pointer flex items-center gap-2">
                                Extract Reviewer Context
                                <Sparkles className="h-3 w-3 text-amber-500" />
                              </label>
                              <p className="text-xs text-muted-foreground">
                                Analyze reviews to infer typical reviewer demographics and characteristics
                              </p>
                            </div>
                          </div>

                          {/* Sample Count Slider - shown when either extraction is enabled */}
                          {(referenceOptions.extractSubjectContext || referenceOptions.extractReviewerContext) && (
                            <div className="space-y-2 pt-2">
                              <div className="flex items-center justify-between">
                                <Label className="text-sm">Sample Size</Label>
                                <span className="text-sm font-medium">{referenceOptions.sampleCount} reviews</span>
                              </div>
                              <Slider
                                value={[referenceOptions.sampleCount]}
                                onValueChange={([value]) =>
                                  setReferenceOptions(prev => ({ ...prev, sampleCount: value }))
                                }
                                min={10}
                                max={50}
                                step={5}
                                className="w-full"
                              />
                              <p className="text-xs text-muted-foreground">
                                Number of reviews to sample for context extraction. Lower values (10-20) work well for homogeneous datasets with consistent review styles. Higher values (30-50) recommended for heterogeneous datasets with diverse reviewers.
                              </p>
                            </div>
                          )}

                          {/* RDE Model Selector and Extract Button */}
                          {(referenceOptions.extractSubjectContext || referenceOptions.extractReviewerContext) && (
                            <div className="space-y-3 pt-3 border-t">
                              <div>
                                <Label className="text-sm font-medium">RDE Model (Reference Dataset Extraction)</Label>
                                <p className="text-xs text-muted-foreground mb-2">
                                  Select a model to extract context from the reference dataset
                                </p>
                                <LLMSelector
                                  providers={providers}
                                  groupedModels={groupedModels}
                                  loading={modelsLoading}
                                  value={rdeModel}
                                  onChange={setRdeModel}
                                  placeholder="Select RDE model..."
                                />
                              </div>

                              {/* Extraction Progress/Status */}
                              {rdeExtractionState.status === 'extracting' && (
                                <div className="rounded-md bg-blue-500/10 border border-blue-500/20 p-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                                    <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                                      Extracting context... ({rdeExtractionState.step}/{rdeExtractionState.totalSteps})
                                    </span>
                                  </div>
                                  {/* Step-by-step progress */}
                                  <div className="grid grid-cols-2 gap-1 text-xs">
                                    {[
                                      { step: 1, label: 'Aspect categories' },
                                      { step: 2, label: 'Polarity distribution' },
                                      { step: 3, label: 'Sex distribution' },
                                      { step: 4, label: 'Domain & Region' },
                                      { step: 5, label: 'Subject context' },
                                      { step: 6, label: 'Noise analysis' },
                                    ].map(({ step, label }) => (
                                      <span
                                        key={step}
                                        className={
                                          rdeExtractionState.step > step
                                            ? 'text-green-600 dark:text-green-400'
                                            : rdeExtractionState.step === step
                                            ? 'text-blue-600 dark:text-blue-400'
                                            : 'text-muted-foreground'
                                        }
                                      >
                                        {rdeExtractionState.step > step ? '✓' : rdeExtractionState.step === step ? '⏳' : '○'} {label}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {rdeExtractionState.status === 'success' && extractedRefContext && (
                                <div className="rounded-md bg-green-500/10 border border-green-500/20 p-3 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <CheckCircle className="h-4 w-4 text-green-500" />
                                      <span className="text-sm font-medium text-green-600 dark:text-green-400">
                                        Context extracted successfully
                                      </span>
                                    </div>
                                    <button
                                      type="button"
                                      onClick={() => {
                                        localStorage.removeItem(REF_CONTEXT_CACHE_KEY)
                                        setExtractedRefContext(null)
                                        setRdeExtractionState({ status: 'idle' })
                                        toast.success('Reference context cache cleared')
                                      }}
                                      className="text-xs text-muted-foreground hover:text-foreground underline"
                                    >
                                      Clear cache
                                    </button>
                                  </div>
                                  <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                                    <span>✓ Categories: {extractedRefContext.aspect_categories.length} extracted</span>
                                    <span>✓ Polarity: {Math.round((extractedRefContext.polarity?.positive || 0) * 100)}% pos, {Math.round((extractedRefContext.polarity?.negative || 0) * 100)}% neg</span>
                                    <span>✓ Sex: {Math.round((extractedRefContext.sex_distribution?.male || 0) * 100)}% m, {Math.round((extractedRefContext.sex_distribution?.female || 0) * 100)}% f</span>
                                    <span>✓ Domain: {extractedRefContext.domain?.value || '—'} {extractedRefContext.domain?.confidence ? `(${Math.round(extractedRefContext.domain.confidence * 100)}%)` : ''}</span>
                                    <span>✓ Region: {extractedRefContext.region?.value || extractedRefContext.region?.reason || '—'}</span>
                                    <span>✓ Noise: ~{((extractedRefContext.noise?.typo_rate || 0) * 100).toFixed(1)}% typo rate</span>
                                  </div>
                                </div>
                              )}

                              {rdeExtractionState.status === 'error' && (
                                <div className="rounded-md bg-red-500/10 border border-red-500/20 p-3">
                                  <div className="flex items-center gap-2">
                                    <XCircle className="h-4 w-4 text-red-500" />
                                    <span className="text-sm font-medium text-red-600 dark:text-red-400">
                                      Extraction failed
                                    </span>
                                  </div>
                                  <p className="text-xs text-muted-foreground mt-1">{rdeExtractionState.error}</p>
                                </div>
                              )}

                              {/* Extract Button */}
                              <Button
                                onClick={handleRdeExtraction}
                                disabled={!rdeModel || rdeExtractionState.status === 'extracting'}
                                className="w-full"
                                variant={rdeExtractionState.status === 'success' ? 'outline' : 'default'}
                              >
                                {rdeExtractionState.status === 'extracting' ? (
                                  <>
                                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                    Extracting...
                                  </>
                                ) : rdeExtractionState.status === 'success' ? (
                                  <>
                                    <RefreshCw className="h-4 w-4 mr-2" />
                                    Re-extract
                                  </>
                                ) : (
                                  <>
                                    <Sparkles className="h-4 w-4 mr-2" />
                                    Extract Reference Context
                                  </>
                                )}
                              </Button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Reference Dataset Inheritance - shown when reusing a job (GENERATION + EVALUATION only) */}
              {selectedPhases.includes('generation') && !selectedPhases.includes('composition') && reusedFrom && sourceConfig && (
                <div className="rounded-lg border border-primary/30 bg-primary/5 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <FileUp className="h-5 w-5 text-primary" />
                      <div>
                        <Label className="text-sm font-medium">Reference Dataset</Label>
                        <p className="text-xs text-muted-foreground">
                          Inherited from: {reusedJobInfo?.name}
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-xs border-primary/30 text-primary">Inherited</Badge>
                  </div>

                  <div className="space-y-2 text-sm">
                    {sourceConfig.referenceDataset?.fileName ? (
                      <>
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span>{sourceConfig.referenceDataset.fileName}</span>
                        </div>
                        <div className="grid gap-1 text-xs text-muted-foreground pl-6">
                          {sourceConfig.referenceDataset.extractedSubjectContext && (
                            <span>• Extracted subject context: Included</span>
                          )}
                          {sourceConfig.referenceDataset.extractedReviewerContext && (
                            <span>• Extracted reviewer context: Included</span>
                          )}
                          {sourceConfig.referenceDataset.useForEvaluation && (
                            <span>• Reference file for MDQA: Available</span>
                          )}
                        </div>
                      </>
                    ) : (
                      <p className="text-muted-foreground text-xs">
                        No reference dataset was configured in the source job
                      </p>
                    )}
                  </div>

                  <div className="flex items-start gap-2 rounded-md bg-muted/50 p-2">
                    <Info className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
                    <p className="text-xs text-muted-foreground">
                      To use different reference dataset settings, create a new job with COMPOSITION phase enabled
                    </p>
                  </div>
                </div>
              )}

              {/* Previous Job Selector - shown when GENERATION without COMPOSITION */}
              {selectedPhases.includes('generation') && !selectedPhases.includes('composition') && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <Copy className="h-5 w-5 text-muted-foreground" />
                    <div className="flex-1">
                      <Label className="text-sm font-medium">Reuse contexts from previous job *</Label>
                      <p className="text-xs text-muted-foreground">
                        Select a completed job to reuse its subject, reviewer, and attributes contexts
                      </p>
                    </div>
                  </div>

                  {!reusedFrom ? (
                    <Select onValueChange={(v) => handleReuseJob(v as Id<"jobs">)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a completed job..." />
                      </SelectTrigger>
                      <SelectContent className="max-h-60">
                        {completedJobs === undefined ? (
                          <SelectItem value="_loading" disabled>
                            <Loader2 className="h-4 w-4 animate-spin inline mr-2" />
                            Loading...
                          </SelectItem>
                        ) : completedJobs.length === 0 ? (
                          <SelectItem value="_empty" disabled>
                            No completed jobs available
                          </SelectItem>
                        ) : (
                          completedJobs.map((job) => (
                            <SelectItem key={job._id} value={job._id}>
                              <div className="flex flex-col">
                                <span className="font-medium">{job.name}</span>
                                <span className="text-xs text-muted-foreground">
                                  {job.subject} • {job.model.split('/').pop()}
                                </span>
                              </div>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  ) : (
                    <div className="flex items-center justify-between rounded-lg border bg-primary/5 p-3">
                      <div className="flex items-center gap-2">
                        <Lock className="h-4 w-4 text-primary" />
                        <div>
                          <p className="text-sm font-medium text-primary">{reusedJobInfo?.name || 'Loading...'}</p>
                          <p className="text-xs text-muted-foreground">{reusedJobInfo?.subject}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {checkingContexts && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                        {contextsValid === true && <CheckCircle className="h-4 w-4 text-green-500" />}
                        {contextsValid === false && <AlertTriangle className="h-4 w-4 text-amber-500" />}
                        <Button variant="ghost" size="sm" onClick={clearReuse}>
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}

                  {contextsValid === false && (
                    <p className="text-xs text-amber-600">
                      Context files not found. The pipeline will regenerate them.
                    </p>
                  )}

                  {/* Source Config Preview */}
                  {sourceConfig && reusedFrom && (
                    <Collapsible className="mt-3">
                      <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors">
                        <Eye className="h-3 w-3" />
                        <span>View composition settings</span>
                        <ChevronDown className="h-3 w-3" />
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-2 text-xs">
                        <div className="rounded-md border bg-muted/30 p-3 space-y-2.5">
                          {/* Subject Profile Section */}
                          {sourceConfig.subject_profile && (
                            <div className="space-y-1">
                              <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                                <Search className="h-3 w-3" />
                                Subject Profile
                              </div>
                              <div className="pl-4 space-y-0.5">
                                <div className="flex items-start gap-1.5">
                                  <span className="font-medium shrink-0">Query:</span>
                                  <span className="text-muted-foreground break-words">{sourceConfig.subject_profile.query}</span>
                                </div>
                                <div className="flex items-center gap-2 flex-wrap">
                                  {sourceConfig.subject_profile.domain && (
                                    <Badge variant="outline" className="text-[10px]">{sourceConfig.subject_profile.domain}</Badge>
                                  )}
                                  {sourceConfig.subject_profile.region && (
                                    <Badge variant="outline" className="text-[10px]">{sourceConfig.subject_profile.region}</Badge>
                                  )}
                                </div>
                                {sourceConfig.subject_profile.aspect_categories && sourceConfig.subject_profile.aspect_categories.length > 0 && (
                                  <div className="flex items-center gap-1.5 flex-wrap">
                                    <span className="font-medium shrink-0">Aspects:</span>
                                    {sourceConfig.subject_profile.aspect_categories.slice(0, 3).map((cat, i) => (
                                      <Badge key={i} variant="secondary" className="text-[10px]">{cat}</Badge>
                                    ))}
                                    {sourceConfig.subject_profile.aspect_categories.length > 3 && (
                                      <span className="text-muted-foreground">+{sourceConfig.subject_profile.aspect_categories.length - 3} more</span>
                                    )}
                                  </div>
                                )}
                                {sourceConfig.subject_profile.additional_context && (
                                  <div className="flex items-start gap-1.5">
                                    <span className="font-medium shrink-0">Context:</span>
                                    <span className="text-muted-foreground italic truncate max-w-[200px]">
                                      {sourceConfig.subject_profile.additional_context.length > 50
                                        ? sourceConfig.subject_profile.additional_context.slice(0, 50) + '...'
                                        : sourceConfig.subject_profile.additional_context}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          <Separator className="my-1.5" />

                          {/* Intelligence & Verification Section */}
                          <div className="space-y-1">
                            <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                              <ShieldCheck className="h-3 w-3" />
                              Intelligence & Verification
                            </div>
                            <div className="pl-4 space-y-0.5">
                              <div className="flex items-center gap-2">
                                <span className="font-medium">SIL:</span>
                                <Badge variant={sourceConfig.ablation?.sil_enabled ? "default" : "secondary"} className="text-[10px]">
                                  {sourceConfig.ablation?.sil_enabled ? 'Enabled' : 'Disabled'}
                                </Badge>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="font-medium">MAV:</span>
                                <Badge variant={sourceConfig.ablation?.mav_enabled || sourceConfig.subject_profile?.mav?.enabled ? "default" : "secondary"} className="text-[10px]">
                                  {(sourceConfig.ablation?.mav_enabled || sourceConfig.subject_profile?.mav?.enabled) ? 'Enabled' : 'Disabled'}
                                </Badge>
                              </div>
                              {sourceConfig.subject_profile?.mav?.enabled && sourceConfig.subject_profile.mav.models && (
                                <>
                                  <div className="flex items-start gap-1.5">
                                    <span className="font-medium shrink-0">Models:</span>
                                    <span className="text-muted-foreground break-words">
                                      {sourceConfig.subject_profile.mav.models.map(m => m.split('/').pop()).join(', ')}
                                    </span>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>

                          <Separator className="my-1.5" />

                          {/* Reviewer Profile Section */}
                          {sourceConfig.reviewer_profile && (
                            <div className="space-y-1">
                              <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                                <Users className="h-3 w-3" />
                                Reviewer Profile
                              </div>
                              <div className="pl-4 space-y-0.5">
                                <div className="flex items-center gap-1.5">
                                  <span className="font-medium">Age:</span>
                                  <span className="text-muted-foreground">
                                    {sourceConfig.reviewer_profile.age_range
                                      ? `${sourceConfig.reviewer_profile.age_range[0]}-${sourceConfig.reviewer_profile.age_range[1]} years`
                                      : 'Disabled'}
                                  </span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                  <span className="font-medium">Sex:</span>
                                  <span className="text-muted-foreground">
                                    M:{Math.round(sourceConfig.reviewer_profile.sex_distribution.male * 100)}%{' '}
                                    F:{Math.round(sourceConfig.reviewer_profile.sex_distribution.female * 100)}%{' '}
                                    U:{Math.round(sourceConfig.reviewer_profile.sex_distribution.unspecified * 100)}%
                                  </span>
                                </div>
                                {sourceConfig.reviewer_profile.additional_context && (
                                  <div className="flex items-start gap-1.5">
                                    <span className="font-medium shrink-0">Context:</span>
                                    <span className="text-muted-foreground italic truncate max-w-[200px]">
                                      {sourceConfig.reviewer_profile.additional_context.length > 50
                                        ? sourceConfig.reviewer_profile.additional_context.slice(0, 50) + '...'
                                        : sourceConfig.reviewer_profile.additional_context}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          <Separator className="my-1.5" />

                          {/* Attributes Profile Section */}
                          {sourceConfig.attributes_profile && (
                            <div className="space-y-1">
                              <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                                <Sliders className="h-3 w-3" />
                                Attributes Profile
                              </div>
                              <div className="pl-4 space-y-0.5">
                                <div className="flex items-center gap-1.5">
                                  <span className="font-medium">Polarity:</span>
                                  <span className="text-green-600">+{sourceConfig.attributes_profile.polarity.positive}%</span>
                                  <span className="text-gray-500">~{sourceConfig.attributes_profile.polarity.neutral}%</span>
                                  <span className="text-red-600">-{sourceConfig.attributes_profile.polarity.negative}%</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                  <span className="font-medium">Length:</span>
                                  <span className="text-muted-foreground">
                                    {sourceConfig.attributes_profile.length_range[0]}-{sourceConfig.attributes_profile.length_range[1]} sentences
                                  </span>
                                </div>
                                {sourceConfig.attributes_profile.temperature_range && (
                                  <div className="flex items-center gap-1.5">
                                    <span className="font-medium">Temperature:</span>
                                    <span className="text-muted-foreground">
                                      {sourceConfig.attributes_profile.temperature_range[0].toFixed(1)}-{sourceConfig.attributes_profile.temperature_range[1].toFixed(1)}
                                    </span>
                                  </div>
                                )}
                                {sourceConfig.attributes_profile.noise.preset && (
                                  <div className="flex items-center gap-1.5">
                                    <span className="font-medium">Noise:</span>
                                    <Badge variant="secondary" className="text-[10px]">
                                      {sourceConfig.attributes_profile.noise.preset}
                                      {sourceConfig.attributes_profile.noise.preset === 'ref_dataset' && sourceConfig.attributes_profile.noise.typo_rate != null &&
                                        ` (~${(sourceConfig.attributes_profile.noise.typo_rate * 100).toFixed(1)}%)`}
                                    </Badge>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Reference Dataset Section (conditional) */}
                          {(sourceConfig.referenceDataset?.fileName || sourceConfig.reference_dataset?.fileName) && (
                            <>
                              <Separator className="my-1.5" />
                              <div className="space-y-1">
                                <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                                  <Database className="h-3 w-3" />
                                  Reference Dataset
                                </div>
                                <div className="pl-4 space-y-0.5">
                                  <div className="flex items-center gap-1.5">
                                    <FileText className="h-3 w-3 text-muted-foreground" />
                                    <span className="text-muted-foreground">
                                      {sourceConfig.referenceDataset?.fileName || sourceConfig.reference_dataset?.fileName}
                                    </span>
                                  </div>
                                  <div className="flex flex-wrap gap-1">
                                    {(sourceConfig.referenceDataset?.extractedSubjectContext || sourceConfig.reference_dataset?.extractedSubjectContext) && (
                                      <Badge variant="outline" className="text-[10px]">Subject extracted</Badge>
                                    )}
                                    {(sourceConfig.referenceDataset?.extractedReviewerContext || sourceConfig.reference_dataset?.extractedReviewerContext) && (
                                      <Badge variant="outline" className="text-[10px]">Reviewer extracted</Badge>
                                    )}
                                    {(sourceConfig.referenceDataset?.useForEvaluation || sourceConfig.reference_dataset?.useForEvaluation) && (
                                      <Badge variant="outline" className="text-[10px]">MDQA reference</Badge>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                </div>
              )}

              {/* Dataset Upload - shown when EVALUATION only (no generation) */}
              {selectedPhases.includes('evaluation') && !selectedPhases.includes('generation') && (
                <div className="rounded-lg border p-4 space-y-3">
                  <div className="flex items-center gap-3">
                    <Upload className="h-5 w-5 text-muted-foreground" />
                    <div className="flex-1">
                      <Label className="text-sm font-medium">Upload Dataset *</Label>
                      <p className="text-xs text-muted-foreground">
                        Upload a dataset file to evaluate (JSONL, CSV, or SemEval XML)
                      </p>
                    </div>
                  </div>

                  {!datasetFile ? (
                    <label
                      className={`flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                        isDragOver ? 'bg-primary/10 border-primary' : 'hover:bg-muted/50'
                      }`}
                      onDragOver={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(true)
                      }}
                      onDragEnter={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(true)
                      }}
                      onDragLeave={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(false)
                      }}
                      onDrop={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setIsDragOver(false)
                        const file = e.dataTransfer.files?.[0]
                        if (file) {
                          const ext = file.name.toLowerCase().split('.').pop()
                          if (['jsonl', 'csv', 'xml'].includes(ext || '')) {
                            setDatasetFile(file)
                          } else {
                            toast.error('Invalid file type. Please upload .jsonl, .csv, or .xml files.')
                          }
                        }
                      }}
                    >
                      <div className="flex flex-col items-center pointer-events-none">
                        <FileText className={`h-8 w-8 mb-2 ${isDragOver ? 'text-primary' : 'text-muted-foreground'}`} />
                        <p className={`text-sm ${isDragOver ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
                          {isDragOver ? 'Drop file here' : 'Click to upload or drag & drop'}
                        </p>
                        <p className="text-xs text-muted-foreground">.jsonl, .csv, .xml</p>
                        {previousDatasetFileName && !datasetFile && (
                          <p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
                            ⚠️ Previously: <span className="font-medium">{previousDatasetFileName}</span> — please re-upload
                          </p>
                        )}
                      </div>
                      <input
                        type="file"
                        className="hidden"
                        accept=".jsonl,.csv,.xml"
                        onChange={(e) => {
                          const file = e.target.files?.[0]
                          if (file) setDatasetFile(file)
                        }}
                      />
                    </label>
                  ) : (
                    <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">{datasetFile.name}</span>
                        <span className="text-xs text-muted-foreground">({(datasetFile.size / 1024).toFixed(1)} KB)</span>
                      </div>
                      <Button variant="ghost" size="sm" onClick={() => setDatasetFile(null)}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* COMPOSITION TAB */}
          {currentTabId === 'composition' && (
            <div className="space-y-6">
              <CompositionPhase config={config} updateConfig={updateConfig} constraints={constraints} modelValidations={modelValidations} referenceOptions={referenceOptions} referenceDatasetFile={referenceDatasetFile} extractedRefContext={extractedRefContext} />
            </div>
          )}

          {/* GENERATION TAB */}
          {currentTabId === 'generation' && (
            <div className="space-y-6">
              <GenerationPhase
                config={config}
                updateConfig={updateConfig}
                constraints={constraints}
                isReusing={!!reusedFrom}
                reusedJobName={reusedJobInfo?.name}
                modelValidations={modelValidations}
              />
            </div>
          )}

          {/* EVALUATION TAB */}
          {currentTabId === 'evaluation' && (
            <EvaluationPhase
              config={config}
              updateConfig={updateConfig}
              referenceFile={referenceDatasetFile}
              onReferenceFileChange={setReferenceDatasetFile}
              referenceFromInput={selectedPhases.includes('composition') && referenceOptions.useForEvaluation && !!referenceDatasetFile}
              inheritedReferenceName={!selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation ? sourceConfig.referenceDataset.fileName : undefined}
              selfTestEnabled={selfTest.enabled}
            />
          )}

          {/* OUTPUT TAB */}
          {currentTabId === 'output' && (
            <div className="space-y-6">
              {/* Job Summary - Multi-Card Grid View */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Pipeline Overview - spans full width */}
                <div className="rounded-lg border p-4 lg:col-span-2">
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="h-4 w-4 text-muted-foreground" />
                    <h4 className="font-medium text-sm">Pipeline Overview</h4>
                  </div>
                  <div className="grid gap-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Job Name</span>
                      <span className="font-medium">{config.name || <span className="text-amber-500 italic">Not set</span>}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Phases</span>
                      <div className="flex gap-1">
                        {selectedPhases.map((p, i) => (
                          <span key={p} className="flex items-center gap-1">
                            {i > 0 && <span className="text-muted-foreground text-xs">→</span>}
                            <Badge variant="secondary" className="text-[10px] px-1.5">{p.charAt(0).toUpperCase() + p.slice(1)}</Badge>
                          </span>
                        ))}
                      </div>
                    </div>
                    {reusedFrom && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Reusing Contexts</span>
                        <span className="font-medium text-primary text-xs">{reusedJobInfo?.name}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* COMPOSITION - unified card with Subject & Intelligence + Reviewer & Attributes */}
                {selectedPhases.includes('composition') && (
                  <div className="rounded-lg border p-4 lg:col-span-2">
                    <div className="flex items-center gap-2 mb-4">
                      <Layers className="h-4 w-4 text-muted-foreground" />
                      <h4 className="font-medium text-sm">Composition</h4>
                    </div>

                    {/* Internal 2-column layout for sections */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Subject & Intelligence Section */}
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <Search className="h-3.5 w-3.5 text-muted-foreground" />
                          <h5 className="font-medium text-xs text-muted-foreground uppercase tracking-wide">Subject & Intelligence</h5>
                        </div>
                        <div className="grid gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Query</span>
                            <span className="font-medium max-w-[200px] truncate text-right">{config.subject_profile.query || <span className="text-amber-500 italic">Not set</span>}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Domain</span>
                            <span className="font-medium capitalize">{config.subject_profile.domain}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Region</span>
                            <span className="font-medium capitalize">{config.subject_profile.region}</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-muted-foreground">Aspect Categories</span>
                            <Badge variant="outline" className="text-[10px]">
                              {config.subject_profile.aspect_category_mode === 'infer' ? 'Infer from Reviews' :
                               config.subject_profile.aspect_category_mode === 'import' ? `Imported (${config.subject_profile.aspect_categories.length})` :
                               `Preset (${config.subject_profile.aspect_categories.length})`}
                            </Badge>
                          </div>
                          <Separator className="my-1" />
                          <div className="flex justify-between items-center">
                            <span className="text-muted-foreground">SIL (Web Search)</span>
                            <Badge variant={config.ablation.sil_enabled ? 'default' : 'secondary'} className="text-[10px]">
                              {config.ablation.sil_enabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-muted-foreground">MAV (Multi-Agent)</span>
                            <Badge variant={config.ablation.mav_enabled ? 'default' : 'secondary'} className="text-[10px]">
                              {config.ablation.mav_enabled ? `${config.subject_profile.mav.models.filter((m: string) => m).length} Models` : 'Single Model'}
                            </Badge>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-muted-foreground">{config.ablation.mav_enabled ? 'MAV Models' : 'Intelligence Model'}</span>
                            {config.ablation.mav_enabled ? (
                              <span className="font-medium text-xs">
                                {config.subject_profile.mav.models.filter((m: string) => m).length >= 2
                                  ? `${config.subject_profile.mav.models.filter((m: string) => m).length} set`
                                  : <span className="text-amber-500 italic">{config.subject_profile.mav.models.filter((m: string) => m).length}/2+ set</span>}
                              </span>
                            ) : (
                              <span className="font-medium text-xs">
                                {config.subject_profile.mav.models[0]
                                  ? config.subject_profile.mav.models[0].split('/').pop()
                                  : <span className="text-amber-500 italic">Not set</span>}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Reviewer & Attributes Section */}
                      <div>
                        <div className="flex items-center gap-2 mb-3">
                          <Users className="h-3.5 w-3.5 text-muted-foreground" />
                          <h5 className="font-medium text-xs text-muted-foreground uppercase tracking-wide">Reviewer & Attributes</h5>
                        </div>
                        <div className="grid gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Age Range</span>
                            <span className="font-medium">
                              {config.ablation.age_enabled
                                ? `${config.reviewer_profile.age_range[0]}–${config.reviewer_profile.age_range[1]}`
                                : <span className="text-muted-foreground italic">Disabled</span>}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Sex Ratio (M/F/U)</span>
                            <span className="font-medium">
                              {config.ablation.sex_enabled
                                ? `${Math.round((config.reviewer_profile.sex_distribution.male ?? 0.5) * 100)}% / ${Math.round((config.reviewer_profile.sex_distribution.female ?? 0.5) * 100)}% / ${Math.round((config.reviewer_profile.sex_distribution.unknown ?? 0) * 100)}%`
                                : <span className="text-muted-foreground italic">Disabled</span>}
                            </span>
                          </div>
                          <Separator className="my-1" />
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Polarity</span>
                            <span className="font-mono text-xs">
                              {config.ablation.polarity_enabled
                                ? `+${Math.round(config.attributes_profile.polarity.positive * 100)}% / ~${Math.round(config.attributes_profile.polarity.neutral * 100)}% / −${Math.round(config.attributes_profile.polarity.negative * 100)}%`
                                : <span className="text-muted-foreground italic font-sans">LLM decides</span>}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Noise</span>
                            <span className="font-medium capitalize">
                              {config.ablation.noise_enabled
                                ? `${config.attributes_profile.noise.preset || 'Custom'} (${(config.attributes_profile.noise.typo_rate * 100).toFixed(1)}%)`
                                : <span className="text-muted-foreground italic normal-case">Disabled</span>}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Sentence Length</span>
                            <span className="font-medium">{config.attributes_profile.length_range[0]}–{config.attributes_profile.length_range[1]} sentences</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Temperature</span>
                            <span className="font-medium">{config.attributes_profile.temperature_range[0].toFixed(1)}–{config.attributes_profile.temperature_range[1].toFixed(1)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Generation - spans full width on lg screens */}
                {selectedPhases.includes('generation') && (
                  <div className="rounded-lg border p-4 lg:col-span-2">
                    <div className="flex items-center gap-2 mb-3">
                      <Cog className="h-4 w-4 text-muted-foreground" />
                      <h4 className="font-medium text-sm">Generation</h4>
                    </div>
                    {/* 2-column grid for generation fields on lg screens */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Model(s)</span>
                        <span className="font-medium text-xs max-w-[240px] truncate">
                          {config.generation.models?.filter(Boolean).length > 0
                            ? config.generation.models.filter(Boolean).map((m: string) => m.split('/').pop()).join(', ')
                            : config.generation.model
                              ? config.generation.model.split('/').pop()
                              : <span className="text-amber-500 italic">Not selected</span>
                          }
                          {config.generation.models?.filter(Boolean).length > 1 && config.generation.parallel_models && (
                            <span className="text-amber-500 ml-1">(parallel)</span>
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Target</span>
                        <span className="font-medium">
                          {config.generation.count_mode === 'sentences'
                            ? `${(config.generation.target_sentences || 1000).toLocaleString()} sentences`
                            : `${config.generation.count.toLocaleString()} reviews`
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Dataset Mode</span>
                        <span className="font-medium capitalize">{config.generation.dataset_mode}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Output Formats</span>
                        <div className="flex gap-1">
                          {(config.generation.output_formats || ['jsonl']).map((f: string) => (
                            <Badge key={f} variant="outline" className="text-[10px] px-1.5">
                              {OUTPUT_FORMATS.find(o => o.value === f)?.label || f}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Batch / Request</span>
                        <span className="font-medium">{config.generation.batch_size} / {config.generation.request_size}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">NEB</span>
                        <span className="font-medium">
                          {!config.generation.neb_enabled || config.generation.neb_depth === 0
                            ? <span className="text-muted-foreground italic">Disabled</span>
                            : `Depth ${config.generation.neb_depth ?? 3} (${(config.generation.neb_depth ?? 3) * config.generation.request_size} reviews)`
                          }
                        </span>
                      </div>
                      {(config.generation.total_runs || 1) > 1 && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Runs</span>
                          <span className="font-medium text-amber-600 dark:text-amber-400">
                            {config.generation.total_runs} runs
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Evaluation - spans full width on lg screens */}
                {selectedPhases.includes('evaluation') && (
                  <div className="rounded-lg border p-4 lg:col-span-2">
                    <div className="flex items-center gap-2 mb-3">
                      <Activity className="h-4 w-4 text-muted-foreground" />
                      <h4 className="font-medium text-sm">Evaluation (MDQA)</h4>
                    </div>
                    <div className="grid gap-2 text-sm">
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Reference Dataset</span>
                        {(() => {
                          const inheritedRef = !selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation
                          if (selfTest.enabled) return <span className="font-medium text-green-600 dark:text-green-400 text-xs">Self Test ({selfTest.splitMode === 'random' ? 'Random' : 'Normal'} 50% Split)</span>
                          if (referenceDatasetFile && referenceOptions.useForEvaluation) return <span className="font-medium text-green-600 dark:text-green-400 text-xs">{referenceDatasetFile.name}</span>
                          if (inheritedRef) return <span className="font-medium text-green-600 dark:text-green-400 text-xs">{sourceConfig.referenceDataset.fileName} (inherited)</span>
                          return <span className="text-muted-foreground/60 italic text-xs">Not provided (Lexical/Semantic skipped)</span>
                        })()}
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Metrics</span>
                        <span className="font-medium">
                          {(selfTest.enabled || (referenceDatasetFile && referenceOptions.useForEvaluation) || (!selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation))
                            ? `${config.evaluation.metrics.length} selected`
                            : `${config.evaluation.metrics.filter((m: string) => ['distinct_1', 'distinct_2', 'self_bleu'].includes(m)).length} active (Diversity only)`
                          }
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {config.evaluation.metrics.map((m: string) => {
                          const isDisabled = !selfTest.enabled && !(referenceDatasetFile && referenceOptions.useForEvaluation) && !(!selectedPhases.includes('composition') && reusedFrom && sourceConfig?.referenceDataset?.useForEvaluation) && !['distinct_1', 'distinct_2', 'self_bleu'].includes(m)
                          return (
                            <Badge
                              key={m}
                              variant="secondary"
                              className={`text-[10px] px-1.5 ${isDisabled ? 'opacity-40 line-through' : ''}`}
                            >
                              {m}
                            </Badge>
                          )
                        })}
                      </div>
                    </div>
                  </div>
                )}

                {/* Cost Estimation - standalone card */}
                <CostEstimationCard
                  config={config}
                  isReusing={!!reusedFrom || !selectedPhases.includes('composition')}
                  selectedPhases={selectedPhases}
                  rdeModel={rdeModel}
                  rdeExtractionComplete={rdeExtractionState.status === 'success'}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-3 justify-between">
                <Button
                  variant="outline"
                  onClick={() => setActiveTab(t => Math.max(0, t - 1))}
                >
                  <ChevronLeft className="mr-2 h-4 w-4" />
                  Previous
                </Button>
                <div className="flex items-center gap-3">
                <Button
                  variant="outline"
                  onClick={() => handleCreateJob(false)}
                  disabled={submitting || validatingModels}
                >
                  {submitting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    'Create Job'
                  )}
                </Button>
                <Button
                  onClick={() => {
                    if (hasInvalidModels) {
                      setShowInvalidModelsAlert(true)
                    } else {
                      handleCreateJob(true)
                    }
                  }}
                  disabled={submitting || validatingModels || hasCheckingModels}
                  style={{ backgroundColor: PHASES[1].strongColor }}
                >
                  {validatingModels ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Validating...
                    </>
                  ) : submitting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Create Job & Run Pipeline
                    </>
                  )}
                </Button>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Navigation (hidden on last tab where Previous is in the action buttons) */}
      {activeTab < wizardTabs.length - 1 && (
        <div className="flex justify-between pt-4">
          <Button
            variant="outline"
            onClick={() => setActiveTab(t => Math.max(0, t - 1))}
            disabled={activeTab === 0}
          >
            <ChevronLeft className="mr-2 h-4 w-4" />
            Previous
          </Button>

          <Button
            onClick={() => setActiveTab(t => t + 1)}
            disabled={isNextDisabledByRde}
            title={isNextDisabledByRde ? 'Extract reference context before continuing' : undefined}
          >
            Next
            <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Invalid Models Alert Dialog */}
      <AlertDialog open={showInvalidModelsAlert} onOpenChange={setShowInvalidModelsAlert}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Model Availability Issues
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3">
                <p>
                  The following models are currently unavailable. Please select different models before proceeding:
                </p>
                <div className="space-y-2">
                  {invalidModelsList.map((item, i) => (
                    <div key={i} className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/50 p-3">
                      <X className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium text-red-700 dark:text-red-400">{item.label}</p>
                        <p className="text-xs text-red-600 dark:text-red-500">{item.error}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground">
                  This can happen when a model provider has capacity issues or when free tier models require specific OpenRouter privacy settings.
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setShowInvalidModelsAlert(false)}>
              Understood
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Validation Errors Alert Dialog */}
      <AlertDialog open={showValidationErrors} onOpenChange={setShowValidationErrors}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Missing Required Fields
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-3">
                <p>
                  Please address the following before creating the job:
                </p>
                <div className="space-y-2">
                  {validationErrors.map((error, i) => (
                    <div key={i} className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/50 p-3">
                      <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-sm text-amber-700 dark:text-amber-400">{error}</p>
                    </div>
                  ))}
                </div>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction onClick={() => setShowValidationErrors(false)}>
              Got it
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

// Extracted reference context type
interface ExtractedRefContext {
  subject_query: string | null
  additional_context: string | null
  reviewer_context: string | null
  domain: { value: string | null; confidence: number; reason?: string } | null
  region: { value: string | null; confidence: number; reason?: string } | null
  polarity: { positive: number; neutral: number; negative: number } | null
  sex_distribution: { male: number; female: number; unknown: number; detected_count: number } | null
  noise: { typo_rate: number; has_colloquialisms: boolean; sample_size: number } | null
  review_length: { avg_sentences: number; min_sentences: number; max_sentences: number; suggested_range: number[] } | null
  aspect_categories: string[]
  sample_count: number
  total_reviews: number
}

// COMPOSITION Phase - Combines Subject, Reviewer, and Attributes with collapsible sections
function CompositionPhase({ config, updateConfig, constraints, modelValidations, referenceOptions, referenceDatasetFile, extractedRefContext }: { config: any; updateConfig: (path: string, value: any) => void; constraints: UIConstraints; modelValidations: Record<string, ModelValidationState>; referenceOptions?: { extractSubjectContext: boolean; extractReviewerContext: boolean }; referenceDatasetFile?: File | null; extractedRefContext?: ExtractedRefContext | null }) {
  const [openSections, setOpenSections] = useState({
    subject: true,
    intelligence: false,
    reviewer: false,
    attributes: false,
  })

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  return (
    <div className="space-y-4">
      {/* Subject Profile Section */}
      <Collapsible open={openSections.subject} onOpenChange={() => toggleSection('subject')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.subject ? PHASES[0].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[0].lightColor }}
                >
                  <Search className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Subject Profile</h3>
                  <p className="text-sm text-muted-foreground">
                    {config.subject_profile.query || 'Define what to generate reviews about'}
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.subject ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4">
              <SubjectStepContent config={config} updateConfig={updateConfig} referenceOptions={referenceOptions} referenceDatasetFile={referenceDatasetFile} extractedRefContext={extractedRefContext} />
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      {/* Intelligence Section (SIL + MAV) */}
      <Collapsible open={openSections.intelligence} onOpenChange={() => toggleSection('intelligence')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.intelligence ? PHASES[0].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[0].lightColor }}
                >
                  <ShieldCheck className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Intelligence & Verification</h3>
                  <p className="text-sm text-muted-foreground">
                    {config.ablation.sil_enabled ? 'SIL web search' : 'SIL disabled'}
                    {' + '}
                    {config.ablation.mav_enabled ? 'MAV multi-agent' : 'Single model'}
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.intelligence ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4">
              <IntelligenceStepContent config={config} updateConfig={updateConfig} constraints={constraints} modelValidations={modelValidations} />
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      {/* Reviewer Profile Section */}
      <Collapsible open={openSections.reviewer} onOpenChange={() => toggleSection('reviewer')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.reviewer ? PHASES[0].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[0].lightColor }}
                >
                  <Users className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Reviewer Profile</h3>
                  <p className="text-sm text-muted-foreground">
                    {config.ablation.age_enabled
                      ? `Age ${config.reviewer_profile.age_range[0]}-${config.reviewer_profile.age_range[1]}, `
                      : ''}
                    demographic controls
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.reviewer ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4">
              <ReviewerStepContent config={config} updateConfig={updateConfig} constraints={constraints} referenceOptions={referenceOptions} referenceDatasetFile={referenceDatasetFile} extractedRefContext={extractedRefContext} />
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      {/* Attributes Profile Section */}
      <Collapsible open={openSections.attributes} onOpenChange={() => toggleSection('attributes')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.attributes ? PHASES[0].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[0].lightColor }}
                >
                  <Sliders className="h-5 w-5" style={{ color: PHASES[0].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Attributes Profile</h3>
                  <p className="text-sm text-muted-foreground">
                    {Math.round(config.attributes_profile.polarity.positive * 100)}% positive, {config.attributes_profile.noise.preset || 'custom'} noise
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.attributes ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4">
              <AttributesStepContent config={config} updateConfig={updateConfig} constraints={constraints} extractedRefContext={extractedRefContext} />
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>
    </div>
  )
}

// Output format options (JSONL first - always produced)
const OUTPUT_FORMATS = [
  { value: 'jsonl', label: 'JSONL', description: 'Always produced (internal format)', alwaysEnabled: true },
  { value: 'semeval_xml', label: 'SemEval XML', description: 'SemEval competition format' },
  { value: 'csv', label: 'CSV', description: 'Comma-separated values' },
]

// More accurate token estimates based on actual SIL pipeline architecture
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

// Helper to get model pricing from OpenRouter models list
// Returns pricing per 1M tokens
function getModelPricing(models: any[], modelId: string): { input: number; output: number } | null {
  // Local vLLM models have zero cost
  if (modelId.startsWith('local/')) return { input: 0, output: 0 }

  const model = models.find((m: any) => m.id === modelId)
  if (!model) return null

  // OpenRouter pricing is per token, convert to per 1M tokens
  const inputPrice = parseFloat(model.pricing?.prompt || '0') * 1_000_000
  const outputPrice = parseFloat(model.pricing?.completion || '0') * 1_000_000

  return { input: inputPrice, output: outputPrice }
}

// Cost Estimation Card Component - standalone card with accurate LLM call counts
function CostEstimationCard({
  config,
  isReusing,
  selectedPhases,
  rdeModel,
  rdeExtractionComplete,
}: {
  config: any
  isReusing?: boolean
  selectedPhases: string[]
  rdeModel?: string
  rdeExtractionComplete?: boolean
}) {
  // Get models from OpenRouter hook for pricing data
  const { models, loading } = useOpenRouterModels()

  // Default pricing if model not found (conservative estimate)
  const defaultPricing = { input: 2.0, output: 8.0 }

  // Get pricing for selected generation model
  const modelPricing = getModelPricing(models, config.generation.model) || defaultPricing

  // Count MAV models
  const mavModels = config.subject_profile.mav.models.filter((m: string) => m)
  const M = config.ablation.mav_enabled ? mavModels.length : 1 // At least 1 model for SIL

  // Calculate review count based on mode
  const sentencesPerReview = config.attributes_profile?.length_range
    ? (config.attributes_profile.length_range[0] + config.attributes_profile.length_range[1]) / 2
    : 3.5

  let N: number
  let displayMode: string
  if (config.generation.count_mode === 'sentences') {
    const targetSentences = config.generation.target_sentences || 1000
    N = Math.ceil(targetSentences / sentencesPerReview)
    displayMode = `~${N.toLocaleString()} reviews (${targetSentences.toLocaleString()} sentences)`
  } else {
    N = config.generation.count
    displayMode = `${N.toLocaleString()} reviews`
  }

  // Calculate SIL cost (composition phase) - 3M + 1 calls
  // Round 1: M models × (research + query generation) = 2M calls
  // Round 3: M models × answer queries = M calls
  // Classification: 1 call
  let compositionCost = 0
  let compositionCalls = 0

  if (!isReusing && selectedPhases.includes('composition')) {
    // Calculate costs for each MAV model across SIL rounds
    for (const mavModelId of (mavModels.length > 0 ? mavModels : [config.generation.model])) {
      const pricing = getModelPricing(models, mavModelId) || defaultPricing

      // Round 1: Research
      const researchCost = (TOKEN_ESTIMATES.sil_research.input * pricing.input + TOKEN_ESTIMATES.sil_research.output * pricing.output) / 1_000_000
      // Round 1: Query Generation
      const queryGenCost = (TOKEN_ESTIMATES.sil_generate_queries.input * pricing.input + TOKEN_ESTIMATES.sil_generate_queries.output * pricing.output) / 1_000_000
      // Round 3: Answer Queries
      const answerCost = (TOKEN_ESTIMATES.sil_answer_queries.input * pricing.input + TOKEN_ESTIMATES.sil_answer_queries.output * pricing.output) / 1_000_000

      compositionCost += researchCost + queryGenCost + answerCost
    }

    // Classification (1 call using first model)
    const classifyModel = mavModels[0] || config.generation.model
    const classifyPricing = getModelPricing(models, classifyModel) || defaultPricing
    const classifyCost = (TOKEN_ESTIMATES.sil_classify.input * classifyPricing.input + TOKEN_ESTIMATES.sil_classify.output * classifyPricing.output) / 1_000_000
    compositionCost += classifyCost

    // Total composition calls: 3M + 1
    compositionCalls = 3 * M + 1
  }

  // Calculate AML cost (generation phase) - N calls × total_runs
  let generationCost = 0
  let generationCalls = 0
  const totalRuns = config.generation.total_runs || 1

  if (selectedPhases.includes('generation')) {
    const amlInputTokens = N * TOKEN_ESTIMATES.aml_per_review.input * totalRuns
    const amlOutputTokens = N * TOKEN_ESTIMATES.aml_per_review.output * totalRuns
    generationCost = (amlInputTokens * modelPricing.input + amlOutputTokens * modelPricing.output) / 1_000_000
    generationCalls = N * totalRuns
  }

  // Calculate RDE cost (one-time extraction from reference dataset)
  // Show estimated cost whenever rdeModel is set (regardless of extraction status)
  let rdeCost = 0
  let rdeCalls = 0

  if (rdeModel) {
    const rdePricing = getModelPricing(models, rdeModel) || defaultPricing
    const rdeInputTokens = 3000 // Estimated: system prompt + sample reviews
    const rdeOutputTokens = 2000 // Estimated: extraction response
    rdeCost = (rdeInputTokens * rdePricing.input + rdeOutputTokens * rdePricing.output) / 1_000_000
    rdeCalls = 1
  }

  // Total calculations
  const totalCalls = rdeCalls + compositionCalls + generationCalls
  const totalCost = rdeCost + compositionCost + generationCost

  // Format currency
  const formatCost = (cost: number) => {
    if (cost === 0) return '$0.00'
    if (cost < 0.01) return '< $0.01'
    return `$${cost.toFixed(2)}`
  }

  // Format number with commas
  const formatNumber = (n: number) => n.toLocaleString()

  if (loading) {
    return (
      <div className="rounded-lg border p-4 lg:col-span-2">
        <div className="flex items-center gap-2 mb-3">
          <DollarSign className="h-4 w-4 text-muted-foreground" />
          <h4 className="font-medium text-sm">Estimated Cost</h4>
        </div>
        <div className="text-xs text-muted-foreground">Loading pricing data...</div>
      </div>
    )
  }

  return (
    <div className="rounded-lg border p-4 lg:col-span-2">
      <div className="flex items-center gap-2 mb-4">
        <DollarSign className="h-4 w-4 text-muted-foreground" />
        <h4 className="font-medium text-sm">Estimated Cost</h4>
      </div>

      <div className="space-y-4">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-lg font-semibold">{formatNumber(totalCalls)}</div>
            <div className="text-[10px] text-muted-foreground">Total LLM Calls</div>
          </div>
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-lg font-semibold">
              {formatNumber(Math.round((compositionCalls > 0 ? (M * (TOKEN_ESTIMATES.sil_research.input + TOKEN_ESTIMATES.sil_research.output + TOKEN_ESTIMATES.sil_generate_queries.input + TOKEN_ESTIMATES.sil_generate_queries.output + TOKEN_ESTIMATES.sil_answer_queries.input + TOKEN_ESTIMATES.sil_answer_queries.output) + TOKEN_ESTIMATES.sil_classify.input + TOKEN_ESTIMATES.sil_classify.output) : 0) + (N * (TOKEN_ESTIMATES.aml_per_review.input + TOKEN_ESTIMATES.aml_per_review.output))))}
            </div>
            <div className="text-[10px] text-muted-foreground">Est. Tokens</div>
          </div>
          <div className="rounded-lg p-3" style={{ backgroundColor: `${PHASES[1].strongColor}15` }}>
            <div className="text-lg font-semibold" style={{ color: PHASES[1].strongColor }}>
              ~{formatCost(totalCost)}
            </div>
            <div className="text-[10px] text-muted-foreground">Est. Total</div>
          </div>
        </div>

        {/* Phase Breakdown - 4 columns in a row */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
          {/* RDE Extraction - show whenever reference dataset is configured */}
          {rdeModel && (
            <div className="rounded border p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                <span className="text-xs font-medium">RDE</span>
              </div>
              <div className={`text-sm font-semibold mb-1 ${rdeExtractionComplete ? 'text-green-600' : ''}`}>
                {rdeExtractionComplete ? `${formatCost(rdeCost)} ✓` : formatCost(rdeCost)}
              </div>
              <div className="text-[10px] text-muted-foreground">
                <div>1 LLM call (one-time)</div>
                <div className="text-[9px] mt-0.5">{rdeModel?.split('/').pop() || 'RDE model'}</div>
              </div>
            </div>
          )}

          {/* Composition Phase */}
          {selectedPhases.includes('composition') && (
            <div className="rounded border p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: PHASES[0].strongColor }} />
                <span className="text-xs font-medium">Composition</span>
              </div>
              <div className={`text-sm font-semibold mb-1 ${isReusing ? 'text-green-600' : ''}`}>
                {isReusing ? '$0.00 (reused)' : formatCost(compositionCost)}
              </div>
              {!isReusing && (
                <div className="text-[10px] text-muted-foreground">
                  <div>{compositionCalls} LLM calls</div>
                  <div className="text-[9px] mt-0.5">{M} MAV models × 3 rounds + 1</div>
                </div>
              )}
            </div>
          )}

          {/* Generation Phase */}
          {selectedPhases.includes('generation') && (
            <div className="rounded border p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: PHASES[1].strongColor }} />
                <span className="text-xs font-medium">Generation</span>
              </div>
              <div className="text-sm font-semibold mb-1">{formatCost(generationCost)}</div>
              <div className="text-[10px] text-muted-foreground">
                <div>{formatNumber(generationCalls)} LLM calls</div>
                <div className="text-[9px] mt-0.5">
                  {config.generation.model ? config.generation.model.split('/').pop() : 'No model'}
                  {totalRuns > 1 && ` × ${totalRuns} runs`}
                </div>
              </div>
            </div>
          )}

          {/* Evaluation Phase */}
          {selectedPhases.includes('evaluation') && (
            <div className="rounded border p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: PHASES[2].strongColor }} />
                <span className="text-xs font-medium">Evaluation</span>
              </div>
              <div className="text-sm font-semibold mb-1 text-green-600">$0.00</div>
              <div className="text-[10px] text-muted-foreground">
                <div>Local metrics</div>
                <div className="text-[9px] mt-0.5">No API calls</div>
              </div>
            </div>
          )}
        </div>

        {/* Formula explanation */}
        <div className="text-[10px] text-muted-foreground pt-2 border-t">
          <p>* Formula: Total calls = (3M + 1) + N{totalRuns > 1 ? ` × R` : ''} where M = {M} MAV models, N = {formatNumber(N)} reviews{totalRuns > 1 ? `, R = ${totalRuns} runs` : ''}</p>
          <p>* Estimates based on average token usage. Actual costs may vary by ±20%.</p>
        </div>
      </div>
    </div>
  )
}

// GENERATION Phase - LLM settings configuration
function GenerationPhase({
  config,
  updateConfig,
  constraints,
  isReusing,
  reusedJobName,
  modelValidations,
}: {
  config: any
  updateConfig: (path: string, value: any) => void
  constraints: UIConstraints
  isReusing?: boolean
  reusedJobName?: string
  modelValidations: Record<string, ModelValidationState>
}) {
  const [openSections, setOpenSections] = useState({
    settings: true,
    output: false,
  })

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // Toggle output format selection (JSONL cannot be removed - always produced)
  const toggleOutputFormat = (format: string) => {
    if (format === 'jsonl') return  // JSONL is always enabled
    const current = config.generation.output_formats || ['jsonl']
    if (current.includes(format)) {
      updateConfig('generation.output_formats', current.filter((f: string) => f !== format))
    } else {
      updateConfig('generation.output_formats', [...current, format])
    }
  }

  return (
    <div className="space-y-4">
      {/* Generation Settings Section */}
      <Collapsible open={openSections.settings} onOpenChange={() => toggleSection('settings')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.settings ? PHASES[1].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[1].lightColor }}
                >
                  <Cog className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Generation Settings</h3>
                  <p className="text-sm text-muted-foreground">
                    {config.generation.targets?.length > 1
                      ? `${config.generation.targets.length} targets: ${config.generation.targets.map((t: any) => t.target_value).join(', ')}`
                      : config.generation.targets?.[0]
                        ? `${config.generation.targets[0].target_value.toLocaleString()} ${config.generation.targets[0].count_mode}`
                        : `${config.generation.count.toLocaleString()} reviews`
                    } • {config.generation.model ? config.generation.model.split('/').pop() : 'no model'}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <PreviewAMLDialog config={config} />
                <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.settings ? 'rotate-180' : ''}`} />
              </div>
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4 space-y-6">
              <GenerationStepContent config={config} updateConfig={updateConfig} constraints={constraints} modelValidations={modelValidations} />
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      {/* Output Settings Section */}
      <Collapsible open={openSections.output} onOpenChange={() => toggleSection('output')}>
        <div
          className="rounded-lg border overflow-hidden"
          style={{ borderColor: openSections.output ? PHASES[1].strongColor : undefined }}
        >
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-4 hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-3">
                <div
                  className="p-2 rounded-lg"
                  style={{ backgroundColor: PHASES[1].lightColor }}
                >
                  <Activity className="h-5 w-5" style={{ color: PHASES[1].strongColor }} />
                </div>
                <div className="text-left">
                  <h3 className="font-semibold">Output Settings</h3>
                  <p className="text-sm text-muted-foreground">
                    JSONL{config.generation.output_formats?.filter((f: string) => f !== 'jsonl').length > 0
                      ? ` + ${config.generation.output_formats?.filter((f: string) => f !== 'jsonl').join(', ')}`
                      : ' (default)'}
                  </p>
                </div>
              </div>
              <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${openSections.output ? 'rotate-180' : ''}`} />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="border-t p-4 space-y-6">
              {/* Output Formats */}
              <div className="space-y-3">
                <Label className="text-base">Output Formats</Label>
                <p className="text-xs text-muted-foreground">
                  Select additional formats for the generated dataset. JSONL is always produced for internal processing.
                </p>
                <div className="grid gap-3 sm:grid-cols-3">
                  {OUTPUT_FORMATS.map((format) => {
                    const isAlwaysEnabled = (format as any).alwaysEnabled
                    const isChecked = isAlwaysEnabled || config.generation.output_formats?.includes(format.value)
                    return (
                      <label
                        key={format.value}
                        className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${
                          isAlwaysEnabled
                            ? 'border-primary/50 bg-primary/5 cursor-not-allowed opacity-75'
                            : isChecked
                              ? 'border-primary bg-primary/5 cursor-pointer'
                              : 'border-border hover:border-primary/50 cursor-pointer'
                        }`}
                      >
                        <Checkbox
                          checked={isChecked}
                          disabled={isAlwaysEnabled}
                          onCheckedChange={() => !isAlwaysEnabled && toggleOutputFormat(format.value)}
                        />
                        <div>
                          <div className="font-medium text-sm">{format.label}</div>
                          <div className="text-xs text-muted-foreground">{format.description}</div>
                        </div>
                      </label>
                    )
                  })}
                </div>
              </div>

              <Separator />

              {/* Dataset Mode */}
              <div className="space-y-3">
                <Label className="text-base">Dataset Mode</Label>
                <p className="text-xs text-muted-foreground">
                  Choose between explicit (with target terms and character offsets) or implicit (category-only) ABSA annotation
                </p>
                <div className="grid gap-3 sm:grid-cols-3">
                  {[
                    { value: 'both', label: 'Both', description: 'Generates explicit, then derives implicit version (saves API calls)' },
                    { value: 'explicit', label: 'Explicit', description: 'Includes target terms with character offsets (e.g., "pasta carbonara")' },
                    { value: 'implicit', label: 'Implicit', description: 'Category and polarity only, target="NULL" (e.g., FOOD#QUALITY)' },
                  ].map((mode) => (
                    <label
                      key={mode.value}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        config.generation.dataset_mode === mode.value
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <input
                        type="radio"
                        name="dataset_mode"
                        value={mode.value}
                        checked={config.generation.dataset_mode === mode.value}
                        onChange={() => updateConfig('generation.dataset_mode', mode.value)}
                        className="mt-0.5"
                      />
                      <div>
                        <div className="font-medium text-sm">{mode.label}</div>
                        <div className="text-xs text-muted-foreground">{mode.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

    </div>
  )
}

// EVALUATION Phase - metric selection
function EvaluationPhase({
  config,
  updateConfig,
  referenceFile,
  onReferenceFileChange,
  referenceFromInput,
  inheritedReferenceName,
  selfTestEnabled,
}: {
  config: any
  updateConfig: (path: string, value: any) => void
  referenceFile: File | null
  onReferenceFileChange: (file: File | null) => void
  referenceFromInput?: boolean // True when reference file is set from INPUT tab
  inheritedReferenceName?: string // File name of reference dataset inherited from a reused job
  selfTestEnabled?: boolean // True when self test mode is active (splits dataset)
}) {
  const METRIC_INFO: Record<string, { label: string; category: string; description: string; detail: string; requiresReference?: boolean }> = {
    bertscore: { label: 'BERTScore', category: 'Semantic', description: 'Contextual similarity using BERT embeddings', detail: 'Computes token-level cosine similarity between generated and reference texts using contextual BERT embeddings. Captures meaning beyond exact word matches. Scores range from 0 to 1, where higher means more semantically similar.', requiresReference: true },
    bleu: { label: 'BLEU', category: 'Lexical', description: 'N-gram overlap precision score', detail: 'Measures how many n-grams (1-4) in the generated text appear in the reference. Originally designed for machine translation evaluation. Scores range from 0 to 1, where higher means more overlap with reference text.', requiresReference: true },
    rouge_l: { label: 'ROUGE-L', category: 'Lexical', description: 'Longest common subsequence recall', detail: 'Finds the longest common subsequence between generated and reference texts to measure recall. Captures sentence-level structure similarity without requiring consecutive matches. Higher scores indicate better structural alignment.', requiresReference: true },
    moverscore: { label: 'MoverScore', category: 'Semantic', description: 'Earth mover distance on word embeddings', detail: 'Uses Word Mover\'s Distance on contextualized embeddings to measure the minimum "effort" to transform one text into another. More robust to paraphrasing than token-level metrics. Higher scores indicate greater semantic similarity.', requiresReference: true },
    distinct_1: { label: 'Distinct-1', category: 'Diversity', description: 'Unique unigram ratio', detail: 'Ratio of unique unigrams (single words) to total words across the generated corpus. Measures vocabulary diversity. Higher values indicate more lexically varied output with less repetition.' },
    distinct_2: { label: 'Distinct-2', category: 'Diversity', description: 'Unique bigram ratio', detail: 'Ratio of unique bigrams (word pairs) to total bigrams across the generated corpus. Captures phrasal diversity beyond single words. Higher values indicate more varied expressions and less formulaic text.' },
    self_bleu: { label: 'Self-BLEU', category: 'Diversity', description: 'Intra-corpus similarity (lower = more diverse)', detail: 'Computes BLEU score of each generated text against all other generated texts, then averages. Measures how similar generated reviews are to each other. Lower scores indicate greater diversity within the corpus.' },
  }

  // Reference metrics are enabled when a reference file is provided or self test is active
  const referenceMetricsEnabled = !!referenceFile || !!inheritedReferenceName || !!selfTestEnabled

  const toggleMetric = (metric: string) => {
    const current = config.evaluation.metrics as string[]
    if (current.includes(metric)) {
      if (current.length > 1) {
        updateConfig('evaluation.metrics', current.filter((m: string) => m !== metric))
      }
    } else {
      updateConfig('evaluation.metrics', [...current, metric])
    }
  }

  const selectAll = () => {
    updateConfig('evaluation.metrics', [...ALL_METRICS])
  }

  // Group metrics by category
  const categories = ['Lexical', 'Semantic', 'Diversity']

  // Count only enabled metrics (exclude Lexical/Semantic when no reference file)
  const enabledMetricsCount = (config.evaluation.metrics as string[]).filter(m => {
    const info = METRIC_INFO[m]
    if (info?.requiresReference && !referenceMetricsEnabled) return false
    return true
  }).length

  const totalEnabledMetrics = referenceMetricsEnabled
    ? ALL_METRICS.length
    : ALL_METRICS.filter(m => !METRIC_INFO[m]?.requiresReference).length

  return (
    <div className="space-y-4">
      {/* Reference Dataset Section */}
      <div
        className="rounded-lg border overflow-hidden"
        style={{ borderColor: PHASES[2].strongColor }}
      >
        <div className="p-4 space-y-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-semibold">Reference Dataset</h3>
              <Badge
                variant={referenceMetricsEnabled ? 'default' : 'secondary'}
                className="text-[10px] px-1.5"
              >
                {selfTestEnabled ? 'Self Test' : referenceFromInput ? 'From Input' : inheritedReferenceName ? 'Inherited' : referenceMetricsEnabled ? 'Provided' : 'Optional'}
              </Badge>
              <TooltipProvider delayDuration={200}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="h-4 w-4 text-muted-foreground/60 hover:text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-sm text-xs">
                    <p className="font-medium mb-1">Why provide a reference dataset?</p>
                    <p>Lexical and Semantic metrics (BLEU, ROUGE-L, BERTScore, MoverScore) compare generated reviews against reference reviews to measure quality.</p>
                    <p className="mt-1">Without a reference dataset, these metrics are skipped since comparing generated reviews against each other doesn't provide useful quality information.</p>
                    <p className="mt-1 text-muted-foreground">Use real reviews from the same domain for meaningful quality assessment.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              {selfTestEnabled
                ? 'Dataset will be split into two halves for self-referencing metric computation'
                : referenceFromInput
                ? 'Configured in Input tab'
                : inheritedReferenceName
                ? 'Inherited from the reused job'
                : 'For Lexical/Semantic metrics to be meaningful, provide a reference dataset'}
            </p>
          </div>

          {/* Reference Dataset Status Indicator */}
          {selfTestEnabled ? (
            <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium">Self Test Mode</span>
                <Badge variant="outline" className="text-xs border-green-500/30 text-green-600 dark:text-green-400">Active</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                The dataset will be split into two halves. Set A is evaluated against Set B as the reference, producing self-referencing metric scores.
              </p>
            </div>
          ) : referenceFile ? (
            <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium">{referenceFile.name}</span>
                <Badge variant="outline" className="text-xs border-green-500/30 text-green-600 dark:text-green-400">Provided</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Lexical and Semantic metrics will compare generated reviews against this reference dataset.
              </p>
              <div className="flex items-start gap-2 rounded-md bg-muted/50 p-2 mt-2">
                <Info className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
                <p className="text-xs text-muted-foreground">
                  To change the reference dataset, update it in the Input tab
                </p>
              </div>
            </div>
          ) : inheritedReferenceName ? (
            <div className="rounded-lg border border-green-500/30 bg-green-500/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium">{inheritedReferenceName}</span>
                <Badge variant="outline" className="text-xs border-green-500/30 text-green-600 dark:text-green-400">Inherited</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Lexical and Semantic metrics will compare generated reviews against the reference dataset from the reused job.
              </p>
            </div>
          ) : (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 space-y-2">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                <span className="text-sm font-medium">No reference dataset</span>
                <Badge variant="outline" className="text-xs border-amber-500/30 text-amber-600 dark:text-amber-400">Optional</Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Lexical and Semantic metrics will be skipped. Only Diversity metrics (Distinct-1, Distinct-2, Self-BLEU) will be computed.
              </p>
              <div className="flex items-start gap-2 rounded-md bg-muted/50 p-2 mt-2">
                <Info className="h-3 w-3 mt-0.5 text-muted-foreground shrink-0" />
                <p className="text-xs text-muted-foreground">
                  To add a reference dataset, upload one in the Input tab with "Use for MDQA evaluation" enabled
                </p>
              </div>
            </div>
          )}

          <div className="rounded-lg p-3 bg-muted/50">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">Note:</span> Use real reviews for meaningful comparison. Using another AI-generated dataset doesn't indicate quality—high scores would just mean the generated text closely matches other AI output.
            </p>
          </div>
        </div>
      </div>

      {/* MDQA Metrics Section */}
      <div
        className="rounded-lg border overflow-hidden"
        style={{ borderColor: PHASES[2].strongColor }}
      >
        <div className="p-4 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold">MDQA Metrics</h3>
              <p className="text-sm text-muted-foreground">
                Select the quality assessment metrics to compute
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={selectAll}
              disabled={config.evaluation.metrics.length === ALL_METRICS.length}
            >
              Select All
            </Button>
          </div>

          {categories.map((category) => {
            const isReferenceCategory = category === 'Lexical' || category === 'Semantic'
            const categoryDisabled = isReferenceCategory && !referenceMetricsEnabled

            return (
              <div key={category} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label className="text-sm text-muted-foreground">{category}</Label>
                  {isReferenceCategory && (
                    <Badge
                      variant={referenceMetricsEnabled ? 'default' : 'secondary'}
                      className="text-[10px] px-1.5"
                    >
                      {referenceMetricsEnabled ? 'Reference-based' : 'Disabled'}
                    </Badge>
                  )}
                </div>
                <div className={`grid gap-2 sm:grid-cols-2 ${categoryDisabled ? 'opacity-40 pointer-events-none' : ''}`}>
                  {Object.entries(METRIC_INFO)
                    .filter(([, info]) => info.category === category)
                    .map(([key, info]) => {
                      const isSelected = (config.evaluation.metrics as string[]).includes(key)
                      return (
                        <label
                          key={key}
                          className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                            isSelected ? 'border-border bg-muted/30' : 'opacity-50 hover:opacity-75'
                          }`}
                        >
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={() => toggleMetric(key)}
                            disabled={categoryDisabled}
                          />
                          <div className="flex-1">
                            <div className="flex items-center gap-1.5">
                              <span className="font-medium text-sm">{info.label}</span>
                              <TooltipProvider delayDuration={200}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <Info className="h-3.5 w-3.5 text-muted-foreground/60 hover:text-muted-foreground cursor-help" />
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="max-w-xs text-xs">
                                    {info.detail}
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                            <div className="text-xs text-muted-foreground">{info.description}</div>
                          </div>
                        </label>
                      )
                    })}
                </div>
              </div>
            )
          })}

          <div className="rounded-lg p-3 bg-muted/50">
            <p className="text-xs text-muted-foreground">
              {enabledMetricsCount} of {totalEnabledMetrics} metrics selected.
              {!referenceMetricsEnabled && (
                <span className="ml-1">
                  Lexical/Semantic metrics disabled (no reference dataset).
                </span>
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Aspect Categories Editor component
function AspectCategoriesEditor({ categories, domain, mode, onModeChange, onChange, extractedRefContext }: {
  categories: string[]
  domain: string
  mode: 'preset' | 'infer' | 'import' | 'ref_dataset'
  onModeChange: (mode: 'preset' | 'infer' | 'import' | 'ref_dataset') => void
  onChange: (cats: string[]) => void
  extractedRefContext?: ExtractedRefContext | null
}) {
  const [newCategory, setNewCategory] = useState('')
  const [presets, setPresets] = useState<Record<string, { label: string; categories: string[] }>>({})
  const [customPresets, setCustomPresets] = useState<Array<{ filename: string; name: string; categories: string[] }>>([])
  const [loading, setLoading] = useState(true)
  const [presetToDelete, setPresetToDelete] = useState<{ filename: string; name: string } | null>(null)
  const [deleting, setDeleting] = useState(false)

  // Fetch built-in presets and custom presets
  useEffect(() => {
    const pythonApiUrl = PYTHON_API_URL
    const fetchPresets = async () => {
      try {
        const [builtinRes, customRes] = await Promise.all([
          fetch(`${pythonApiUrl}/api/aspect-categories`),
          fetch(`${pythonApiUrl}/api/aspect-category-presets`),
        ])
        if (builtinRes.ok) setPresets(await builtinRes.json())
        if (customRes.ok) setCustomPresets(await customRes.json())
      } catch { /* ignore */ }
      setLoading(false)
    }
    fetchPresets()
  }, [])

  // Auto-populate from domain when categories are empty (only in preset mode)
  useEffect(() => {
    if (mode === 'preset' && categories.length === 0 && domain && presets[domain.toLowerCase()]) {
      onChange(presets[domain.toLowerCase()].categories)
    }
  }, [domain, presets])

  // Auto-load categories when ref_dataset mode is selected
  useEffect(() => {
    if (mode === 'ref_dataset' && extractedRefContext?.aspect_categories?.length) {
      onChange(extractedRefContext.aspect_categories)
    }
  }, [mode, extractedRefContext])

  // Auto-select ref_dataset mode when extractedRefContext becomes available
  useEffect(() => {
    if (extractedRefContext?.aspect_categories?.length && mode === 'infer') {
      onModeChange('ref_dataset')
    }
  }, [extractedRefContext])

  const addCategory = () => {
    const cat = newCategory.trim().toUpperCase()
    if (cat && !categories.includes(cat)) {
      onChange([...categories, cat])
      setNewCategory('')
    }
  }

  const removeCategory = (cat: string) => {
    onChange(categories.filter(c => c !== cat))
  }

  const loadPreset = (cats: string[]) => {
    onChange(cats)
  }

  const handleFileImport = async (file: File) => {
    try {
      const text = await file.text()
      const pythonApiUrl = PYTHON_API_URL
      const res = await fetch(`${pythonApiUrl}/api/extract-aspect-categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text, format: 'auto' }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      onChange(data.categories)
      toast.success(`Imported ${data.count} aspect categories from ${file.name}`)
    } catch (err: any) {
      toast.error(`Import failed: ${err.message}`)
    }
  }

  const deletePreset = async () => {
    if (!presetToDelete) return
    setDeleting(true)
    try {
      const pythonApiUrl = PYTHON_API_URL
      const res = await fetch(`${pythonApiUrl}/api/delete-aspect-preset/${encodeURIComponent(presetToDelete.filename)}`, {
        method: 'DELETE',
      })
      if (!res.ok) throw new Error(await res.text())
      setCustomPresets(prev => prev.filter(p => p.filename !== presetToDelete.filename))
      toast.success(`Deleted preset: ${presetToDelete.name}`)
    } catch (err: any) {
      toast.error(`Delete failed: ${err.message}`)
    }
    setDeleting(false)
    setPresetToDelete(null)
  }

  return (
    <div className="space-y-3">
      {/* Delete preset confirmation dialog */}
      <AlertDialog open={!!presetToDelete} onOpenChange={(open) => !open && setPresetToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Preset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the preset "{presetToDelete?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={deletePreset}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? 'Deleting...' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Mode selector */}
      <div className="flex gap-1 p-1 rounded-md border bg-muted/30 w-fit flex-wrap">
        {[
          ...(extractedRefContext?.aspect_categories?.length ? [{ value: 'ref_dataset' as const, label: 'Reference Dataset', icon: Database }] : []),
          { value: 'infer' as const, label: 'Infer from Reviews', icon: Sparkles },
          { value: 'import' as const, label: 'Import from File', icon: FileUp },
          { value: 'preset' as const, label: 'From Preset', icon: Search },
        ].map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            type="button"
            className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded transition-colors ${
              mode === value
                ? 'bg-background shadow-sm font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => onModeChange(value)}
          >
            <Icon className="h-3 w-3" />
            {label}
          </button>
        ))}
      </div>

      {/* Reference Dataset mode */}
      {mode === 'ref_dataset' && (
        <>
          <div className="flex items-start gap-2 rounded-md bg-blue-500/10 border border-blue-500/20 p-3">
            <Database className="h-4 w-4 mt-0.5 text-blue-500 shrink-0" />
            <div className="space-y-1">
              <p className="text-xs font-medium text-blue-700 dark:text-blue-300">
                Categories from Reference Dataset
                <RefContextIndicator />
              </p>
              <p className="text-xs text-muted-foreground">
                {extractedRefContext?.aspect_categories?.length || 0} categories extracted from the reference dataset.
                These will be used for generating ABSA-annotated reviews.
              </p>
            </div>
          </div>

          {/* Category badges */}
          <div className="flex flex-wrap gap-1.5 min-h-[32px] p-2 rounded-md border bg-muted/20">
            {categories.length === 0 && (
              <span className="text-xs text-muted-foreground">No categories extracted from reference dataset.</span>
            )}
            {categories.map((cat) => (
              <Badge key={cat} variant="secondary" className="text-xs gap-1 py-0.5">
                {cat}
                <button type="button" onClick={() => removeCategory(cat)} className="hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>

          {/* Add custom category */}
          <div className="flex gap-2">
            <Input
              placeholder="e.g., FOOD#QUALITY"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addCategory())}
              className="h-8 text-xs"
            />
            <Button type="button" size="sm" onClick={addCategory} className="h-8">
              Add
            </Button>
          </div>
        </>
      )}

      {/* Preset mode */}
      {mode === 'preset' && (
        <>
          <div className="flex items-center gap-2">
            <Select onValueChange={(val) => {
              if (val.startsWith('builtin:')) {
                const key = val.replace('builtin:', '')
                if (presets[key]) loadPreset(presets[key].categories)
              } else if (val.startsWith('custom:')) {
                const idx = parseInt(val.replace('custom:', ''))
                if (customPresets[idx]) loadPreset(customPresets[idx].categories)
              }
            }}>
              <SelectTrigger className="w-[200px] h-8 text-xs">
                <SelectValue placeholder="Load from preset..." />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(presets).map(([key, preset]) => (
                  <SelectItem key={key} value={`builtin:${key}`}>{preset.label}</SelectItem>
                ))}
                {customPresets.length > 0 && (
                  <>
                    <div className="px-2 py-1 text-xs text-muted-foreground font-medium">Extracted Presets</div>
                    {customPresets.map((p, i) => (
                      <div key={`custom-${i}`} className="relative">
                        <SelectItem value={`custom:${i}`} className="pr-8" hideIndicator>{p.name}</SelectItem>
                        <button
                          type="button"
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive"
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            setPresetToDelete({ filename: p.filename, name: p.name })
                          }}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </>
                )}
              </SelectContent>
            </Select>
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear all
            </button>
          </div>

          {/* Category badges */}
          <div className="flex flex-wrap gap-1.5 min-h-[32px] p-2 rounded-md border bg-muted/20">
            {categories.length === 0 && (
              <span className="text-xs text-muted-foreground">No categories selected. Pick a domain preset or add manually.</span>
            )}
            {categories.map((cat) => (
              <Badge key={cat} variant="secondary" className="text-xs gap-1 py-0.5">
                {cat}
                <button type="button" onClick={() => removeCategory(cat)} className="hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>

          {/* Add custom category */}
          <div className="flex gap-2">
            <Input
              placeholder="e.g., FOOD#QUALITY"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addCategory())}
              className="h-8 text-xs"
            />
            <Button type="button" variant="outline" size="sm" onClick={addCategory} className="h-8 text-xs">
              Add
            </Button>
          </div>
        </>
      )}

      {/* Infer mode */}
      {mode === 'infer' && (
        <div className="rounded-md border bg-muted/20 p-3 space-y-1">
          <p className="text-sm">The generation LLM will autonomously assign aspect categories based on the review content.</p>
          <p className="text-xs text-muted-foreground">
            Categories are inferred from context — no predefined pool is enforced. Useful when you want the model to decide naturally.
          </p>
        </div>
      )}

      {/* Import mode */}
      {mode === 'import' && (
        <>
          <FileDropZone
            onFile={handleFileImport}
            accept=".jsonl,.csv,.xml,.json"
            placeholder="Import categories from a dataset file (.jsonl, .csv, .xml)"
          />

          {categories.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium">{categories.length} categories imported</span>
                <button
                  type="button"
                  onClick={() => onChange([])}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5 min-h-[32px] p-2 rounded-md border bg-muted/20">
                {categories.map((cat) => (
                  <Badge key={cat} variant="secondary" className="text-xs gap-1 py-0.5">
                    {cat}
                    <button type="button" onClick={() => removeCategory(cat)} className="hover:text-destructive">
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// Helper component for ref-context indicator
function RefContextIndicator() {
  return (
    <span className="text-xs text-blue-500 italic ml-1">
      (LLM-generated from reference dataset)
    </span>
  )
}

// Content components (extracted from original Step components)
function SubjectStepContent({ config, updateConfig, referenceOptions, referenceDatasetFile, extractedRefContext }: { config: any; updateConfig: (path: string, value: any) => void; referenceOptions?: { extractSubjectContext: boolean }; referenceDatasetFile?: File | null; extractedRefContext?: ExtractedRefContext | null }) {
  const [domainOpen, setDomainOpen] = useState(false)
  const prevExtractedRef = useRef<ExtractedRefContext | null | undefined>(null)

  const DOMAIN_PRESETS = ['General', 'Electronics', 'Restaurant', 'Hotel', 'Software', 'Service']

  // Auto-fill from extracted reference context (runs when extractedRefContext changes to a new object)
  useEffect(() => {
    if (extractedRefContext && extractedRefContext !== prevExtractedRef.current) {
      prevExtractedRef.current = extractedRefContext
      // Auto-fill subject query
      if (extractedRefContext.subject_query) {
        updateConfig('subject_profile.query', extractedRefContext.subject_query)
      }
      // Auto-fill additional context
      if (extractedRefContext.additional_context) {
        updateConfig('subject_profile.additional_context', extractedRefContext.additional_context)
      }
      // Auto-fill domain if confidence >= 50%
      if (extractedRefContext.domain?.value && extractedRefContext.domain.confidence >= 0.5) {
        updateConfig('subject_profile.domain', extractedRefContext.domain.value)
      }
      // Auto-fill region if confidence >= 50%
      if (extractedRefContext.region?.value && extractedRefContext.region.confidence >= 0.5) {
        updateConfig('subject_profile.region', extractedRefContext.region.value)
      }
    }
  }, [extractedRefContext, updateConfig])

  // Track if fields were auto-filled from reference
  const isQueryFromRef = extractedRefContext?.subject_query && config.subject_profile.query === extractedRefContext.subject_query
  const isContextFromRef = extractedRefContext?.additional_context && config.subject_profile.additional_context === extractedRefContext.additional_context
  const isDomainFromRef = extractedRefContext?.domain?.value && config.subject_profile.domain === extractedRefContext.domain.value
  const isRegionFromRef = extractedRefContext?.region?.value && config.subject_profile.region === extractedRefContext.region.value

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2 sm:col-span-2">
            <div className="flex items-center">
              <Label>Subject Query *</Label>
              {isQueryFromRef && <RefContextIndicator />}
            </div>
            <Input
              placeholder="e.g., iPhone 15 Pro, Starbucks coffee"
              value={config.subject_profile.query}
              onChange={(e) => updateConfig('subject_profile.query', e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              The product, service, or topic to generate reviews about
            </p>
          </div>
          <div className="space-y-2 sm:col-span-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Label>Additional Context</Label>
                {isContextFromRef && <RefContextIndicator />}
              </div>
              {!isContextFromRef && referenceOptions?.extractSubjectContext && referenceDatasetFile && (
                <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Will extract from reference
                </span>
              )}
            </div>
            <textarea
              className="flex min-h-[100px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-y"
              placeholder="e.g., Limited edition with only 1000 units produced, recently discontinued model"
              value={config.subject_profile.additional_context || ''}
              onChange={(e) => updateConfig('subject_profile.additional_context', e.target.value)}
              rows={5}
            />
            <p className="text-xs text-muted-foreground">
              {!isContextFromRef && referenceOptions?.extractSubjectContext && referenceDatasetFile
                ? 'Extracted context will be appended when job is created'
                : 'Optional context passed to all MAV agents for more targeted research'}
            </p>
          </div>
          <div className="space-y-2">
            <div className="flex items-center">
              <Label>Region</Label>
              {isRegionFromRef && <RefContextIndicator />}
            </div>
            <Input
              placeholder="united states"
              value={config.subject_profile.region}
              onChange={(e) => updateConfig('subject_profile.region', e.target.value)}
            />
            {extractedRefContext?.region?.reason && (
              <p className={`text-xs ${extractedRefContext.region.value ? 'text-muted-foreground' : 'text-amber-500'}`}>
                {extractedRefContext.region.reason}
              </p>
            )}
          </div>
          <div className="space-y-2">
            <div className="flex items-center">
              <Label>Domain</Label>
              {isDomainFromRef && <RefContextIndicator />}
            </div>
            <Popover open={domainOpen} onOpenChange={setDomainOpen}>
              <PopoverTrigger asChild>
                <button
                  type="button"
                  className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <span className={config.subject_profile.domain ? '' : 'text-muted-foreground'}>
                    {config.subject_profile.domain || 'Select or type...'}
                  </span>
                  <ChevronDown className="h-4 w-4 opacity-50" />
                </button>
              </PopoverTrigger>
              <PopoverContent className="w-[--radix-popover-trigger-width] p-2" align="start">
                <Input
                  placeholder="Type a custom domain..."
                  value={config.subject_profile.domain}
                  onChange={(e) => updateConfig('subject_profile.domain', e.target.value)}
                  className="mb-2"
                />
                <div className="flex flex-wrap gap-1">
                  {DOMAIN_PRESETS.map((d) => (
                    <button
                      key={d}
                      type="button"
                      className={`px-2 py-1 text-xs rounded-md border transition-colors ${
                        (config.subject_profile.domain || '').toLowerCase() === d.toLowerCase()
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'hover:bg-muted border-border'
                      }`}
                      onClick={() => {
                        updateConfig('subject_profile.domain', d.toLowerCase())
                        setDomainOpen(false)
                      }}
                    >
                      {d}
                    </button>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
          </div>

          {/* Aspect Categories */}
          <div className="space-y-2 sm:col-span-2">
            <Label>Aspect Categories</Label>
            <p className="text-xs text-muted-foreground mb-2">
              ABSA annotation categories for the generated dataset
            </p>
            <AspectCategoriesEditor
              categories={config.subject_profile.aspect_categories}
              domain={config.subject_profile.domain}
              mode={config.subject_profile.aspect_category_mode}
              onModeChange={(m) => updateConfig('subject_profile.aspect_category_mode', m)}
              onChange={(cats) => updateConfig('subject_profile.aspect_categories', cats)}
              extractedRefContext={extractedRefContext}
            />
          </div>
        </div>
    </div>
  )
}

// Intelligence Step Content (SIL + MAV) - separate collapsible
function IntelligenceStepContent({ config, updateConfig, constraints, modelValidations }: { config: any; updateConfig: (path: string, value: any) => void; constraints: UIConstraints; modelValidations: Record<string, ModelValidationState> }) {
  const { providers, groupedModels, loading: modelsLoading } = useOpenRouterModels()
  const silEnabled = config.ablation.sil_enabled
  const mavEnabled = config.ablation.mav_enabled
  const c = constraints.constraints

  const updateMavModel = (index: number, modelId: string) => {
    const newModels = [...config.subject_profile.mav.models]
    newModels[index] = modelId
    updateConfig('subject_profile.mav.models', newModels)
  }

  return (
    <div className="space-y-6">
        {/* SIL - Subject Intelligence Layer */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <Label className="text-base">Subject Intelligence Layer (SIL)</Label>
              <p className="text-xs text-muted-foreground">
                Web search grounding for factual accuracy
              </p>
            </div>
            <Switch
              checked={silEnabled}
              onCheckedChange={(v) => updateConfig('ablation.sil_enabled', v)}
            />
          </div>
          <AblationSection enabled={silEnabled} effect="No web search, LLM uses internal knowledge only">
            <div className="rounded-lg border bg-muted/30 p-4 space-y-2">
              <p className="text-sm">
                SIL performs web searches to gather real, up-to-date information about the subject:
              </p>
              <ul className="text-xs text-muted-foreground space-y-1 ml-4 list-disc">
                <li>Each MAV agent independently performs its own SIL web research</li>
                <li>Product specifications, features, and pricing</li>
                <li>Common praise points and complaints from real reviews</li>
                <li>Use cases and target audience information</li>
                <li>Competitor comparisons and market positioning</li>
              </ul>
              <p className="text-xs text-muted-foreground pt-2">
                This ensures generated reviews mention accurate, verifiable details rather than hallucinated information.
              </p>
            </div>
          </AblationSection>
        </div>

        <Separator />

        {/* MAV - Multi-Agent Verification */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <Label className="text-base">Multi-Agent Verification (MAV)</Label>
              <p className="text-xs text-muted-foreground">
                Cross-validate facts using multiple LLMs (2/3 consensus)
              </p>
            </div>
            <Switch
              checked={mavEnabled}
              onCheckedChange={(v) => updateConfig('ablation.mav_enabled', v)}
            />
          </div>
          <AblationSection enabled={mavEnabled} effect="Single model, no cross-validation">
            <div className="space-y-4">
              <div className="rounded-lg border bg-muted/30 p-4 space-y-2">
                <p className="text-sm">
                  MAV uses multiple models to independently research the subject via a 4-round query-based protocol:
                </p>
                <ul className="text-xs text-muted-foreground space-y-1 ml-4 list-disc">
                  <li>Each model generates neutral factual queries about the subject</li>
                  <li>Queries are pooled, deduplicated, and answered independently by all models</li>
                  <li>Each model reviews others' answers and votes on agreement (LLM-judged consensus)</li>
                  <li>A fact passes if it receives mutual agreement from 2+ models</li>
                  <li>Using different providers is recommended for true independence</li>
                </ul>
              </div>

              {modelsLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading models...</span>
                </div>
              ) : (
                <>
                  <p className="text-sm text-muted-foreground">
                    Select 2-3 models for verification (at least 2 required):
                  </p>
                  <div className="space-y-3">
                    {[0, 1, 2].map((index) => (
                      <div key={index} className="space-y-1">
                        <Label className="text-xs text-muted-foreground">Model {index + 1}{index < 2 ? ' *' : ' (optional)'}</Label>
                        <LLMSelector
                          providers={providers}
                          groupedModels={groupedModels}
                          loading={modelsLoading}
                          value={config.subject_profile.mav.models[index]}
                          onChange={(v) => updateMavModel(index, v)}
                          disabledModels={config.subject_profile.mav.models.filter((_: string, i: number) => i !== index)}
                          placeholder="Select model..."
                          validationStatus={modelValidations[`mav-${index}`]?.status}
                          validationError={modelValidations[`mav-${index}`]?.error}
                        />
                      </div>
                    ))}
                  </div>
                  {config.subject_profile.mav.models.filter((m: string) => m).length < 2 && (
                    <p className="text-xs text-destructive">
                      Please select at least 2 models to continue
                    </p>
                  )}

                  {/* Query Dedup Threshold Slider (Round 2: embedding-based) */}
                  <Separator className="my-4" />
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-sm">Query Dedup Threshold (τ)</Label>
                        <p className="text-xs text-muted-foreground">
                          How similar queries must be to merge during deduplication (Round 2)
                        </p>
                      </div>
                      <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                        {config.subject_profile.mav.similarity_threshold.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[config.subject_profile.mav.similarity_threshold]}
                      onValueChange={([v]) => updateConfig('subject_profile.mav.similarity_threshold', v)}
                      min={c.mav_similarity_threshold.min}
                      max={c.mav_similarity_threshold.max}
                      step={c.mav_similarity_threshold.step}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Lenient (fewer queries)</span>
                      <span>Strict (more queries)</span>
                    </div>
                  </div>

                  {/* MAV Queries Cap */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-sm">Queries Cap</Label>
                        <p className="text-xs text-muted-foreground">
                          Max factual queries after deduplication
                        </p>
                      </div>
                      <span className="text-sm font-mono bg-muted px-2 py-1 rounded">
                        {config.subject_profile.mav.max_queries}
                      </span>
                    </div>
                    <Slider
                      value={[config.subject_profile.mav.max_queries]}
                      onValueChange={([v]) => updateConfig('subject_profile.mav.max_queries', v)}
                      min={c.mav_max_queries.min}
                      max={c.mav_max_queries.max}
                      step={c.mav_max_queries.step}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Fewer (faster)</span>
                      <span>More (thorough)</span>
                    </div>
                  </div>

                  {/* Free model privacy notice */}
                  {config.subject_profile.mav.models.some((m: string) => m?.includes(':free')) && (
                    <div className="flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                      <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-xs">
                        <span className="font-medium">Free models selected.</span> These require{' '}
                        <a
                          href="https://openrouter.ai/settings/privacy"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline hover:text-amber-600"
                        >
                          OpenRouter privacy settings
                        </a>{' '}
                        to allow free endpoints. Your prompts may be used for training.
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          </AblationSection>

          {/* Single-Agent Verification (SAV) mode when MAV is disabled */}
          {!mavEnabled && (
            <div className="space-y-4 pt-2">
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />
                  <span className="text-sm font-medium">Single-Agent Verification (SAV) Mode</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Using a single model without cross-validation. Facts will not be verified across multiple LLMs.
                </p>
              </div>
              <p className="text-sm text-muted-foreground">
                Select a model for subject intelligence gathering:
              </p>
              {modelsLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Loading models...</span>
                </div>
              ) : (
                <>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Model *</Label>
                    <LLMSelector
                      providers={providers}
                      groupedModels={groupedModels}
                      loading={modelsLoading}
                      value={config.subject_profile.mav.models[0]}
                      onChange={(v) => updateMavModel(0, v)}
                      placeholder="Select model..."
                      validationStatus={modelValidations['mav-0']?.status}
                      validationError={modelValidations['mav-0']?.error}
                    />
                  </div>
                  {!config.subject_profile.mav.models[0] && (
                    <p className="text-xs text-destructive">
                      Please select a model to continue
                    </p>
                  )}

                  {/* Free model privacy notice */}
                  {config.subject_profile.mav.models[0]?.includes(':free') && (
                    <div className="flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
                      <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-xs">
                        <span className="font-medium">Free model selected.</span> Requires{' '}
                        <a
                          href="https://openrouter.ai/settings/privacy"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline hover:text-amber-600"
                        >
                          OpenRouter privacy settings
                        </a>{' '}
                        to allow free endpoints. Your prompts may be used for training.
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
    </div>
  )
}

function ReviewerStepContent({ config, updateConfig, constraints, referenceOptions, referenceDatasetFile, extractedRefContext }: { config: any; updateConfig: (path: string, value: any) => void; constraints: UIConstraints; referenceOptions?: { extractReviewerContext: boolean }; referenceDatasetFile?: File | null; extractedRefContext?: ExtractedRefContext | null }) {
  const ageEnabled = config.ablation.age_enabled
  const sexEnabled = config.ablation.sex_enabled

  // Handle both old format (male_ratio) and new format (male/female/unknown)
  const sexDist = config.reviewer_profile.sex_distribution
  const malePercent = sexDist.male !== undefined
    ? Math.round(sexDist.male * 100)
    : Math.round((sexDist.male_ratio || 0.5) * 100)
  const femalePercent = sexDist.female !== undefined
    ? Math.round(sexDist.female * 100)
    : 100 - malePercent
  const unknownPercent = sexDist.unknown !== undefined
    ? Math.round(sexDist.unknown * 100)
    : 0

  const c = constraints.constraints

  // Track previous extraction for auto-fill detection
  const prevExtractedRef = useRef<ExtractedRefContext | null | undefined>(null)
  const isContextFromRef = extractedRefContext?.reviewer_context && config.reviewer_profile.additional_context === extractedRefContext.reviewer_context
  const isSexFromRef = extractedRefContext?.sex_distribution &&
    Math.abs((sexDist.male || 0) - extractedRefContext.sex_distribution.male) < 0.01 &&
    Math.abs((sexDist.female || 0) - extractedRefContext.sex_distribution.female) < 0.01

  // Auto-fill from extracted reference context (runs when extractedRefContext changes to a new object)
  useEffect(() => {
    if (extractedRefContext && extractedRefContext !== prevExtractedRef.current) {
      prevExtractedRef.current = extractedRefContext
      // Auto-fill sex distribution
      if (extractedRefContext.sex_distribution) {
        updateConfig('reviewer_profile.sex_distribution', {
          male: extractedRefContext.sex_distribution.male,
          female: extractedRefContext.sex_distribution.female,
          unknown: extractedRefContext.sex_distribution.unknown,
        })
      }
      // Auto-fill reviewer context
      if (extractedRefContext.reviewer_context) {
        updateConfig('reviewer_profile.additional_context', extractedRefContext.reviewer_context)
      }
      // Disable age range since age detection from text is unreliable
      updateConfig('ablation.age_enabled', false)
    }
  }, [extractedRefContext, updateConfig])

  // Track which sex sliders are locked
  const [lockedSex, setLockedSex] = useState<Record<'male' | 'female' | 'unknown', boolean>>({
    male: false,
    female: false,
    unknown: false,
  })

  const toggleSexLock = (key: 'male' | 'female' | 'unknown') => {
    setLockedSex(prev => ({ ...prev, [key]: !prev[key] }))
  }

  // Linked sex sliders - when one changes, only unlocked others adjust
  const updateLinkedSex = (changed: 'male' | 'female' | 'unknown', newValue: number) => {
    const current = {
      male: sexDist.male ?? (sexDist.male_ratio ?? 0.5),
      female: sexDist.female ?? (1 - (sexDist.male_ratio ?? 0.5)),
      unknown: sexDist.unknown ?? 0,
    }

    // Get unlocked others (excluding locked sliders and the changed one)
    const unlockedOthers = (['male', 'female', 'unknown'] as const)
      .filter(k => k !== changed && !lockedSex[k])

    // Calculate locked sum (values that can't change)
    const lockedSum = (['male', 'female', 'unknown'] as const)
      .filter(k => k !== changed && lockedSex[k])
      .reduce((sum, k) => sum + current[k], 0)

    // Max value this slider can have is 100% minus locked values
    const maxValue = 1 - lockedSum
    newValue = Math.min(newValue, maxValue)

    // Sum of unlocked others
    const unlockedOtherSum = unlockedOthers.reduce((sum, k) => sum + current[k], 0)

    let newSex: Record<string, number> = { ...current, [changed]: newValue }

    // Calculate remaining space for unlocked others
    const remaining = 1 - newValue - lockedSum

    if (unlockedOthers.length === 0) {
      // No unlocked others to adjust - just set the value (capped at max)
    } else if (unlockedOtherSum === 0) {
      // All unlocked others are at 0 - distribute equally
      const share = remaining / unlockedOthers.length
      unlockedOthers.forEach(key => {
        newSex[key] = Math.max(0, share)
      })
    } else {
      // Distribute proportionally among unlocked others
      unlockedOthers.forEach(key => {
        const ratio = current[key] / unlockedOtherSum
        newSex[key] = Math.max(0, Math.min(1, remaining * ratio))
      })

      // Ensure unlocked values sum correctly (handle floating point issues)
      const unlockedTotal = unlockedOthers.reduce((sum, k) => sum + newSex[k], 0)
      if (Math.abs(unlockedTotal - remaining) > 0.001 && unlockedOthers.length > 0) {
        const largestUnlocked = unlockedOthers.reduce((a, b) => newSex[a] > newSex[b] ? a : b)
        newSex[largestUnlocked] = Math.max(0, newSex[largestUnlocked] + (remaining - unlockedTotal))
      }
    }

    // Round to 2 decimal places
    Object.keys(newSex).forEach(k => {
      newSex[k] = Math.round(newSex[k] * 100) / 100
    })

    updateConfig('reviewer_profile.sex_distribution', newSex)
  }

  return (
    <div className="space-y-6">
      {/* Age Range - controlled by ablation */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <Label className="text-base">Age Range</Label>
              <p className="text-xs text-muted-foreground">
                Control the age distribution of generated reviewer personas
              </p>
            </div>
            <Switch
              checked={ageEnabled}
              onCheckedChange={(v) => updateConfig('ablation.age_enabled', v)}
            />
          </div>
          {/* Info note when reference dataset is used */}
          {extractedRefContext && (
            <div className="flex items-start gap-2 p-3 bg-muted/50 rounded-md text-xs text-muted-foreground">
              <Info className="h-4 w-4 shrink-0 mt-0.5" />
              <span>
                Age detection from review text is unreliable. Using default range. Adjust manually if needed.
              </span>
            </div>
          )}
          <AblationSection enabled={ageEnabled} effect={`Random ages (${c.age_range.min}-${c.age_range.max}) assigned`}>
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Range: {config.reviewer_profile.age_range[0]} - {config.reviewer_profile.age_range[1]} years</span>
              </div>
              <Slider
                value={config.reviewer_profile.age_range}
                onValueChange={(v) => updateConfig('reviewer_profile.age_range', v)}
                min={c.age_range.min}
                max={c.age_range.max}
                step={c.age_range.step}
              />
            </div>
          </AblationSection>
        </div>

        <Separator />

        {/* Sex Distribution - controlled by ablation */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center">
              <Label className="text-base">Sex Distribution</Label>
              {isSexFromRef && <RefContextIndicator />}
            </div>
            <Switch
              checked={sexEnabled}
              onCheckedChange={(v) => updateConfig('ablation.sex_enabled', v)}
            />
          </div>
          <p className="text-xs text-muted-foreground -mt-2">
            Control the male/female/unknown distribution of generated reviewer personas
          </p>
          <AblationSection enabled={sexEnabled} effect="All reviewers have unspecified sex">
            <div className="space-y-4">
              {/* Percentage labels row */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-blue-500 font-medium">Male {malePercent}%</span>
                <span className="text-pink-500 font-medium">Female {femalePercent}%</span>
                <span className="text-gray-500 font-medium">Unknown {unknownPercent}%</span>
              </div>

              {/* Visual stacked bar */}
              <div className="h-3 rounded-full overflow-hidden flex">
                <div
                  className="bg-blue-500 transition-all"
                  style={{ width: `${malePercent}%` }}
                />
                <div
                  className="bg-pink-500 transition-all"
                  style={{ width: `${femalePercent}%` }}
                />
                <div
                  className="bg-gray-400 transition-all"
                  style={{ width: `${unknownPercent}%` }}
                />
              </div>

              {/* Linked sliders with lock buttons */}
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-blue-500">Male</Label>
                    <button
                      type="button"
                      onClick={() => toggleSexLock('male')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedSex.male ? 'text-blue-500' : 'text-muted-foreground'}`}
                      title={lockedSex.male ? 'Unlock' : 'Lock'}
                    >
                      {lockedSex.male ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[malePercent]}
                    onValueChange={([v]) => updateLinkedSex('male', v / 100)}
                    min={0}
                    max={100}
                    step={1}
                    disabled={lockedSex.male}
                    className={lockedSex.male ? 'opacity-50' : ''}
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-pink-500">Female</Label>
                    <button
                      type="button"
                      onClick={() => toggleSexLock('female')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedSex.female ? 'text-pink-500' : 'text-muted-foreground'}`}
                      title={lockedSex.female ? 'Unlock' : 'Lock'}
                    >
                      {lockedSex.female ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[femalePercent]}
                    onValueChange={([v]) => updateLinkedSex('female', v / 100)}
                    min={0}
                    max={100}
                    step={1}
                    disabled={lockedSex.female}
                    className={lockedSex.female ? 'opacity-50' : ''}
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-gray-500">Unknown</Label>
                    <button
                      type="button"
                      onClick={() => toggleSexLock('unknown')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedSex.unknown ? 'text-gray-500' : 'text-muted-foreground'}`}
                      title={lockedSex.unknown ? 'Unlock' : 'Lock'}
                    >
                      {lockedSex.unknown ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[unknownPercent]}
                    onValueChange={([v]) => updateLinkedSex('unknown', v / 100)}
                    min={0}
                    max={100}
                    step={1}
                    disabled={lockedSex.unknown}
                    className={lockedSex.unknown ? 'opacity-50' : ''}
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Reviewers with "unknown" sex won't mention their gender in reviews.
              </p>
            </div>
          </AblationSection>
        </div>

        <Separator />

        {/* Persona Ratio */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Label className="text-base">Persona Ratio</Label>
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-4 w-4 text-muted-foreground/60 hover:text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-sm text-xs">
                  <p>Percentage of unique personas relative to review count. 90% means most reviews get a unique persona, with some natural overlap.</p>
                  <p className="mt-1 text-muted-foreground">Personas are generated during composition and assigned via shuffled round-robin.</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <p className="text-xs text-muted-foreground -mt-2">
            Controls how many unique reviewer personas are generated relative to the total review count
          </p>
          <div className="flex items-center gap-4">
            <Slider
              value={[Math.round((config.reviewer_profile.persona_ratio ?? 0.9) * 100)]}
              onValueChange={([v]) => updateConfig('reviewer_profile.persona_ratio', v / 100)}
              min={10}
              max={100}
              step={5}
              className="flex-1"
            />
            <span className="text-sm font-medium tabular-nums w-12 text-right">
              {Math.round((config.reviewer_profile.persona_ratio ?? 0.9) * 100)}%
            </span>
          </div>
        </div>

        <Separator />

        {/* Additional Context */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Label className="text-base">Additional Context</Label>
              {isContextFromRef && <RefContextIndicator />}
            </div>
            {!isContextFromRef && referenceOptions?.extractReviewerContext && referenceDatasetFile && (
              <span className="text-xs text-amber-600 dark:text-amber-400 flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                Will extract from reference
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground -mt-2">
            Describe the typical reviewer persona for this subject to generate more realistic reviews
          </p>
          <Textarea
            placeholder="e.g., Visitors to this attraction are typically tourists or families. They appreciate historical significance but may complain about crowds or parking..."
            value={config.reviewer_profile.additional_context || ''}
            onChange={(e) => updateConfig('reviewer_profile.additional_context', e.target.value)}
            className="min-h-[120px] resize-y"
            rows={5}
          />
          <p className="text-xs text-muted-foreground">
            {!isContextFromRef && referenceOptions?.extractReviewerContext && referenceDatasetFile
              ? 'Extracted context will be appended when job is created'
              : 'This context will be included verbatim in the generation prompt to ensure reviews reflect realistic perspectives.'}
          </p>
        </div>
    </div>
  )
}

function AttributesStepContent({ config, updateConfig, constraints, extractedRefContext }: { config: any; updateConfig: (path: string, value: any) => void; constraints: UIConstraints; extractedRefContext?: ExtractedRefContext | null }) {
  const polarityEnabled = config.ablation.polarity_enabled
  const noiseEnabled = config.ablation.noise_enabled
  const c = constraints.constraints
  const prevExtractedRef = useRef<ExtractedRefContext | null | undefined>(null)

  // Track if values were auto-filled from reference
  const isPolarityFromRef = extractedRefContext?.polarity &&
    Math.abs(config.attributes_profile.polarity.positive - extractedRefContext.polarity.positive) < 0.01 &&
    Math.abs(config.attributes_profile.polarity.negative - extractedRefContext.polarity.negative) < 0.01
  const isLengthFromRef = extractedRefContext?.review_length &&
    config.attributes_profile.length_range[0] === extractedRefContext.review_length.suggested_range[0] &&
    config.attributes_profile.length_range[1] === extractedRefContext.review_length.suggested_range[1]

  // Auto-fill from extracted reference context (runs when extractedRefContext changes to a new object)
  useEffect(() => {
    if (extractedRefContext && extractedRefContext !== prevExtractedRef.current) {
      prevExtractedRef.current = extractedRefContext
      // Auto-fill polarity
      if (extractedRefContext.polarity) {
        updateConfig('attributes_profile.polarity', {
          positive: extractedRefContext.polarity.positive,
          neutral: extractedRefContext.polarity.neutral,
          negative: extractedRefContext.polarity.negative,
        })
      }
      // Auto-fill review length
      if (extractedRefContext.review_length?.suggested_range) {
        updateConfig('attributes_profile.length_range', extractedRefContext.review_length.suggested_range)
      }
      // Auto-select reference noise preset when noise data is available
      if (extractedRefContext.noise) {
        updateConfig('attributes_profile.noise.preset', 'ref_dataset')
        updateConfig('attributes_profile.noise.typo_rate', extractedRefContext.noise.typo_rate || 0.01)
        updateConfig('attributes_profile.noise.colloquialism', extractedRefContext.noise.has_colloquialisms ?? true)
      }
    }
  }, [extractedRefContext, updateConfig])

  // Track which polarity sliders are locked
  const [lockedPolarity, setLockedPolarity] = useState<Record<'positive' | 'neutral' | 'negative', boolean>>({
    positive: false,
    neutral: false,
    negative: false,
  })

  const toggleLock = (key: 'positive' | 'neutral' | 'negative') => {
    setLockedPolarity(prev => ({ ...prev, [key]: !prev[key] }))
  }


  // Linked polarity sliders - when one changes, only unlocked others adjust
  const updateLinkedPolarity = (changed: 'positive' | 'neutral' | 'negative', newValue: number) => {
    const current = config.attributes_profile.polarity

    // Get unlocked others (excluding locked sliders and the changed one)
    const unlockedOthers = (['positive', 'neutral', 'negative'] as const)
      .filter(k => k !== changed && !lockedPolarity[k])

    // Calculate locked sum (values that can't change)
    const lockedSum = (['positive', 'neutral', 'negative'] as const)
      .filter(k => k !== changed && lockedPolarity[k])
      .reduce((sum, k) => sum + current[k], 0)

    // Max value this slider can have is 100% minus locked values
    const maxValue = 1 - lockedSum
    newValue = Math.min(newValue, maxValue)

    // Sum of unlocked others
    const unlockedOtherSum = unlockedOthers.reduce((sum, k) => sum + current[k], 0)

    let newPolarity: Record<string, number> = { ...current, [changed]: newValue }

    // Calculate remaining space for unlocked others
    const remaining = 1 - newValue - lockedSum

    if (unlockedOthers.length === 0) {
      // No unlocked others to adjust - just set the value (capped at max)
      // Nothing else to do
    } else if (unlockedOtherSum === 0) {
      // All unlocked others are at 0 - distribute equally
      const share = remaining / unlockedOthers.length
      unlockedOthers.forEach(key => {
        newPolarity[key] = Math.max(0, share)
      })
    } else {
      // Distribute proportionally among unlocked others
      unlockedOthers.forEach(key => {
        const ratio = current[key] / unlockedOtherSum
        newPolarity[key] = Math.max(0, Math.min(1, remaining * ratio))
      })

      // Ensure unlocked values sum correctly (handle floating point issues)
      const unlockedTotal = unlockedOthers.reduce((sum, k) => sum + newPolarity[k], 0)
      if (Math.abs(unlockedTotal - remaining) > 0.001 && unlockedOthers.length > 0) {
        const largestUnlocked = unlockedOthers.reduce((a, b) => newPolarity[a] > newPolarity[b] ? a : b)
        newPolarity[largestUnlocked] = Math.max(0, newPolarity[largestUnlocked] + (remaining - unlockedTotal))
      }
    }

    // Round to 2 decimal places
    Object.keys(newPolarity).forEach(k => {
      newPolarity[k] = Math.round(newPolarity[k] * 100) / 100
    })

    updateConfig('attributes_profile.polarity', newPolarity)
  }

  return (
    <div className="space-y-6">
      {/* Polarity Distribution - controlled by ablation */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center">
                <Label className="text-base">Polarity Distribution</Label>
                {isPolarityFromRef && <RefContextIndicator />}
              </div>
              <p className="text-xs text-muted-foreground">
                Control the mix of sentiments within each review (sentence-level)
              </p>
            </div>
            <Switch
              checked={polarityEnabled}
              onCheckedChange={(v) => updateConfig('ablation.polarity_enabled', v)}
            />
          </div>
          <AblationSection enabled={polarityEnabled} effect="LLM decides sentiment naturally">
            <div className="space-y-4">
              {/* Percentage labels row */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-green-500 font-medium">Positive {Math.round(config.attributes_profile.polarity.positive * 100)}%</span>
                <span className="text-yellow-500 font-medium">Neutral {Math.round(config.attributes_profile.polarity.neutral * 100)}%</span>
                <span className="text-red-500 font-medium">Negative {Math.round(config.attributes_profile.polarity.negative * 100)}%</span>
              </div>

              {/* Visual stacked bar */}
              <div className="h-3 rounded-full overflow-hidden flex">
                <div
                  className="bg-green-500 transition-all"
                  style={{ width: `${config.attributes_profile.polarity.positive * 100}%` }}
                />
                <div
                  className="bg-yellow-500 transition-all"
                  style={{ width: `${config.attributes_profile.polarity.neutral * 100}%` }}
                />
                <div
                  className="bg-red-500 transition-all"
                  style={{ width: `${config.attributes_profile.polarity.negative * 100}%` }}
                />
              </div>

              {/* Linked sliders with lock buttons */}
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-green-500">Positive</Label>
                    <button
                      type="button"
                      onClick={() => toggleLock('positive')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedPolarity.positive ? 'text-green-500' : 'text-muted-foreground'}`}
                      title={lockedPolarity.positive ? 'Unlock' : 'Lock'}
                    >
                      {lockedPolarity.positive ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[config.attributes_profile.polarity.positive * 100]}
                    onValueChange={([v]) => updateLinkedPolarity('positive', v / 100)}
                    min={c.polarity.min}
                    max={c.polarity.max}
                    step={c.polarity.step}
                    disabled={lockedPolarity.positive}
                    className={lockedPolarity.positive ? 'opacity-50' : ''}
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-yellow-500">Neutral</Label>
                    <button
                      type="button"
                      onClick={() => toggleLock('neutral')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedPolarity.neutral ? 'text-yellow-500' : 'text-muted-foreground'}`}
                      title={lockedPolarity.neutral ? 'Unlock' : 'Lock'}
                    >
                      {lockedPolarity.neutral ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[config.attributes_profile.polarity.neutral * 100]}
                    onValueChange={([v]) => updateLinkedPolarity('neutral', v / 100)}
                    min={c.polarity.min}
                    max={c.polarity.max}
                    step={c.polarity.step}
                    disabled={lockedPolarity.neutral}
                    className={lockedPolarity.neutral ? 'opacity-50' : ''}
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-red-500">Negative</Label>
                    <button
                      type="button"
                      onClick={() => toggleLock('negative')}
                      className={`p-1 rounded hover:bg-muted transition-colors ${lockedPolarity.negative ? 'text-red-500' : 'text-muted-foreground'}`}
                      title={lockedPolarity.negative ? 'Unlock' : 'Lock'}
                    >
                      {lockedPolarity.negative ? <Lock className="h-3 w-3" /> : <Unlock className="h-3 w-3" />}
                    </button>
                  </div>
                  <Slider
                    value={[config.attributes_profile.polarity.negative * 100]}
                    onValueChange={([v]) => updateLinkedPolarity('negative', v / 100)}
                    min={c.polarity.min}
                    max={c.polarity.max}
                    step={c.polarity.step}
                    disabled={lockedPolarity.negative}
                    className={lockedPolarity.negative ? 'opacity-50' : ''}
                  />
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Click the lock icon to freeze a value. Unlocked sliders automatically redistribute to maintain 100% total.
              </p>
            </div>
          </AblationSection>
        </div>

        {/* Capitalization Style Weights */}
        <div className="space-y-3 mt-4">
          <div className="flex items-center gap-2">
            <Label>Capitalization Style Distribution</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-sm">
                  <p className="font-medium mb-1">Controls capitalization variation</p>
                  <p>Each review is assigned a capitalization style via weighted random sampling. LLMs default to proper capitalization — this forces realistic variation. Weights are relative (don't need to sum to 1.0).</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {([
              { key: 'standard', label: 'Standard', desc: 'Proper capitalization' },
              { key: 'lowercase', label: 'Lowercase', desc: 'Mostly lowercase, "i" instead of "I"' },
              { key: 'mixed', label: 'Mixed', desc: 'Inconsistent capitalization' },
              { key: 'emphasis', label: 'ALL CAPS emphasis', desc: 'Normal + occasional ALL CAPS' },
            ] as const).map(({ key, label, desc }) => (
              <div key={key} className="flex items-center gap-2">
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium">{label}</span>
                  <p className="text-[10px] text-muted-foreground truncate">{desc}</p>
                </div>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={config.attributes_profile.cap_weights?.[key] ?? 0}
                  onChange={(e) => updateConfig('attributes_profile.cap_weights', {
                    ...config.attributes_profile.cap_weights,
                    [key]: Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)),
                  })}
                  className="w-20 text-center"
                />
              </div>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground">
            Total: {((config.attributes_profile.cap_weights?.standard ?? 0.55) + (config.attributes_profile.cap_weights?.lowercase ?? 0.20) + (config.attributes_profile.cap_weights?.mixed ?? 0.15) + (config.attributes_profile.cap_weights?.emphasis ?? 0.10)).toFixed(2)}
          </p>
        </div>

        <Separator />

        {/* Sentence Range - NEW */}
        <div className="space-y-4">
          <div>
            <div className="flex items-center">
              <Label className="text-base">Review Length</Label>
              {isLengthFromRef && <RefContextIndicator />}
            </div>
            <p className="text-xs text-muted-foreground">
              Number of sentences per review
            </p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                Range: {config.attributes_profile.length_range[0]} - {config.attributes_profile.length_range[1]} sentences
              </span>
            </div>
            <Slider
              value={config.attributes_profile.length_range}
              onValueChange={(v) => updateConfig('attributes_profile.length_range', v)}
              min={c.sentence_range.min}
              max={c.sentence_range.max}
              step={c.sentence_range.step}
            />
            <p className="text-xs text-muted-foreground">
              Each review will contain between {config.attributes_profile.length_range[0]} and {config.attributes_profile.length_range[1]} sentences
            </p>
          </div>

          {/* Edge Lengths */}
          <div className="mt-4 space-y-3">
            <div>
              <Label className="text-sm font-medium">Edge Lengths</Label>
              <p className="text-xs text-muted-foreground">
                Per-review chance to generate outlier-length reviews for natural variation
              </p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Min. edge length & chance</Label>
                <div className="flex items-center gap-2">
                  <Input
                    type="number"
                    min={1}
                    max={config.attributes_profile.length_range[0] - 1 || 1}
                    value={config.attributes_profile.edge_lengths?.min_length ?? 1}
                    onChange={(e) => updateConfig('attributes_profile.edge_lengths', {
                      ...config.attributes_profile.edge_lengths,
                      min_length: Math.max(1, parseInt(e.target.value) || 1),
                    })}
                    className="w-16 h-8 text-center"
                  />
                  <span className="text-xs text-muted-foreground">sent</span>
                  <Input
                    type="number"
                    min={0}
                    max={50}
                    value={Math.round((config.attributes_profile.edge_lengths?.min_chance ?? 0.15) * 100)}
                    onChange={(e) => updateConfig('attributes_profile.edge_lengths', {
                      ...config.attributes_profile.edge_lengths,
                      min_chance: Math.max(0, Math.min(50, parseInt(e.target.value) || 0)) / 100,
                    })}
                    className="w-16 h-8 text-center"
                  />
                  <span className="text-xs text-muted-foreground">%</span>
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground">Max. edge length & chance</Label>
                <div className="flex items-center gap-2">
                  <Input
                    type="number"
                    min={config.attributes_profile.length_range[1] + 1 || 6}
                    value={config.attributes_profile.edge_lengths?.max_length ?? 15}
                    onChange={(e) => updateConfig('attributes_profile.edge_lengths', {
                      ...config.attributes_profile.edge_lengths,
                      max_length: Math.max(config.attributes_profile.length_range[1] + 1, parseInt(e.target.value) || 6),
                    })}
                    className="w-16 h-8 text-center"
                  />
                  <span className="text-xs text-muted-foreground">sent</span>
                  <Input
                    type="number"
                    min={0}
                    max={50}
                    value={Math.round((config.attributes_profile.edge_lengths?.max_chance ?? 0.05) * 100)}
                    onChange={(e) => updateConfig('attributes_profile.edge_lengths', {
                      ...config.attributes_profile.edge_lengths,
                      max_chance: Math.max(0, Math.min(50, parseInt(e.target.value) || 0)) / 100,
                    })}
                    className="w-16 h-8 text-center"
                  />
                  <span className="text-xs text-muted-foreground">%</span>
                </div>
              </div>
            </div>
            {((config.attributes_profile.edge_lengths?.min_chance ?? 0) + (config.attributes_profile.edge_lengths?.max_chance ?? 0)) > 0.40 && (
              <p className="text-xs text-amber-500">
                Combined edge chance is {Math.round(((config.attributes_profile.edge_lengths?.min_chance ?? 0) + (config.attributes_profile.edge_lengths?.max_chance ?? 0)) * 100)}% — most reviews will be outlier lengths
              </p>
            )}
          </div>
        </div>

        <Separator />

        {/* Temperature Range */}
        <div className="space-y-4">
          <div>
            <Label className="text-base">Temperature</Label>
            <p className="text-xs text-muted-foreground">
              LLM generation temperature (higher = more creative/varied)
            </p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                Range: {config.attributes_profile.temperature_range[0].toFixed(1)} - {config.attributes_profile.temperature_range[1].toFixed(1)}
              </span>
            </div>
            <Slider
              value={config.attributes_profile.temperature_range}
              onValueChange={(v) => updateConfig('attributes_profile.temperature_range', v)}
              min={c.temperature_range.min}
              max={c.temperature_range.max}
              step={c.temperature_range.step}
            />
            <p className="text-xs text-muted-foreground">
              Each review uses a randomly sampled temperature between {config.attributes_profile.temperature_range[0].toFixed(1)} and {config.attributes_profile.temperature_range[1].toFixed(1)}
            </p>
          </div>
        </div>

        <Separator />

        {/* Noise Settings - controlled by ablation */}
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center">
                <Label className="text-base">Noise Level</Label>
                {config.attributes_profile.noise.preset === 'ref_dataset' && extractedRefContext?.noise && <RefContextIndicator />}
              </div>
              <p className="text-xs text-muted-foreground">
                Add realistic imperfections to generated text
              </p>
            </div>
            <Switch
              checked={noiseEnabled}
              onCheckedChange={(v) => updateConfig('ablation.noise_enabled', v)}
            />
          </div>
          <AblationSection enabled={noiseEnabled} effect="Clean, error-free text only">
            <div className="space-y-6">
              {/* Noise Preset */}
              <div className={`grid gap-3 ${extractedRefContext?.noise ? 'sm:grid-cols-4' : 'sm:grid-cols-3'}`}>
                {/* Reference Dataset preset - only shown when extracted noise data available */}
                {extractedRefContext?.noise && (
                  <button
                    onClick={() => {
                      updateConfig('attributes_profile.noise.preset', 'ref_dataset')
                      updateConfig('attributes_profile.noise.typo_rate', extractedRefContext.noise?.typo_rate || 0.01)
                      updateConfig('attributes_profile.noise.colloquialism', extractedRefContext.noise?.has_colloquialisms ?? true)
                    }}
                    className={`p-3 rounded-lg border text-left transition-colors ${
                      config.attributes_profile.noise.preset === 'ref_dataset'
                        ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500'
                        : 'border-border hover:border-blue-500/50'
                    }`}
                  >
                    <div className="font-medium text-sm flex items-center gap-1.5">
                      <Database className="h-3.5 w-3.5 text-blue-500" />
                      Reference
                    </div>
                    <div className="text-xs text-muted-foreground">
                      ~{((extractedRefContext.noise?.typo_rate || 0) * 100).toFixed(1)}% typos (detected)
                    </div>
                  </button>
                )}
                {NOISE_PRESETS.map((preset) => (
                  <button
                    key={preset.value}
                    onClick={() => updateConfig('attributes_profile.noise.preset', preset.value)}
                    className={`p-3 rounded-lg border text-left transition-colors ${
                      config.attributes_profile.noise.preset === preset.value
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <div className="font-medium text-sm">{preset.label}</div>
                    <div className="text-xs text-muted-foreground">{preset.description}</div>
                  </button>
                ))}
              </div>

              {/* Advanced Noise Settings */}
              <div className="space-y-4">
                <Label>Advanced Noise Settings</Label>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Colloquialisms</p>
                    <p className="text-xs text-muted-foreground">Add informal language</p>
                  </div>
                  <Switch
                    checked={config.attributes_profile.noise.colloquialism}
                    onCheckedChange={(v) => updateConfig('attributes_profile.noise.colloquialism', v)}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">Grammar Variations</p>
                    <p className="text-xs text-muted-foreground">Add minor grammar imperfections</p>
                  </div>
                  <Switch
                    checked={config.attributes_profile.noise.grammar_errors}
                    onCheckedChange={(v) => updateConfig('attributes_profile.noise.grammar_errors', v)}
                  />
                </div>
              </div>
            </div>
          </AblationSection>
        </div>
    </div>
  )
}

// Format a prompt template with config values
function formatPromptTemplate(template: string, config: any): string {
  const subject = config.subject_profile.query || '[Subject]'
  const region = config.subject_profile.region || 'united states'
  const domain = config.subject_profile.domain || 'general'
  const ageEnabled = config.ablation.age_enabled
  const sexEnabled = config.ablation.sex_enabled
  const polarityEnabled = config.ablation.polarity_enabled

  // Sample reviewer based on config
  const sampleAge = ageEnabled
    ? String(Math.floor((config.reviewer_profile.age_range[0] + config.reviewer_profile.age_range[1]) / 2))
    : 'unspecified'
  const sampleSex = sexEnabled
    ? ((config.reviewer_profile.sex_distribution.male ?? 0.5) > (config.reviewer_profile.sex_distribution.female ?? 0.5) ? 'male' : 'female')
    : 'unspecified'
  const samplePolarity = polarityEnabled ? 'positive' : 'natural'

  // Polarity distribution values
  const polarityDist = config.attributes_profile.polarity_distribution || { positive: 0.4, neutral: 0.3, negative: 0.3 }
  const polarityPositive = Math.round(polarityDist.positive * 100)
  const polarityNeutral = Math.round(polarityDist.neutral * 100)
  const polarityNegative = Math.round(polarityDist.negative * 100)

  // Aspect categories - show current categories or placeholder
  const aspectCategories = config.subject_profile.aspect_categories?.length > 0
    ? config.subject_profile.aspect_categories.join(', ')
    : '[No categories configured - using default ABSA categories]'

  // Dataset mode instruction and output example
  const datasetMode = config.generation?.dataset_mode || 'semeval'
  const datasetModeInstruction = datasetMode === 'semeval'
    ? 'Use SemEval format with aspect categories and opinion target expressions (OTE).'
    : 'Use ACSA format with aspect categories only (no OTE extraction).'

  const outputExample = datasetMode === 'semeval'
    ? `{
  "text": "The food was excellent but service was slow.",
  "sentences": [
    {
      "text": "The food was excellent but service was slow.",
      "opinions": [
        { "category": "FOOD#QUALITY", "polarity": "positive", "target": "food" },
        { "category": "SERVICE#GENERAL", "polarity": "negative", "target": "service" }
      ]
    }
  ]
}`
    : `{
  "text": "The food was excellent but service was slow.",
  "sentences": [
    {
      "text": "The food was excellent but service was slow.",
      "opinions": [
        { "category": "FOOD#QUALITY", "polarity": "positive" },
        { "category": "SERVICE#GENERAL", "polarity": "negative" }
      ]
    }
  ]
}`

  // Replace all placeholders
  return template
    .replace(/{subject}/g, subject)
    .replace(/{domain}/g, domain)
    .replace(/{region}/g, region)
    .replace(/{min_sentences}/g, String(config.attributes_profile.length_range[0]))
    .replace(/{max_sentences}/g, String(config.attributes_profile.length_range[1]))
    .replace(/{age}/g, sampleAge + (ageEnabled ? '' : ' (ablation: disabled)'))
    .replace(/{sex}/g, sampleSex + (sexEnabled ? '' : ' (ablation: disabled)'))
    .replace(/{polarity}/g, samplePolarity + (polarityEnabled ? '' : ' (ablation: let LLM decide)'))
    .replace(/{additional_context}/g, config.reviewer_profile.additional_context || 'No additional context')
    .replace(/{subject_context}/g, '[Will be populated from SIL web search results]')
    .replace(/{features}/g, '[Features extracted from web search]')
    .replace(/{pros}/g, '[Positive aspects from web search]')
    .replace(/{cons}/g, '[Negative aspects from web search]')
    .replace(/{polarity_positive}/g, String(polarityPositive))
    .replace(/{polarity_neutral}/g, String(polarityNeutral))
    .replace(/{polarity_negative}/g, String(polarityNegative))
    .replace(/{aspect_categories}/g, aspectCategories)
    .replace(/{dataset_mode_instruction}/g, datasetModeInstruction)
    .replace(/{output_example}/g, outputExample)
}

// Preview AML Dialog Component - fetches prompts from Python API
function PreviewAMLDialog({ config }: { config: any }) {
  const [open, setOpen] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState<string | null>(null)
  const [userPrompt, setUserPrompt] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch prompts when dialog opens
  useEffect(() => {
    if (!open) return

    const fetchPrompts = async () => {
      setLoading(true)
      setError(null)

      const pythonApiUrl = PYTHON_API_URL

      try {
        const [systemRes, userRes] = await Promise.all([
          fetch(`${pythonApiUrl}/api/prompts/aml/system`),
          fetch(`${pythonApiUrl}/api/prompts/aml/user`),
        ])

        if (!systemRes.ok || !userRes.ok) {
          throw new Error('Failed to fetch prompts from API')
        }

        const systemData = await systemRes.json()
        const userData = await userRes.json()

        setSystemPrompt(formatPromptTemplate(systemData.content, config))
        setUserPrompt(formatPromptTemplate(userData.content, config))
      } catch (err) {
        setError('Could not fetch prompts from Python API. Make sure the CLI server is running (python -m cera serve).')
      } finally {
        setLoading(false)
      }
    }

    fetchPrompts()
  }, [open, config])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Eye className="h-4 w-4 mr-2" />
          Preview AML
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-auto">
        <DialogHeader>
          <DialogTitle>AML Prompt Preview</DialogTitle>
          <DialogDescription>
            This is what an AML (Authenticity Modeling Layer) prompt looks like based on your current configuration
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 mt-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading prompts...</span>
            </div>
          ) : error ? (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Cannot Load Prompts</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : (
            <>
              <div className="rounded-lg border p-4">
                <h4 className="font-medium mb-2 text-sm">System Prompt</h4>
                <pre className="text-xs bg-muted p-3 rounded overflow-auto whitespace-pre-wrap font-mono">
                  {systemPrompt}
                </pre>
              </div>
              <div className="rounded-lg border p-4">
                <h4 className="font-medium mb-2 text-sm">User Prompt (Example)</h4>
                <pre className="text-xs bg-muted p-3 rounded overflow-auto whitespace-pre-wrap font-mono">
                  {userPrompt}
                </pre>
              </div>
            </>
          )}
          <p className="text-xs text-muted-foreground">
            Note: The actual prompts will include subject intelligence data (features, pros, cons) gathered during the SIL phase.
            Each review will have a unique reviewer profile based on your configured distributions.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function GenerationStepContent({
  config,
  updateConfig,
  constraints,
  modelValidations,
}: {
  config: any
  updateConfig: (path: string, value: any) => void
  constraints: UIConstraints
  modelValidations: Record<string, ModelValidationState>
}) {
  const { providers, groupedModels, processedModels, loading: modelsLoading } = useOpenRouterModels()
  const selectedModelInfo = processedModels.find(m => m.id === config.generation.model)
  const c = constraints.constraints

  // Ensure targets array exists (backward compat)
  const targets: CeraTarget[] = config.generation.targets?.length > 0
    ? config.generation.targets
    : [DEFAULT_CERA_TARGET]

  const handleTargetChange = (index: number, updated: CeraTarget) => {
    const newTargets = [...targets]
    newTargets[index] = updated
    updateConfig('generation.targets', newTargets)
  }

  const handleTargetRemove = (index: number) => {
    const newTargets = targets.filter((_: CeraTarget, i: number) => i !== index)
    updateConfig('generation.targets', newTargets)
  }

  const handleAddTarget = () => {
    const lastTarget = targets[targets.length - 1]
    updateConfig('generation.targets', [...targets, {
      ...lastTarget,
      target_value: lastTarget.target_value + 400,
    }])
  }

  // Summary for collapsible header
  const targetsSummary = targets
    .map((t: CeraTarget) => `${t.target_value.toLocaleString()} ${t.count_mode === 'sentences' ? 'sents' : 'reviews'}`)
    .join(', ')

  return (
    <div className="space-y-6">
        {/* Target Prefix */}
        <div className="space-y-2">
          <div className="flex items-center gap-1">
            <Label>Target Prefix</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">Prefix for generated dataset file names. Example: "rq1-cera" produces "rq1-cera-100-explicit.xml"</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <Input
            type="text"
            placeholder={config.name ? config.name.toLowerCase().replace(/\s+/g, '-') : 'e.g., rq1-cera'}
            value={config.generation.target_prefix || ''}
            onChange={(e) => updateConfig('generation.target_prefix', e.target.value)}
            className="max-w-xs"
          />
        </div>

        {/* Parallelize Target Datasets toggle (only when 2+ targets) */}
        {targets.length > 1 && (
          <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-3">
            <div className="flex items-center gap-2">
              <Label className="text-sm">Parallelize Target Datasets</Label>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs text-xs">Run all target dataset sizes concurrently. Each target gets its own generation pipeline.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <Switch
              checked={config.generation.parallel_targets || false}
              onCheckedChange={(checked) => updateConfig('generation.parallel_targets', checked)}
            />
          </div>
        )}

        {/* Target Dataset Rows */}
        <div className="space-y-2">
          {targets.map((target: CeraTarget, idx: number) => (
            <CeraTargetRow
              key={idx}
              index={idx}
              target={target}
              onChange={handleTargetChange}
              onRemove={handleTargetRemove}
              canRemove={targets.length > 1}
              lengthRange={config.attributes_profile.length_range as [number, number]}
            />
          ))}

          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleAddTarget}
          >
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add target dataset
          </Button>
        </div>

        <Separator />

        {/* LLM Selection — Generation Model(s) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Generation Model{config.generation.models?.length > 1 ? 's' : ''}</Label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Multi-Model</span>
              <Switch
                checked={config.generation.models?.length > 0}
                onCheckedChange={(checked) => {
                  if (checked) {
                    // Toggle ON: current model becomes first entry in models array
                    const currentModel = config.generation.model
                    if (currentModel) {
                      updateConfig('generation.models', [currentModel])
                    } else {
                      updateConfig('generation.models', [''])
                    }
                  } else {
                    // Toggle OFF: keep first model as the single model, clear rest
                    const first = config.generation.models?.[0] || config.generation.model || ''
                    updateConfig('generation.models', [])
                    updateConfig('generation.model', first)
                    if (first) {
                      updateConfig('generation.provider', first.split('/')[0])
                    }
                    updateConfig('generation.parallel_models', false)
                  }
                }}
              />
            </div>
          </div>

          {config.generation.models?.length > 0 ? (
            /* Multi-model mode: stacked LLMSelectors */
            <div className="space-y-3">
              {config.generation.models.map((modelId: string, idx: number) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-xs text-muted-foreground font-mono mt-3 w-5 shrink-0">{idx + 1}.</span>
                  <div className="flex-1">
                    <LLMSelector
                      providers={providers}
                      groupedModels={groupedModels}
                      loading={modelsLoading}
                      value={modelId}
                      disabledModels={config.generation.models.filter((_: string, i: number) => i !== idx).filter(Boolean)}
                      onChange={(newModelId) => {
                        const updated = [...config.generation.models]
                        updated[idx] = newModelId
                        updateConfig('generation.models', updated)
                        // Keep primary model = first in list
                        if (idx === 0) {
                          updateConfig('generation.model', newModelId)
                          updateConfig('generation.provider', newModelId.split('/')[0])
                        }
                      }}
                      placeholder="Select model..."
                      validationStatus={modelValidations[`generation-${idx}`]?.status}
                      validationError={modelValidations[`generation-${idx}`]?.error}
                    />
                  </div>
                  {/* Remove button (only for 2nd+ models) */}
                  {idx > 0 && (
                    <button
                      type="button"
                      className="text-muted-foreground hover:text-destructive mt-3 shrink-0"
                      onClick={() => {
                        const updated = config.generation.models.filter((_: string, i: number) => i !== idx)
                        updateConfig('generation.models', updated)
                        // If we're back to 1 model, keep it in sync
                        if (updated.length > 0) {
                          updateConfig('generation.model', updated[0])
                          updateConfig('generation.provider', updated[0].split('/')[0])
                        }
                      }}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              ))}

              {/* Add Model button */}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  updateConfig('generation.models', [...config.generation.models, ''])
                }}
              >
                <Plus className="h-3.5 w-3.5 mr-1.5" />
                Add Model
              </Button>

              {/* Parallel execution toggle (only when 2+ models selected) */}
              {config.generation.models.filter(Boolean).length > 1 && (
                <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label className="text-sm">Parallel Model Execution</Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p>Run all models concurrently. Each model gets its own request_size concurrent calls.</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      checked={config.generation.parallel_models || false}
                      onCheckedChange={(checked) => updateConfig('generation.parallel_models', checked)}
                    />
                  </div>
                  {config.generation.parallel_models && (
                    <p className="text-xs text-amber-600 dark:text-amber-400">
                      {config.generation.models.filter(Boolean).length} models x {config.generation.request_size} request_size = {config.generation.models.filter(Boolean).length * config.generation.request_size} concurrent API calls
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : (
            /* Single-model mode: standard LLMSelector */
            <LLMSelector
              providers={providers}
              groupedModels={groupedModels}
              loading={modelsLoading}
              value={config.generation.model || ''}
              onChange={(modelId) => {
                updateConfig('generation.model', modelId)
                updateConfig('generation.provider', modelId.split('/')[0])
              }}
              placeholder="Select model..."
              validationStatus={modelValidations['generation']?.status}
              validationError={modelValidations['generation']?.error}
            />
          )}
        </div>

        {/* Free model privacy notice */}
        {(config.generation.models?.length > 0
          ? config.generation.models.some((m: string) => m.includes(':free'))
          : config.generation.model?.includes(':free')
        ) && (
          <div className="flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3">
            <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-xs">
              <span className="font-medium">Free model selected.</span> Requires{' '}
              <a
                href="https://openrouter.ai/settings/privacy"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-amber-600"
              >
                OpenRouter privacy settings
              </a>{' '}
              to allow free endpoints. Your prompts may be used for training.
            </p>
          </div>
        )}
    </div>
  )
}
