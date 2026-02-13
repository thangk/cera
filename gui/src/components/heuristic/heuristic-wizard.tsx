import { useState, useCallback } from 'react'
import { useMutation, useAction, useQuery } from 'convex/react'
import { api } from 'convex/_generated/api'
import { useNavigate } from '@tanstack/react-router'
import { toast } from 'sonner'
import { PYTHON_API_URL } from '../../lib/api-urls'
import {
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Info,
  Play,
  Loader2,
  FileText,
  Sparkles,
  Plus,
  X,
  HelpCircle,
} from 'lucide-react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Textarea } from '../ui/textarea'
import { Checkbox } from '../ui/checkbox'
import { Badge } from '../ui/badge'
import { Separator } from '../ui/separator'
import { Switch } from '../ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { FileDropZone } from '../ui/file-drop-zone'
import { LLMSelector, type ValidationStatus } from '../llm-selector'
import { useOpenRouterModels, type OpenRouterModel } from '../../hooks/use-openrouter-models'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'
import { DollarSign } from 'lucide-react'
import { HeuristicTargetRow, DEFAULT_HEURISTIC_TARGET, type HeuristicTarget } from '../target-dataset-row'

// Default prompt template with placeholders
const DEFAULT_PROMPT_TEMPLATE = `Generate {review_count} product reviews for a restaurant. Each review should have approximately {avg_sentences} sentences.

Include a mix of positive, negative, and neutral sentiments. Be realistic and varied in your writing style. Write as if you were actual customers with different backgrounds and experiences.

Focus on aspects from these categories: FOOD#QUALITY, FOOD#PRICES, SERVICE#GENERAL, AMBIENCE#GENERAL, RESTAURANT#GENERAL, RESTAURANT#PRICES, DRINKS#QUALITY, DRINKS#PRICES, LOCATION#GENERAL.`

// Default format prompt for ABSA JSON output
const DEFAULT_FORMAT_PROMPT = `Return your response as a JSON array with ABSA (Aspect-Based Sentiment Analysis) format.
Each review should have sentences, and each sentence should have aspect-level opinions.

Structure:
[
  {
    "sentences": [
      {
        "text": "The food was delicious and fresh.",
        "opinions": [
          {"target": "food", "category": "FOOD#QUALITY", "polarity": "positive", "from": 4, "to": 8}
        ]
      },
      {
        "text": "However, the service was quite slow.",
        "opinions": [
          {"target": "service", "category": "SERVICE#GENERAL", "polarity": "negative", "from": 13, "to": 20}
        ]
      }
    ]
  }
]

Rules:
- "text": The sentence text (non-empty string)
- "target": The aspect term as it appears in the text (exact substring)
- "category": Use categories mentioned in the prompt above
- "polarity": One of "positive", "negative", "neutral"
- "from": Character offset where target starts in the text (0-indexed)
- "to": Character offset where target ends (exclusive)

Important:
- The "from" and "to" must be accurate character positions for the target substring
- A sentence can have multiple opinions about different aspects
- A sentence can have zero opinions if it's neutral/general
Return ONLY the JSON array, no other text, no markdown code blocks.`

// MDQA Metrics configuration
const ALL_METRICS = [
  'bertscore', 'bleu', 'rouge_l', 'moverscore',
  'distinct_1', 'distinct_2', 'self_bleu'
]

const REFERENCE_METRICS = ['bertscore', 'bleu', 'rouge_l', 'moverscore']
const DIVERSITY_METRICS = ['distinct_1', 'distinct_2', 'self_bleu']

// Heuristic config type
export interface HeuristicConfig {
  name: string
  prompt: string
  useFormatPrompt: boolean
  formatPrompt: string
  avgSentencesPerReview: string
  outputFormat: 'semeval_xml' | 'jsonl' | 'csv'
  knowledgeSourceJobId: string | null // Job ID to import SIL knowledge from
  // Multi-target dataset support
  targetPrefix: string // File naming prefix (e.g., "rq1-heuristic")
  targets: HeuristicTarget[]
  parallelTargets: boolean
  // Multi-model support
  models: string[]
  parallelModels: boolean
  // Legacy fields (backward compat, derived from targets[0] and models[0])
  model: string
  targetMode: 'reviews' | 'sentences'
  targetValue: number
  reviewsPerBatch: number
  requestSize: number // Number of parallel batch requests (default: 3)
  totalRuns: number // Number of times to run generation (default: 1)
  parallelRuns: boolean // Run all runs concurrently (default: false)
  // Evaluation settings
  referenceFile: string | null
  referenceFileName: string | null
  metrics: string[]
}

// Parse avg sentences string (e.g. "4-7" → 5.5, "5" → 5) into a numeric midpoint for calculations
function parseAvgSentences(val: string): number {
  const rangeMatch = val.match(/^(\d+)\s*-\s*(\d+)$/)
  if (rangeMatch) return (parseInt(rangeMatch[1]) + parseInt(rangeMatch[2])) / 2
  const num = parseFloat(val)
  return isNaN(num) || num <= 0 ? 5 : num
}

const DEFAULT_HEURISTIC_CONFIG: HeuristicConfig = {
  name: '',
  prompt: DEFAULT_PROMPT_TEMPLATE,
  useFormatPrompt: true,
  formatPrompt: DEFAULT_FORMAT_PROMPT,
  avgSentencesPerReview: '5',
  outputFormat: 'semeval_xml',
  knowledgeSourceJobId: null,
  // Multi-target
  targetPrefix: '',
  targets: [{ ...DEFAULT_HEURISTIC_TARGET }],
  parallelTargets: true,
  // Multi-model
  models: [''],
  parallelModels: true,
  // Legacy (backward compat)
  model: '',
  targetMode: 'sentences',
  targetValue: 100,
  reviewsPerBatch: 1,
  requestSize: 25,
  totalRuns: 1,
  parallelRuns: true,
  // Evaluation
  referenceFile: null,
  referenceFileName: null,
  metrics: [...DIVERSITY_METRICS],
}

// Tab definitions
const HEURISTIC_TABS = [
  { id: 'input', label: 'Input', color: '#6b7280' }, // Grey
  { id: 'prompt', label: 'Heuristic Prompt', color: '#8b5cf6' }, // Purple
  { id: 'settings', label: 'Settings', color: '#8b5cf6' }, // Purple
  { id: 'output', label: 'Output', color: '#6b7280' }, // Grey
]

interface HeuristicWizardProps {
  onBack: () => void
  onReset: () => void
}

export function HeuristicWizard({ onBack, onReset }: HeuristicWizardProps) {
  const navigate = useNavigate()
  const createJob = useMutation(api.jobs.create)
  const runPipeline = useAction(api.pipelineAction.runPipeline)
  const settings = useQuery(api.settings.get)
  const ceraJobs = useQuery(api.jobs.listCompletedCeraJobs)
  const { providers, groupedModels, models: rawModels, loading: modelsLoading } = useOpenRouterModels()

  // State (must be declared before any hooks that reference it)
  const [activeTab, setActiveTab] = useState(0)
  const [config, setConfig] = useState<HeuristicConfig>(DEFAULT_HEURISTIC_CONFIG)
  const [isSubmitting, setIsSubmitting] = useState(false)
  // Model validation state (idle by default - LLMSelector validates internally)
  const [modelValidation] = useState<{
    status: ValidationStatus
    error?: string
  }>({ status: 'idle' })

  // Update config helper
  const updateConfig = useCallback((key: keyof HeuristicConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }, [])

  // Calculate derived values
  const avgSentencesNum = parseAvgSentences(config.avgSentencesPerReview)
  // Use first target for cost/batch estimates (legacy compat)
  const firstTarget = config.targets?.[0] || DEFAULT_HEURISTIC_TARGET
  const totalReviews = firstTarget.targetMode === 'reviews'
    ? firstTarget.targetValue
    : Math.ceil(firstTarget.targetValue / avgSentencesNum)

  const totalBatches = Math.ceil(totalReviews / firstTarget.reviewsPerBatch)

  // Token estimates for heuristic generation
  const TOKEN_ESTIMATES = {
    heuristic_per_batch: { input: 800, output: 2000 }, // Larger output for batch of reviews
  }

  // Calculate cost estimates (must be after config and totalBatches are defined)
  const calculateCostEstimates = useCallback(() => {
    const defaultPricing = { input: 2.0, output: 8.0 } // Default pricing per 1M tokens

    const getModelPricing = (modelId: string): { input: number; output: number } => {
      if (!rawModels || !modelId) return defaultPricing
      const model = rawModels.find((m: OpenRouterModel) => m.id === modelId)
      if (!model) return defaultPricing
      const inputPrice = parseFloat(model.pricing?.prompt || '0') * 1_000_000
      const outputPrice = parseFloat(model.pricing?.completion || '0') * 1_000_000
      return { input: inputPrice, output: outputPrice }
    }

    const pricing = getModelPricing(config.model)

    // Calculate tokens per batch (heuristic generates multiple reviews at once)
    const inputTokensPerBatch = TOKEN_ESTIMATES.heuristic_per_batch.input +
      (config.prompt.length / 4) + // Rough token estimate from prompt
      (config.useFormatPrompt ? config.formatPrompt.length / 4 : 0)

    const outputTokensPerBatch = TOKEN_ESTIMATES.heuristic_per_batch.output * (config.reviewsPerBatch / 10) // Scale with batch size

    // Total tokens across all batches and runs
    const totalInputTokens = inputTokensPerBatch * totalBatches * config.totalRuns
    const totalOutputTokens = outputTokensPerBatch * totalBatches * config.totalRuns
    const totalTokens = totalInputTokens + totalOutputTokens

    // Calculate cost
    const cost = (totalInputTokens * pricing.input + totalOutputTokens * pricing.output) / 1_000_000
    const calls = totalBatches * config.totalRuns

    return {
      cost,
      calls,
      tokens: totalTokens,
      promptTokens: totalInputTokens,
      completionTokens: totalOutputTokens,
    }
  }, [config, rawModels, totalBatches])

  const costEstimates = calculateCostEstimates()

  // Handle reference file upload
  const handleReferenceFileSelect = useCallback((file: File | null) => {
    if (file) {
      // Read file and store content
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        setConfig(prev => ({
          ...prev,
          referenceFile: content,
          referenceFileName: file.name,
          // Enable reference metrics when file is uploaded
          metrics: [...ALL_METRICS],
        }))
      }
      reader.readAsText(file)
    } else {
      setConfig(prev => ({
        ...prev,
        referenceFile: null,
        referenceFileName: null,
        // Disable reference metrics when file is removed
        metrics: [...DIVERSITY_METRICS],
      }))
    }
  }, [])

  // Toggle metric
  const toggleMetric = useCallback((metric: string) => {
    setConfig(prev => {
      const isEnabled = prev.metrics.includes(metric)
      if (isEnabled) {
        return { ...prev, metrics: prev.metrics.filter(m => m !== metric) }
      } else {
        return { ...prev, metrics: [...prev.metrics, metric] }
      }
    })
  }, [])

  // Validation
  const getValidationErrors = useCallback(() => {
    const errors: string[] = []
    if (!config.name.trim()) errors.push('Job name is required')
    if (!config.prompt.trim()) errors.push('Prompt template is required')
    if (!config.model) errors.push('LLM model is required')
    if (config.targetValue <= 0) errors.push('Target value must be positive')
    if (config.reviewsPerBatch <= 0) errors.push('Reviews per batch must be positive')
    if (!config.avgSentencesPerReview.trim()) {
      errors.push('Average sentences is required')
    } else if (!/^\d+(\s*-\s*\d+)?$/.test(config.avgSentencesPerReview.trim())) {
      errors.push('Average sentences must be a number (e.g., "5") or range (e.g., "4-7")')
    } else if (parseAvgSentences(config.avgSentencesPerReview) <= 0) {
      errors.push('Average sentences must be positive')
    }
    if (modelValidation.status === 'error') errors.push('Selected model is not valid')
    return errors
  }, [config, modelValidation])

  const isValid = getValidationErrors().length === 0

  // Submit handlers
  const handleCreateJob = useCallback(async (runAfterCreate: boolean) => {
    const errors = getValidationErrors()
    if (errors.length > 0) {
      toast.error('Validation Error', {
        description: errors[0],
      })
      return
    }

    setIsSubmitting(true)

    try {
      // Create the job
      const jobId = await createJob({
        name: config.name,
        config: {}, // Empty config for heuristic jobs (CERA config not used)
        phases: ['generation', 'evaluation'], // Heuristic skips composition
        method: 'heuristic',
        heuristicConfig: (() => {
          const firstTarget = config.targets?.[0]
          const effectiveModels = config.models?.filter(Boolean) || []
          return {
            prompt: config.prompt,
            useFormatPrompt: config.useFormatPrompt,
            formatPrompt: config.useFormatPrompt ? config.formatPrompt : undefined,
            avgSentencesPerReview: config.avgSentencesPerReview,
            outputFormat: config.outputFormat,
            knowledgeSourceJobId: config.knowledgeSourceJobId || undefined,
            // Multi-target fields
            targetPrefix: config.targetPrefix || undefined,
            targets: config.targets,
            parallelTargets: config.parallelTargets,
            // Multi-model fields
            models: effectiveModels.length > 0 ? effectiveModels : undefined,
            parallelModels: config.parallelModels,
            // Legacy fields from targets[0] and models[0] for backward compat
            targetMode: firstTarget?.targetMode || config.targetMode,
            targetValue: firstTarget?.targetValue || config.targetValue,
            reviewsPerBatch: firstTarget?.reviewsPerBatch || config.reviewsPerBatch,
            requestSize: firstTarget?.requestSize || config.requestSize,
            totalRuns: firstTarget?.totalRuns || config.totalRuns,
            parallelRuns: firstTarget?.runsMode === 'parallel',
            model: effectiveModels[0] || config.model,
          }
        })(),
        evaluationConfig: {
          metrics: config.metrics,
          reference_metrics_enabled: !!config.referenceFile,
          reference_file: config.referenceFileName || undefined,
        },
      })

      // Upload reference dataset file if provided (for Lexical/Semantic metrics)
      if (config.referenceFile && config.referenceFileName) {
        try {
          const pythonApiUrl = PYTHON_API_URL
          // Convert text content back to a File for upload
          const blob = new Blob([config.referenceFile], { type: 'text/plain' })
          const file = new File([blob], config.referenceFileName)

          const formData = new FormData()
          formData.append('file', file)
          formData.append('jobId', jobId)
          formData.append('jobName', config.name)
          formData.append('fileType', 'reference')

          await fetch(`${pythonApiUrl}/api/upload-dataset`, {
            method: 'POST',
            body: formData,
          })
        } catch (err) {
          console.warn('Reference dataset upload failed:', err)
        }
      }

      toast.success('Job created successfully')

      if (runAfterCreate) {
        // Start the pipeline
        try {
          await runPipeline({ jobId })
          toast.success('Pipeline started')
        } catch (error) {
          console.error('Failed to start pipeline:', error)
          toast.error('Failed to start pipeline', {
            description: error instanceof Error ? error.message : 'Unknown error',
          })
        }
      }

      // Navigate to job detail
      navigate({ to: '/jobs/$jobId', params: { jobId } })

    } catch (error) {
      console.error('Failed to create job:', error)
      toast.error('Failed to create job', {
        description: error instanceof Error ? error.message : 'Unknown error',
      })
    } finally {
      setIsSubmitting(false)
    }
  }, [config, createJob, runPipeline, navigate, getValidationErrors])

  // Tab navigation
  const canGoNext = activeTab < HEURISTIC_TABS.length - 1
  const canGoPrev = activeTab > 0

  const goNext = () => {
    if (canGoNext) setActiveTab(prev => prev + 1)
  }

  const goPrev = () => {
    if (canGoPrev) setActiveTab(prev => prev - 1)
  }

  // Render tab content
  const renderTabContent = () => {
    switch (HEURISTIC_TABS[activeTab].id) {
      case 'input':
        return (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="job-name">Job Name</Label>
              <Input
                id="job-name"
                placeholder="e.g., heuristic-restaurant-reviews-500"
                value={config.name}
                onChange={(e) => updateConfig('name', e.target.value)}
              />
              <p className="text-sm text-muted-foreground">
                A descriptive name to identify this heuristic generation job
              </p>
            </div>
          </div>
        )

      case 'prompt':
        return (
          <div className="space-y-6">
            {/* Knowledge Source */}
            <div className="space-y-2">
              <Label>Knowledge Source</Label>
              <p className="text-xs text-muted-foreground">
                Import verified facts from a completed CERA job to constrain the heuristic to the same domain knowledge
              </p>
              <Select
                value={config.knowledgeSourceJobId || 'none'}
                onValueChange={(val) => updateConfig('knowledgeSourceJobId', val === 'none' ? null : val)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="None (unconstrained)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (unconstrained)</SelectItem>
                  {ceraJobs?.map((job) => (
                    <SelectItem key={job._id} value={job._id}>
                      {job.name} <span className="text-muted-foreground ml-1">({job.method})</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Separator />

            {/* Prompt Template */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="prompt-template">Prompt Template</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => updateConfig('prompt', DEFAULT_PROMPT_TEMPLATE)}
                  className="text-muted-foreground"
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  Reset
                </Button>
              </div>
              <Textarea
                id="prompt-template"
                value={config.prompt}
                onChange={(e) => updateConfig('prompt', e.target.value)}
                rows={12}
                className="font-mono text-sm"
                placeholder="Write your prompt here..."
              />
            </div>

            {/* Available Variables */}
            <div className="rounded-lg border bg-muted/30 p-4">
              <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                <Info className="h-4 w-4" />
                Available Variables (auto-injected from Settings)
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li><code className="text-purple-600">{'{review_count}'}</code> — Reviews per batch (from Settings)</li>
                <li><code className="text-purple-600">{'{avg_sentences}'}</code> — Avg sentences per review</li>
                <li><code className="text-purple-600">{'{sentence_count}'}</code> — Total sentences per batch (review_count × avg_sentences)</li>
              </ul>
            </div>

            {/* Helpful Tips */}
            <div className="rounded-lg border bg-purple-50 dark:bg-purple-950/20 p-4">
              <h4 className="text-sm font-medium mb-2 text-purple-700 dark:text-purple-300">
                Helpful Tips
              </h4>
              <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>Use <code>{'{review_count}'}</code> instead of hardcoding numbers</li>
                <li>Describe the domain/subject for the reviews</li>
                <li>Mention polarity distribution if needed (positive/negative/neutral)</li>
                <li>
                  <span className="font-medium">Include aspect categories</span> you want reviews to cover, e.g.:
                  <div className="mt-1 flex flex-wrap gap-1">
                    {['FOOD#QUALITY', 'FOOD#PRICES', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL', 'RESTAURANT#GENERAL', 'DRINKS#QUALITY'].map(cat => (
                      <code key={cat} className="text-xs bg-purple-100 dark:bg-purple-900/30 px-1 py-0.5 rounded">{cat}</code>
                    ))}
                  </div>
                </li>
              </ul>
            </div>

            <Separator />

            {/* Format Prompt */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="format-prompt-toggle" className="cursor-pointer">ABSA Format Prompt</Label>
                  <p className="text-xs text-muted-foreground">
                    Append JSON format instructions for parseable ABSA output
                  </p>
                </div>
                <Switch
                  id="format-prompt-toggle"
                  checked={config.useFormatPrompt}
                  onCheckedChange={(checked) => updateConfig('useFormatPrompt', checked)}
                />
              </div>
              {config.useFormatPrompt && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="format-prompt" className="text-sm text-muted-foreground">Format Instructions (appended to prompt)</Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => updateConfig('formatPrompt', DEFAULT_FORMAT_PROMPT)}
                      className="text-muted-foreground h-7"
                    >
                      <RotateCcw className="h-3 w-3 mr-1" />
                      Reset
                    </Button>
                  </div>
                  <Textarea
                    id="format-prompt"
                    value={config.formatPrompt}
                    onChange={(e) => updateConfig('formatPrompt', e.target.value)}
                    rows={10}
                    className="font-mono text-xs bg-muted/30"
                  />
                </div>
              )}
            </div>
          </div>
        )

      case 'settings':
        const hTargets: HeuristicTarget[] = config.targets?.length > 0
          ? config.targets
          : [DEFAULT_HEURISTIC_TARGET]

        const handleHTargetChange = (index: number, updated: HeuristicTarget) => {
          const newTargets = [...hTargets]
          newTargets[index] = updated
          updateConfig('targets', newTargets)
        }

        const handleHTargetRemove = (index: number) => {
          updateConfig('targets', hTargets.filter((_: HeuristicTarget, i: number) => i !== index))
        }

        const handleHAddTarget = () => {
          const lastTarget = hTargets[hTargets.length - 1]
          updateConfig('targets', [...hTargets, {
            ...lastTarget,
            targetValue: lastTarget.targetValue + 400,
          }])
        }

        const hModels: string[] = config.models?.length > 0
          ? config.models
          : [config.model || '']

        const isMultiModel = hModels.length > 0 && (hModels.length > 1 || config.models?.length > 0)

        return (
          <div className="space-y-6">
            {/* Generation Settings */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Generation Settings</h3>

              {/* Avg Sentences per Review (global) */}
              <div className="space-y-2">
                <Label htmlFor="avg-sentences">Avg Sentences per Review</Label>
                <Input
                  id="avg-sentences"
                  type="text"
                  placeholder="e.g., 5 or 4-7"
                  value={config.avgSentencesPerReview}
                  onChange={(e) => updateConfig('avgSentencesPerReview', e.target.value)}
                  className="max-w-[120px]"
                />
              </div>

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
                        <p className="max-w-xs text-xs">Prefix for generated dataset file names. Example: "rq1-heuristic" produces "rq1-heuristic-100-explicit.xml"</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Input
                  type="text"
                  placeholder={config.name ? config.name.toLowerCase().replace(/\s+/g, '-') : 'e.g., rq1-heuristic'}
                  value={config.targetPrefix || ''}
                  onChange={(e) => updateConfig('targetPrefix', e.target.value)}
                  className="max-w-xs"
                />
              </div>

              {/* Parallelize Target Datasets toggle (only when 2+ targets) */}
              {hTargets.length > 1 && (
                <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-3">
                  <div className="flex items-center gap-2">
                    <Label className="text-sm">Parallelize Target Datasets</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="max-w-xs text-xs">Run all target dataset sizes concurrently.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Switch
                    checked={config.parallelTargets || false}
                    onCheckedChange={(checked) => updateConfig('parallelTargets', checked)}
                  />
                </div>
              )}

              {/* Target Dataset Rows */}
              <div className="space-y-2">
                {hTargets.map((target: HeuristicTarget, idx: number) => (
                  <HeuristicTargetRow
                    key={idx}
                    index={idx}
                    target={target}
                    onChange={handleHTargetChange}
                    onRemove={handleHTargetRemove}
                    canRemove={hTargets.length > 1}
                    avgSentencesPerReview={config.avgSentencesPerReview}
                  />
                ))}

                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleHAddTarget}
                >
                  <Plus className="h-3.5 w-3.5 mr-1.5" />
                  Add target dataset
                </Button>
              </div>
            </div>

            <Separator />

            {/* Generation Model(s) — moved from Prompt tab */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Generation Model{hModels.length > 1 ? 's' : ''}</Label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">Multi-Model</span>
                  <Switch
                    checked={isMultiModel}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        const currentModel = config.model || hModels[0] || ''
                        updateConfig('models', currentModel ? [currentModel] : [''])
                      } else {
                        const first = hModels[0] || ''
                        updateConfig('models', [])
                        updateConfig('model', first)
                        updateConfig('parallelModels', true)
                      }
                    }}
                  />
                </div>
              </div>

              {isMultiModel ? (
                /* Multi-model mode */
                <div className="space-y-3">
                  {hModels.map((modelId: string, idx: number) => (
                    <div key={idx} className="flex items-start gap-2">
                      <span className="text-xs text-muted-foreground font-mono mt-3 w-5 shrink-0">{idx + 1}.</span>
                      <div className="flex-1">
                        <LLMSelector
                          providers={providers}
                          groupedModels={groupedModels}
                          loading={modelsLoading}
                          value={modelId}
                          disabledModels={hModels.filter((_: string, i: number) => i !== idx).filter(Boolean)}
                          onChange={(newModelId: string) => {
                            const updated = [...hModels]
                            updated[idx] = newModelId
                            updateConfig('models', updated)
                            if (idx === 0) {
                              updateConfig('model', newModelId)
                            }
                          }}
                          placeholder="Select model..."
                        />
                      </div>
                      {idx > 0 && (
                        <button
                          type="button"
                          className="text-muted-foreground hover:text-destructive mt-3 shrink-0"
                          onClick={() => {
                            const updated = hModels.filter((_: string, i: number) => i !== idx)
                            updateConfig('models', updated)
                            if (updated.length > 0) updateConfig('model', updated[0])
                          }}
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  ))}

                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => updateConfig('models', [...hModels, ''])}
                  >
                    <Plus className="h-3.5 w-3.5 mr-1.5" />
                    Add Model
                  </Button>

                  {/* Parallel Model Execution toggle */}
                  {hModels.filter(Boolean).length > 1 && (
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
                          checked={config.parallelModels ?? true}
                          onCheckedChange={(checked) => updateConfig('parallelModels', checked)}
                        />
                      </div>
                      {(config.parallelModels ?? true) && (
                        <p className="text-xs text-amber-600 dark:text-amber-400">
                          {hModels.filter(Boolean).length} models x {hTargets[0]?.requestSize || 3} request_size = {hModels.filter(Boolean).length * (hTargets[0]?.requestSize || 3)} concurrent API calls
                        </p>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                /* Single-model mode */
                <LLMSelector
                  providers={providers}
                  groupedModels={groupedModels}
                  loading={modelsLoading}
                  value={config.model || ''}
                  onChange={(modelId: string) => updateConfig('model', modelId)}
                  placeholder="Select model..."
                  validationStatus={modelValidation.status}
                  validationError={modelValidation.error}
                />
              )}
            </div>

            <Separator />

            {/* Reference Dataset */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Reference Dataset (Optional)</h3>
              <p className="text-sm text-muted-foreground">
                Upload a reference dataset to enable lexical and semantic evaluation metrics
              </p>

              <FileDropZone
                accept=".jsonl,.csv,.xml"
                onFileSelect={handleReferenceFileSelect}
                selectedFileName={config.referenceFileName}
                placeholder="Drop reference dataset here or click to upload"
              />
            </div>

            <Separator />

            {/* MDQA Metrics */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">MDQA Metrics</h3>

              {/* Lexical & Semantic (require reference) */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">
                  Lexical & Semantic {!config.referenceFile && '(requires reference)'}
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {REFERENCE_METRICS.map(metric => (
                    <label
                      key={metric}
                      className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
                        !config.referenceFile
                          ? 'opacity-50 cursor-not-allowed'
                          : config.metrics.includes(metric)
                            ? 'bg-primary/10 border-primary'
                            : 'hover:bg-muted'
                      }`}
                    >
                      <Checkbox
                        checked={config.metrics.includes(metric)}
                        onCheckedChange={() => toggleMetric(metric)}
                        disabled={!config.referenceFile}
                      />
                      <span className="text-sm uppercase">{metric.replace('_', '-')}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Diversity (always available) */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-muted-foreground">
                  Diversity (always available)
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {DIVERSITY_METRICS.map(metric => (
                    <label
                      key={metric}
                      className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
                        config.metrics.includes(metric)
                          ? 'bg-primary/10 border-primary'
                          : 'hover:bg-muted'
                      }`}
                    >
                      <Checkbox
                        checked={config.metrics.includes(metric)}
                        onCheckedChange={() => toggleMetric(metric)}
                      />
                      <span className="text-sm uppercase">{metric.replace('_', '-')}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )

      case 'output':
        return (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Job Summary</h3>

            {/* Summary Grid */}
            <div className="grid gap-4 sm:grid-cols-2">
              {/* Job Info */}
              <div className="rounded-lg border p-4 space-y-2">
                <h4 className="font-medium flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Job Info
                </h4>
                <div className="text-sm space-y-1">
                  <p><span className="text-muted-foreground">Name:</span> {config.name || '(not set)'}</p>
                  <p><span className="text-muted-foreground">Method:</span> Heuristic Prompting</p>
                </div>
              </div>

              {/* Model */}
              <div className="rounded-lg border p-4 space-y-2">
                <h4 className="font-medium flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  LLM Model
                </h4>
                <p className="text-sm font-mono truncate">
                  {config.model || '(not selected)'}
                </p>
              </div>

              {/* Generation Settings */}
              <div className="rounded-lg border p-4 space-y-2">
                <h4 className="font-medium">Generation Settings</h4>
                <div className="text-sm space-y-1">
                  <p><span className="text-muted-foreground">Target:</span> {config.targetValue.toLocaleString()} {config.targetMode}</p>
                  <p><span className="text-muted-foreground">Reviews per batch:</span> {config.reviewsPerBatch}</p>
                  <p><span className="text-muted-foreground">Request size:</span> {config.requestSize} parallel</p>
                  <p><span className="text-muted-foreground">Avg sentences:</span> {config.avgSentencesPerReview}</p>
                  <p><span className="text-muted-foreground">Batches:</span> {totalBatches}</p>
                  {config.totalRuns > 1 && (
                    <p>
                      <span className="text-muted-foreground">Total Runs:</span>{' '}
                      <span className="text-amber-600 dark:text-amber-400 font-medium">{config.totalRuns} runs</span>
                    </p>
                  )}
                </div>
              </div>

              {/* Metrics */}
              <div className="rounded-lg border p-4 space-y-2">
                <h4 className="font-medium">MDQA Metrics</h4>
                <div className="flex flex-wrap gap-1">
                  {config.metrics.map(metric => (
                    <Badge key={metric} variant="secondary" className="text-xs">
                      {metric.replace('_', '-').toUpperCase()}
                    </Badge>
                  ))}
                </div>
                {config.referenceFileName && (
                  <p className="text-xs text-muted-foreground">
                    Reference: {config.referenceFileName}
                  </p>
                )}
              </div>
            </div>

            {/* Cost Estimation */}
            <div className="rounded-lg border p-4 space-y-3">
              <div className="flex items-center gap-2">
                <DollarSign className="h-4 w-4 text-muted-foreground" />
                <h4 className="font-medium">Estimated Cost</h4>
                <TooltipProvider delayDuration={200}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-4 w-4 text-muted-foreground/60 hover:text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-sm text-xs">
                      <p>Estimates based on prompt length and model pricing from OpenRouter. Actual costs may vary.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">LLM Calls</p>
                  <p className="font-medium">{costEstimates.calls.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Est. Tokens</p>
                  <p className="font-medium">{costEstimates.tokens.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Est. Cost</p>
                  <p className="font-medium text-green-600 dark:text-green-400">
                    ${costEstimates.cost.toFixed(4)}
                  </p>
                </div>
              </div>
              {config.totalRuns > 1 && (
                <p className="text-xs text-muted-foreground">
                  * Includes {config.totalRuns} runs × {totalBatches} batches = {costEstimates.calls} total API calls
                </p>
              )}
            </div>

            {/* Prompt Preview */}
            <div className="space-y-2">
              <h4 className="font-medium">Prompt Preview</h4>
              <div className="rounded-lg border bg-muted/30 p-4 max-h-40 overflow-y-auto">
                <pre className="text-sm whitespace-pre-wrap font-mono">
                  {config.prompt.substring(0, 500)}{config.prompt.length > 500 ? '...' : ''}
                </pre>
              </div>
            </div>

            {/* Output Format */}
            <div className="space-y-2">
              <Label>Output Format</Label>
              <Select
                value={config.outputFormat}
                onValueChange={(v) => updateConfig('outputFormat', v as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="semeval_xml">SemEval XML</SelectItem>
                  <SelectItem value="jsonl">JSONL</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                </SelectContent>
              </Select>
              <div className="rounded-lg border bg-muted/30 p-3 mt-2">
                <p className="text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">Note:</span> Dataset will be produced in both{' '}
                  <span className="font-medium">explicit</span> (with aspect targets and offsets) and{' '}
                  <span className="font-medium">implicit</span> (targets set to NULL) forms for ABSA research comparison.
                </p>
              </div>
            </div>

            {/* Validation Errors */}
            {getValidationErrors().length > 0 && (
              <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
                <h4 className="font-medium text-destructive mb-2">Please fix the following:</h4>
                <ul className="list-disc list-inside text-sm text-destructive space-y-1">
                  {getValidationErrors().map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4 pt-4">
              <Button
                variant="outline"
                onClick={() => handleCreateJob(false)}
                disabled={!isValid || isSubmitting}
              >
                {isSubmitting ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
                Create Job
              </Button>
              <Button
                onClick={() => handleCreateJob(true)}
                disabled={!isValid || isSubmitting || !settings?.openrouterApiKey}
              >
                {isSubmitting ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Play className="h-4 w-4 mr-2" />}
                Create & Run Pipeline
              </Button>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Create a job</h1>
          <p className="text-muted-foreground">
            Heuristic Prompting — RQ1 Baseline
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="text-muted-foreground"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Methods
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="text-muted-foreground"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Reset
          </Button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex items-center gap-1">
        {HEURISTIC_TABS.map((tab, index) => {
          const isActive = index === activeTab
          const isPast = index < activeTab

          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(index)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'text-white shadow-sm'
                  : isPast
                    ? 'text-muted-foreground bg-muted hover:bg-muted/80'
                    : 'text-muted-foreground hover:bg-muted'
              }`}
              style={{
                backgroundColor: isActive ? tab.color : undefined,
              }}
            >
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                isActive ? 'bg-white/20' : 'bg-muted-foreground/20'
              }`}>
                {index + 1}
              </span>
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {renderTabContent()}
      </div>

      {/* Navigation Buttons */}
      {activeTab < HEURISTIC_TABS.length - 1 && (
        <div className="flex justify-between pt-4 border-t">
          <Button
            variant="outline"
            onClick={goPrev}
            disabled={!canGoPrev}
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Previous
          </Button>
          <Button onClick={goNext} disabled={!canGoNext}>
            Next
            <ChevronRight className="h-4 w-4 ml-1" />
          </Button>
        </div>
      )}
    </div>
  )
}
