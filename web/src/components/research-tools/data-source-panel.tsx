import { useState, useCallback } from 'react'
import { useQuery } from 'convex/react'
import { api } from '../../../convex/_generated/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { FileDropZone } from '@/components/ui/file-drop-zone'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Plus, X, Database, FileUp, Loader2 } from 'lucide-react'
import type { DataEntry, Method, JobSource, FileSource, MetricStat, MdqaMetricKey, PerRunMetrics, LadyMetricKey } from './types'
import { MDQA_METRIC_KEYS } from './types'
import { PYTHON_API_URL } from '@/lib/api-urls'

interface DataSourcePanelProps {
  entries: DataEntry[]
  onEntriesChange: (entries: DataEntry[]) => void
  /** Which metric types this table needs */
  metricType: 'mdqa' | 'lady' | 'both'
  /** Expected file format description */
  fileHint?: string
}

export function DataSourcePanel({ entries, onEntriesChange, metricType, fileHint }: DataSourcePanelProps) {
  const [showAddForm, setShowAddForm] = useState(false)

  const addEntry = useCallback((entry: DataEntry) => {
    onEntriesChange([...entries, entry])
    setShowAddForm(false)
  }, [entries, onEntriesChange])

  const removeEntry = useCallback((id: string) => {
    onEntriesChange(entries.filter(e => e.id !== id))
  }, [entries, onEntriesChange])

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Data Sources</CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAddForm(true)}
            disabled={showAddForm}
          >
            <Plus className="h-4 w-4 mr-1.5" />
            Add Data Source
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Existing entries */}
        {entries.length > 0 && (
          <div className="space-y-2">
            {entries.map(entry => (
              <EntryRow key={entry.id} entry={entry} onRemove={removeEntry} />
            ))}
          </div>
        )}

        {entries.length === 0 && !showAddForm && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No data sources added. Click "Add Data Source" to begin.
          </p>
        )}

        {/* Add form */}
        {showAddForm && (
          <AddEntryForm
            onAdd={addEntry}
            onCancel={() => setShowAddForm(false)}
            metricType={metricType}
            fileHint={fileHint}
          />
        )}
      </CardContent>
    </Card>
  )
}

function EntryRow({ entry, onRemove }: { entry: DataEntry; onRemove: (id: string) => void }) {
  const methodColors: Record<Method, string> = {
    real: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    cera: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    heuristic: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  }

  const sourceName = entry.source.type === 'job'
    ? entry.source.jobName
    : entry.source.fileName

  const hasMetrics = entry.source.type === 'job'
    ? entry.source.mdqaMetrics !== null
    : (entry.source.mdqaMetrics !== null || entry.source.ladyMetrics !== null)

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-md border bg-muted/30">
      <Badge variant="outline" className={methodColors[entry.method]}>
        {entry.method.toUpperCase()}
      </Badge>
      <span className="text-sm font-mono">n={entry.size}</span>
      <span className="text-sm text-muted-foreground flex-1 truncate">
        {entry.source.type === 'job' ? <Database className="h-3 w-3 inline mr-1" /> : <FileUp className="h-3 w-3 inline mr-1" />}
        {sourceName}
      </span>
      {hasMetrics && <Badge variant="secondary" className="text-xs">Loaded</Badge>}
      {!hasMetrics && <Badge variant="destructive" className="text-xs">No metrics</Badge>}
      <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => onRemove(entry.id)}>
        <X className="h-3 w-3" />
      </Button>
    </div>
  )
}

function AddEntryForm({
  onAdd,
  onCancel,
  metricType,
  fileHint,
}: {
  onAdd: (entry: DataEntry) => void
  onCancel: () => void
  metricType: 'mdqa' | 'lady' | 'both'
  fileHint?: string
}) {
  const [method, setMethod] = useState<Method>('cera')
  const [size, setSize] = useState<number>(1000)
  const [sourceType, setSourceType] = useState<'job' | 'file'>('job')

  // Job selection state
  const completedJobs = useQuery(api.jobs.listForResearch)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [jobSource, setJobSource] = useState<JobSource | null>(null)
  const [loadingMetrics, setLoadingMetrics] = useState(false)

  // Multi-model state: when a job has perModelMetrics with >1 entries
  const [availableModels, setAvailableModels] = useState<Array<{ model: string; modelSlug: string; metrics: Record<string, number>; perRunMetrics?: Array<{ run: number; datasetFile: string; metrics: Record<string, number> }> }>>([])
  const [selectedModelSlug, setSelectedModelSlug] = useState<string | null>(null)

  // File drop state
  const [fileSource, setFileSource] = useState<FileSource | null>(null)

  // Helper to extract MDQA metrics from a flat metrics object
  const extractMetricsFromFlat = useCallback((metricsObj: Record<string, number>): Partial<Record<MdqaMetricKey, MetricStat>> => {
    const result: Partial<Record<MdqaMetricKey, MetricStat>> = {}
    for (const key of MDQA_METRIC_KEYS) {
      if (typeof metricsObj[key] === 'number') {
        result[key] = { mean: metricsObj[key] }
      }
    }
    return result
  }, [])

  // Handle model selection from multi-model picker
  const handleModelSelect = useCallback((modelSlug: string) => {
    setSelectedModelSlug(modelSlug)
    const modelData = availableModels.find(m => m.modelSlug === modelSlug)
    if (!modelData) return

    const job = completedJobs?.find((j: { _id: string }) => j._id === selectedJobId)
    if (!job) return

    const mdqaMetrics = extractMetricsFromFlat(modelData.metrics)
    let perRunMetrics: PerRunMetrics[] | null = null
    if (modelData.perRunMetrics && modelData.perRunMetrics.length > 0) {
      perRunMetrics = modelData.perRunMetrics.map(r => ({
        run: r.run,
        metrics: r.metrics as Partial<Record<MdqaMetricKey, number>>,
      }))
    }

    setJobSource({
      type: 'job',
      jobId: job._id,
      jobName: `${job.name} [${modelSlug}]`,
      mdqaMetrics: Object.keys(mdqaMetrics).length > 0 ? mdqaMetrics : null,
      perRunMetrics,
    })
  }, [availableModels, completedJobs, selectedJobId, extractMetricsFromFlat])

  const handleJobSelect = useCallback(async (jobId: string) => {
    setSelectedJobId(jobId)
    setSelectedModelSlug(null)
    setAvailableModels([])
    const job = completedJobs?.find((j: { _id: string }) => j._id === jobId)
    if (!job) return

    // Check for multi-model job
    const pmm = (job as any).perModelMetrics as Array<{
      model: string; modelSlug: string;
      metrics: Record<string, number>;
      perRunMetrics?: Array<{ run: number; datasetFile: string; metrics: Record<string, number> }>
    }> | undefined
    if (pmm && pmm.length > 1) {
      setAvailableModels(pmm)
      // Don't auto-set jobSource — wait for model selection
      // Auto-detect method/size
      if (job.method === 'cera' || job.method === 'heuristic') {
        setMethod(job.method as Method)
      }
      if (job.generationCount > 0) {
        setSize(job.generationCount)
      }
      return
    }

    // Auto-detect method and size from jobDir folder name (e.g. "j973...-rq1-cera-1000")
    const jobDirName = (job as { jobDir?: string }).jobDir
    if (jobDirName) {
      const parts = jobDirName.replace(/\/$/, '').split('/').pop()?.split('-') || []
      if (parts.length >= 2) {
        const sizeStr = parts[parts.length - 1]
        const methodStr = parts[parts.length - 2]
        const parsedSize = parseInt(sizeStr)
        if (!isNaN(parsedSize) && parsedSize > 0) {
          setSize(parsedSize)
        }
        if (methodStr === 'cera' || methodStr === 'heuristic' || methodStr === 'real') {
          setMethod(methodStr as Method)
        }
      }
    } else {
      // Fallback: auto-detect method from job field
      if (job.method === 'cera' || job.method === 'heuristic') {
        setMethod(job.method as Method)
      }
      // Fallback: auto-detect size from generation count
      if (job.generationCount > 0) {
        setSize(job.generationCount)
      }
    }

    // Extract metrics from Convex data
    const mdqaMetrics: Partial<Record<MdqaMetricKey, MetricStat>> = {}

    // Try averageMetrics first (multi-run format with mean/std)
    if (job.averageMetrics) {
      for (const key of MDQA_METRIC_KEYS) {
        const val = job.averageMetrics[key]
        if (val && typeof val === 'object' && 'mean' in val) {
          mdqaMetrics[key] = { mean: val.mean, std: val.std ?? undefined }
        }
      }
    }

    // Fallback to evaluationMetrics (flat format with key/key_std)
    if (Object.keys(mdqaMetrics).length === 0) {
      const evalMetrics = (job as { evaluationMetrics?: Record<string, number | undefined> }).evaluationMetrics
      if (evalMetrics) {
        for (const key of MDQA_METRIC_KEYS) {
          const val = evalMetrics[key]
          if (typeof val === 'number') {
            const stdKey = `${key}_std`
            mdqaMetrics[key] = {
              mean: val,
              std: typeof evalMetrics[stdKey] === 'number' ? evalMetrics[stdKey] as number : undefined,
            }
          }
        }
      }
    }

    // Extract per-run metrics from Convex
    let perRunMetrics: PerRunMetrics[] | null = null
    if (job.perRunMetrics && Array.isArray(job.perRunMetrics)) {
      perRunMetrics = job.perRunMetrics.map((run: { run: number; metrics: Record<string, number | undefined> }) => ({
        run: run.run,
        metrics: run.metrics as Partial<Record<MdqaMetricKey, number>>,
      }))
    }

    // If no metrics found in Convex, try reading from filesystem via Python API
    if (Object.keys(mdqaMetrics).length === 0 && jobDirName) {
      setLoadingMetrics(true)
      try {
        const res = await fetch(`${PYTHON_API_URL}/api/read-metrics`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ jobDir: jobDirName }),
        })
        if (res.ok) {
          const apiData = await res.json()
          const metrics = apiData.metrics
          if (metrics) {
            // Parse flat metrics from API response categories
            for (const category of ['lexical', 'semantic', 'diversity']) {
              const catMetrics = metrics[category]
              if (Array.isArray(catMetrics)) {
                for (const item of catMetrics) {
                  const key = item.metric as MdqaMetricKey
                  if (MDQA_METRIC_KEYS.includes(key)) {
                    mdqaMetrics[key] = { mean: item.score }
                  }
                }
              }
            }
            // Extract std from API response
            if (metrics.std) {
              for (const category of ['lexical', 'semantic', 'diversity']) {
                const catStd = metrics.std[category]
                if (Array.isArray(catStd)) {
                  for (const item of catStd) {
                    const key = item.metric as MdqaMetricKey
                    if (mdqaMetrics[key]) {
                      mdqaMetrics[key] = { ...mdqaMetrics[key]!, std: item.score }
                    }
                  }
                }
              }
            }
            // Extract per-run metrics from API response
            if (metrics.perRun && !perRunMetrics) {
              const runMap = new Map<number, Partial<Record<MdqaMetricKey, number>>>()
              for (const category of ['lexical', 'semantic', 'diversity']) {
                const catRuns = metrics.perRun[category]
                if (catRuns && typeof catRuns === 'object') {
                  for (const [runKey, runMetrics] of Object.entries(catRuns)) {
                    const runNum = parseInt(runKey.replace('run', ''))
                    if (!runMap.has(runNum)) runMap.set(runNum, {})
                    const existing = runMap.get(runNum)!
                    for (const item of runMetrics as { metric: string; score: number }[]) {
                      const key = item.metric as MdqaMetricKey
                      if (MDQA_METRIC_KEYS.includes(key)) {
                        existing[key] = item.score
                      }
                    }
                  }
                }
              }
              if (runMap.size > 0) {
                perRunMetrics = Array.from(runMap.entries())
                  .sort(([a], [b]) => a - b)
                  .map(([run, metrics]) => ({ run, metrics }))
              }
            }
          }
        }
      } catch {
        // API not available, metrics remain empty
      } finally {
        setLoadingMetrics(false)
      }
    }

    setJobSource({
      type: 'job',
      jobId: job._id,
      jobName: job.name,
      mdqaMetrics: Object.keys(mdqaMetrics).length > 0 ? mdqaMetrics : null,
      perRunMetrics,
    })
  }, [completedJobs, extractMetricsFromFlat])

  const handleFileDrop = useCallback((file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      const content = reader.result as string
      const parsed = parseMetricsFile(content, metricType)
      setFileSource({
        type: 'file',
        fileName: file.name,
        mdqaMetrics: parsed.mdqa,
        perRunMetrics: parsed.perRunMetrics,
        ladyMetrics: parsed.lady,
      })
    }
    reader.readAsText(file)
  }, [metricType])

  const handleAdd = () => {
    const source = sourceType === 'job' ? jobSource : fileSource
    if (!source) return

    onAdd({
      id: crypto.randomUUID(),
      method,
      size,
      source,
    })
  }

  const canAdd = sourceType === 'job'
    ? jobSource !== null && (availableModels.length <= 1 || selectedModelSlug !== null)
    : fileSource !== null

  return (
    <div className="border rounded-lg p-4 space-y-4 bg-background">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Add Data Source</span>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Method + Size row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs">Method</Label>
          <Select value={method} onValueChange={(v) => setMethod(v as Method)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="real">Real</SelectItem>
              <SelectItem value="cera">CERA</SelectItem>
              <SelectItem value="heuristic">Heuristic</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">Dataset Size</Label>
          <Input
            type="number"
            value={size}
            onChange={(e) => setSize(parseInt(e.target.value) || 0)}
            min={1}
          />
        </div>
      </div>

      {/* Source tabs */}
      <Tabs value={sourceType} onValueChange={(v) => setSourceType(v as 'job' | 'file')}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="job" className="flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5" />
            Select Job
          </TabsTrigger>
          <TabsTrigger value="file" className="flex items-center gap-1.5">
            <FileUp className="h-3.5 w-3.5" />
            Drop File
          </TabsTrigger>
        </TabsList>

        <TabsContent value="job" className="mt-3">
          <Select
            value={selectedJobId || ''}
            onValueChange={handleJobSelect}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a completed job..." />
            </SelectTrigger>
            <SelectContent className="max-h-60">
              {completedJobs?.map((job: { _id: string; name: string; method: string; generationCount: number; totalRuns: number }) => (
                <SelectItem key={job._id} value={job._id}>
                  <span className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px] px-1">
                      {(job.method || 'cera').toUpperCase()}
                    </Badge>
                    {job.name}
                    <span className="text-muted-foreground ml-1">
                      (n={job.generationCount}, {job.totalRuns} run{job.totalRuns > 1 ? 's' : ''})
                    </span>
                  </span>
                </SelectItem>
              ))}
              {(!completedJobs || completedJobs.length === 0) && (
                <div className="px-2 py-4 text-sm text-muted-foreground text-center">
                  No completed jobs with metrics found
                </div>
              )}
            </SelectContent>
          </Select>
          {/* Multi-model picker */}
          {availableModels.length > 1 && (
            <div className="mt-2">
              <Label className="text-xs text-muted-foreground">Select model (multi-model job)</Label>
              <Select
                value={selectedModelSlug || ''}
                onValueChange={handleModelSelect}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select a model..." />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map(m => (
                    <SelectItem key={m.modelSlug} value={m.modelSlug}>
                      {m.modelSlug}
                      {m.perRunMetrics && m.perRunMetrics.length > 1 && (
                        <span className="text-muted-foreground ml-1">({m.perRunMetrics.length} runs)</span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          {loadingMetrics && (
            <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1.5">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading metrics from job directory...
            </p>
          )}
          {jobSource && !loadingMetrics && (
            <p className="text-xs text-muted-foreground mt-2">
              {jobSource.mdqaMetrics ? 'MDQA metrics loaded' : 'No MDQA metrics'}
              {jobSource.perRunMetrics ? ` | ${jobSource.perRunMetrics.length} runs` : ''}
            </p>
          )}
          {availableModels.length > 1 && !selectedModelSlug && (
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
              Please select a model to load its metrics.
            </p>
          )}
        </TabsContent>

        <TabsContent value="file" className="mt-3">
          <FileDropZone
            onFile={handleFileDrop}
            accept=".json,.jsonl,.csv"
            formatLabel="JSON"
            placeholder={fileHint || 'Drop mdqa-results.json or mdqa_metrics_average.json'}
            description={metricType === 'lady'
              ? 'Expected: LADy-kap evaluation results (JSON)'
              : 'Expected: mdqa-results.json (CERA) or mdqa_metrics_average.json (Heuristic)'
            }
          />
          {fileSource && (
            <p className="text-xs text-muted-foreground mt-2">
              {fileSource.mdqaMetrics ? 'MDQA metrics parsed' : ''}
              {fileSource.ladyMetrics ? 'LADy-kap metrics parsed' : ''}
              {fileSource.perRunMetrics ? ` | ${fileSource.perRunMetrics.length} runs` : ''}
              {!fileSource.mdqaMetrics && !fileSource.ladyMetrics ? 'No metrics parsed from file' : ''}
            </p>
          )}
        </TabsContent>
      </Tabs>

      {/* Action buttons */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
        <Button size="sm" onClick={handleAdd} disabled={!canAdd}>Add</Button>
      </div>
    </div>
  )
}

/**
 * Parse a metrics file (mdqa-results.json or LADy-kap output).
 */
function parseMetricsFile(content: string, metricType: 'mdqa' | 'lady' | 'both'): {
  mdqa: Partial<Record<MdqaMetricKey, MetricStat>> | null
  perRunMetrics: PerRunMetrics[] | null
  lady: Partial<Record<LadyMetricKey, MetricStat>> | null
} {
  try {
    const data = JSON.parse(content)
    const result: {
      mdqa: Partial<Record<MdqaMetricKey, MetricStat>> | null
      perRunMetrics: PerRunMetrics[] | null
      lady: Partial<Record<LadyMetricKey, MetricStat>> | null
    } = { mdqa: null, perRunMetrics: null, lady: null }

    // Try parsing as MDQA results
    if (metricType === 'mdqa' || metricType === 'both') {
      const mdqa: Partial<Record<MdqaMetricKey, MetricStat>> = {}

      // Multi-run format: { runs: [...], average: {...}, std: {...} }
      if (data.runs && Array.isArray(data.runs)) {
        // Extract per-run metrics
        result.perRunMetrics = data.runs.map((run: Record<string, unknown>) => {
          const metrics: Partial<Record<MdqaMetricKey, number>> = {}
          // Metrics may be nested in categories or flat
          const flat = (run._flat || run) as Record<string, unknown>
          for (const key of MDQA_METRIC_KEYS) {
            if (typeof flat[key] === 'number') {
              metrics[key] = flat[key] as number
            }
          }
          // Also check nested categories
          for (const category of ['lexical', 'semantic', 'diversity']) {
            const cat = run[category]
            if (cat && typeof cat === 'object') {
              for (const key of MDQA_METRIC_KEYS) {
                if (typeof (cat as Record<string, unknown>)[key] === 'number') {
                  metrics[key] = (cat as Record<string, unknown>)[key] as number
                }
              }
            }
          }
          return { run: (run.run as number) || 0, metrics }
        })

        // Extract averages
        const avg = data.average?._flat || data.average
        const std = data.std?._flat || data.std
        if (avg) {
          for (const key of MDQA_METRIC_KEYS) {
            if (typeof avg[key] === 'number') {
              mdqa[key] = {
                mean: avg[key],
                std: typeof std?.[key] === 'number' ? std[key] : undefined,
              }
            }
          }
        }
      }
      // Single-run format: { lexical: {...}, semantic: {...}, diversity: {...}, _flat: {...} }
      else if (data._flat || data.lexical || data.semantic || data.diversity) {
        const flat = data._flat || {}
        // Merge from categories
        for (const category of ['lexical', 'semantic', 'diversity']) {
          if (data[category] && typeof data[category] === 'object') {
            Object.assign(flat, data[category])
          }
        }
        for (const key of MDQA_METRIC_KEYS) {
          if (typeof flat[key] === 'number') {
            mdqa[key] = { mean: flat[key] }
          }
        }
      }
      // Heuristic flat format: { key: value, key_std: value } (e.g. mdqa_metrics_average.json)
      else {
        for (const key of MDQA_METRIC_KEYS) {
          if (typeof data[key] === 'number') {
            const stdKey = `${key}_std`
            mdqa[key] = {
              mean: data[key],
              std: typeof data[stdKey] === 'number' ? data[stdKey] : undefined,
            }
          }
        }
      }

      if (Object.keys(mdqa).length > 0) {
        result.mdqa = mdqa
      }
    }

    // Try parsing as LADy-kap results
    if (metricType === 'lady' || metricType === 'both') {
      const lady: Partial<Record<LadyMetricKey, MetricStat>> = {}

      // Try multiple key formats
      const keyMappings: Record<string, LadyMetricKey> = {
        'precision_at_5': 'precision_at_5',
        'P@5': 'precision_at_5',
        'p_at_5': 'precision_at_5',
        'map_at_5': 'map_at_5',
        'MAP@5': 'map_at_5',
        'ndcg_at_5': 'ndcg_at_5',
        'NDCG@5': 'ndcg_at_5',
        'recall_at_5': 'recall_at_5',
        'R@5': 'recall_at_5',
        'r_at_5': 'recall_at_5',
        'specificity_at_5': 'specificity_at_5',
        'S@5': 'specificity_at_5',
        's_at_5': 'specificity_at_5',
      }

      for (const [fileKey, metricKey] of Object.entries(keyMappings)) {
        if (typeof data[fileKey] === 'number') {
          lady[metricKey] = { mean: data[fileKey] }
        } else if (data[fileKey] && typeof data[fileKey] === 'object' && 'mean' in data[fileKey]) {
          lady[metricKey] = data[fileKey]
        }
      }

      if (Object.keys(lady).length > 0) {
        result.lady = lady
      }
    }

    return result
  } catch {
    return { mdqa: null, perRunMetrics: null, lady: null }
  }
}
