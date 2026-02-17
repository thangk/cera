import { useState, useCallback } from 'react'
import { useQuery } from 'convex/react'
import { api } from '../../../convex/_generated/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { X, Database, Loader2 } from 'lucide-react'
import type {
  DataEntry,
  Method,
  JobSource,
  MetricStat,
  MdqaMetricKey,
  PerRunMetrics,
  ScannedTarget,
  ScanTargetsResponse,
} from './types'
import { MDQA_METRIC_KEYS } from './types'
import { PYTHON_API_URL } from '@/lib/api-urls'

interface BatchDataSourcePanelProps {
  entries: DataEntry[]
  onEntriesChange: (entries: DataEntry[]) => void
}

export function BatchDataSourcePanel({ entries, onEntriesChange }: BatchDataSourcePanelProps) {
  const removeEntry = useCallback((id: string) => {
    onEntriesChange(entries.filter(e => e.id !== id))
  }, [entries, onEntriesChange])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Data Sources</CardTitle>
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

        {/* Always-visible batch add form */}
        <BatchAddForm
          onAdd={(newEntries) => onEntriesChange([...entries, ...newEntries])}
        />
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

  const hasMetrics = entry.source.mdqaMetrics !== null

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-md border bg-muted/30">
      <Badge variant="outline" className={methodColors[entry.method]}>
        {entry.method.toUpperCase()}
      </Badge>
      <span className="text-sm font-mono">n={entry.size}</span>
      <span className="text-sm text-muted-foreground flex-1 truncate">
        <Database className="h-3 w-3 inline mr-1" />
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

/**
 * Parse read-metrics API response into MDQA metrics with std.
 */
function parseApiMetrics(metrics: Record<string, unknown>): {
  mdqaMetrics: Partial<Record<MdqaMetricKey, MetricStat>>
  perRunMetrics: PerRunMetrics[] | null
  nRuns: number | undefined
} {
  const mdqaMetrics: Partial<Record<MdqaMetricKey, MetricStat>> = {}
  let perRunMetrics: PerRunMetrics[] | null = null

  // Parse aggregate metrics from API response categories
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

  // Extract std
  if (metrics.std && typeof metrics.std === 'object') {
    const std = metrics.std as Record<string, unknown>
    for (const category of ['lexical', 'semantic', 'diversity']) {
      const catStd = std[category]
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

  // Extract per-run metrics
  if (metrics.perRun && typeof metrics.perRun === 'object') {
    const perRun = metrics.perRun as Record<string, unknown>
    const runMap = new Map<number, Partial<Record<MdqaMetricKey, number>>>()
    for (const category of ['lexical', 'semantic', 'diversity']) {
      const catRuns = perRun[category]
      if (catRuns && typeof catRuns === 'object') {
        for (const [runKey, runMetrics] of Object.entries(catRuns as Record<string, unknown>)) {
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
        .map(([run, m]) => ({ run, metrics: m }))
    }
  }

  // Extract totalRuns (from multi-run or heuristic with run directories)
  const nRuns = typeof metrics.totalRuns === 'number' ? metrics.totalRuns
    : perRunMetrics ? perRunMetrics.length
    : undefined

  return { mdqaMetrics, perRunMetrics, nRuns }
}

function BatchAddForm({ onAdd }: { onAdd: (entries: DataEntry[]) => void }) {
  const [method, setMethod] = useState<Method>('cera')
  const completedJobs = useQuery(api.jobs.listForResearch)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [scannedTargets, setScannedTargets] = useState<ScannedTarget[]>([])
  const [selectedTargets, setSelectedTargets] = useState<Set<number>>(new Set())
  const [selectedModelSlug, setSelectedModelSlug] = useState<string | null>(null)
  const [isScanning, setIsScanning] = useState(false)
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)

  const handleMethodChange = useCallback((v: string) => {
    setMethod(v as Method)
    setSelectedJobId(null)
    setScannedTargets([])
    setSelectedTargets(new Set())
    setSelectedModelSlug(null)
    setScanError(null)
  }, [])

  const handleJobSelect = useCallback(async (jobId: string) => {
    setSelectedJobId(jobId)
    setScannedTargets([])
    setSelectedTargets(new Set())
    setSelectedModelSlug(null)
    setScanError(null)

    const job = completedJobs?.find((j: { _id: string }) => j._id === jobId)
    if (!job) return

    // Auto-detect method from job
    const jobMethod = (job as { method?: string }).method
    if (jobMethod === 'cera' || jobMethod === 'heuristic') {
      setMethod(jobMethod as Method)
    }

    const jobDir = (job as { jobDir?: string }).jobDir
    if (!jobDir) {
      setScanError('Job has no directory path stored.')
      return
    }

    setIsScanning(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/scan-targets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobDir }),
      })
      if (!res.ok) {
        throw new Error(`Scan failed (${res.status})`)
      }
      const scanData: ScanTargetsResponse = await res.json()
      const withMetrics = scanData.targets.filter(t => t.hasMetrics)
      setScannedTargets(withMetrics)
      setSelectedTargets(new Set(withMetrics.map(t => t.targetValue)))
    } catch {
      setScanError('Failed to scan job directory. Ensure Python API is running.')
    } finally {
      setIsScanning(false)
    }
  }, [completedJobs])

  const handleTargetToggle = (targetValue: number) => {
    setSelectedTargets(prev => {
      const next = new Set(prev)
      if (next.has(targetValue)) next.delete(targetValue)
      else next.add(targetValue)
      return next
    })
  }

  const handleBatchAdd = useCallback(async () => {
    const job = completedJobs?.find((j: { _id: string }) => j._id === selectedJobId)
    if (!job) return
    const jobDir = (job as { jobDir?: string }).jobDir
    if (!jobDir) return

    setIsLoadingMetrics(true)
    try {
      const loadPromises = Array.from(selectedTargets).map(async (targetValue): Promise<DataEntry | null> => {
        try {
          const res = await fetch(`${PYTHON_API_URL}/api/read-metrics`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jobDir, targetSize: targetValue }),
          })
          if (!res.ok) return null

          const apiData = await res.json()
          const metrics = apiData.metrics
          if (!metrics) return null

          const { mdqaMetrics, perRunMetrics, nRuns } = parseApiMetrics(metrics)

          const suffix = selectedModelSlug ? ` [${selectedModelSlug}]` : ''
          const jobSource: JobSource = {
            type: 'job',
            jobId: job._id,
            jobName: `${job.name}${suffix}`,
            mdqaMetrics: Object.keys(mdqaMetrics).length > 0 ? mdqaMetrics : null,
            perRunMetrics,
            perModelMetrics: null,
            ladyMetrics: null,
            perRunLadyMetrics: null,
            nRuns,
          }

          // Resolve model slug: explicit selection > single-model job fallback
          let resolvedModelSlug = selectedModelSlug || undefined
          if (!resolvedModelSlug) {
            const slugs = (job as { modelSlugs?: string[] }).modelSlugs
            if (slugs && slugs.length === 1) resolvedModelSlug = slugs[0]
          }

          return {
            id: crypto.randomUUID(),
            method,
            size: targetValue,
            modelSlug: resolvedModelSlug,
            source: jobSource,
          }
        } catch {
          return null
        }
      })

      const results = await Promise.all(loadPromises)
      const valid = results.filter((e): e is DataEntry => e !== null)
      if (valid.length > 0) onAdd(valid)
    } finally {
      setIsLoadingMetrics(false)
      // Reset form after adding
      setSelectedJobId(null)
      setScannedTargets([])
      setSelectedTargets(new Set())
      setSelectedModelSlug(null)
    }
  }, [completedJobs, selectedJobId, selectedTargets, selectedModelSlug, method, onAdd])

  // Detect multi-model from scanned targets
  const modelSlugs = [...new Set(scannedTargets.flatMap(t => t.modelSlugs))]
  const needsModelSelection = modelSlugs.length > 1 && !selectedModelSlug
  const canAdd = selectedJobId && selectedTargets.size > 0 && !isLoadingMetrics && !needsModelSelection

  // Filter jobs by method
  const filteredJobs = completedJobs?.filter((job: { method: string }) => {
    return (job.method || 'cera') === method
  })

  return (
    <div className="border rounded-lg p-4 space-y-4 bg-background">
      <span className="text-sm font-medium">Add from Job</span>

      {/* Method + Job row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs">Method</Label>
          <Select value={method} onValueChange={handleMethodChange}>
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
          <Label className="text-xs">Job</Label>
          <Select value={selectedJobId || ''} onValueChange={handleJobSelect}>
            <SelectTrigger>
              <SelectValue placeholder="Select a completed job..." />
            </SelectTrigger>
            <SelectContent className="max-h-60">
              {filteredJobs?.map((job: { _id: string; name: string; method: string; generationCount: number; totalRuns: number; targets?: Array<{ value: number }> | null }) => (
                <SelectItem key={job._id} value={job._id}>
                  <span className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px] px-1">
                      {(job.method || 'cera').toUpperCase()}
                    </Badge>
                    {job.name}
                    <span className="text-muted-foreground ml-1">
                      {job.targets && job.targets.length > 0
                        ? `(${job.targets.map(t => t.value).join(', ')} sents, ${job.totalRuns} run${job.totalRuns > 1 ? 's' : ''})`
                        : `(n=${job.generationCount}, ${job.totalRuns} run${job.totalRuns > 1 ? 's' : ''})`
                      }
                    </span>
                  </span>
                </SelectItem>
              ))}
              {(!filteredJobs || filteredJobs.length === 0) && (
                <div className="px-2 py-4 text-sm text-muted-foreground text-center">
                  {method === 'real'
                    ? 'No completed jobs found'
                    : `No completed ${method.toUpperCase()} jobs found`}
                </div>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Scanning state */}
      {isScanning && (
        <p className="text-xs text-muted-foreground flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" />
          Scanning job directory for targets...
        </p>
      )}

      {scanError && (
        <p className="text-xs text-destructive">{scanError}</p>
      )}

      {/* Model selector (if multi-model) */}
      {modelSlugs.length > 1 && (
        <div className="space-y-1.5">
          <Label className="text-xs text-amber-600 dark:text-amber-400">
            Multi-model job detected — select one model
          </Label>
          <Select value={selectedModelSlug || ''} onValueChange={setSelectedModelSlug}>
            <SelectTrigger>
              <SelectValue placeholder="Select a model..." />
            </SelectTrigger>
            <SelectContent>
              {modelSlugs.map(slug => (
                <SelectItem key={slug} value={slug}>{slug}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Target size checkboxes */}
      {scannedTargets.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Target Sizes</Label>
            <span className="text-xs text-muted-foreground">
              {selectedTargets.size} of {scannedTargets.length} selected
            </span>
          </div>
          <div className="flex flex-wrap gap-3">
            {scannedTargets.map(target => (
              <label
                key={target.targetValue}
                className="flex items-center gap-1.5 cursor-pointer"
              >
                <Checkbox
                  checked={selectedTargets.has(target.targetValue)}
                  onCheckedChange={() => handleTargetToggle(target.targetValue)}
                />
                <span className="text-sm font-mono">{target.targetValue}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Add button */}
      {scannedTargets.length > 0 && (
        <div className="flex justify-end">
          <Button size="sm" onClick={handleBatchAdd} disabled={!canAdd}>
            {isLoadingMetrics && <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />}
            {isLoadingMetrics
              ? 'Loading...'
              : `Add ${selectedTargets.size} Target${selectedTargets.size !== 1 ? 's' : ''}`}
          </Button>
        </div>
      )}
    </div>
  )
}
