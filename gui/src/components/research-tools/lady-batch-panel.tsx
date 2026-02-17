import { useState, useCallback } from 'react'
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
  LadyMetricKey,
  MetricStat,
  LadyOutputDir,
  PerRunLadyMetrics,
} from './types'
import { PYTHON_API_URL } from '@/lib/api-urls'

interface LadyBatchPanelProps {
  entries: DataEntry[]
  onEntriesChange: (entries: DataEntry[]) => void
}

export function LadyBatchPanel({ entries, onEntriesChange }: LadyBatchPanelProps) {
  const removeEntry = useCallback((id: string) => {
    onEntriesChange(entries.filter(e => e.id !== id))
  }, [entries, onEntriesChange])

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Data Sources</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {entries.length > 0 && (
          <div className="space-y-2">
            {entries.map(entry => (
              <EntryRow key={entry.id} entry={entry} onRemove={removeEntry} />
            ))}
          </div>
        )}

        <LadyBatchAddForm
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

  const hasMetrics = entry.source.type === 'job'
    ? entry.source.ladyMetrics !== null
    : entry.source.ladyMetrics !== null

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

function LadyBatchAddForm({ onAdd }: { onAdd: (entries: DataEntry[]) => void }) {
  const [method, setMethod] = useState<Method>('cera')
  const [ladyOutputs, setLadyOutputs] = useState<LadyOutputDir[]>([])
  const [selectedOutputPath, setSelectedOutputPath] = useState<string | null>(null)
  const [selectedTargets, setSelectedTargets] = useState<Set<number>>(new Set())
  const [isScanning, setIsScanning] = useState(false)
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)
  const [hasScanned, setHasScanned] = useState(false)

  const handleMethodChange = useCallback((v: string) => {
    setMethod(v as Method)
    setSelectedOutputPath(null)
    setSelectedTargets(new Set())
    setScanError(null)
  }, [])

  // Scan for LADy outputs on first render or method change
  const handleScan = useCallback(async () => {
    setIsScanning(true)
    setScanError(null)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/scan-lady-outputs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      if (!res.ok) throw new Error(`Scan failed (${res.status})`)
      const data = await res.json()
      setLadyOutputs(data.outputs || [])
      setHasScanned(true)
    } catch {
      setScanError('Failed to scan LADy output directory. Ensure Python API is running and LADy output is mounted.')
    } finally {
      setIsScanning(false)
    }
  }, [])

  // Auto-scan on first interaction
  const handleMethodSelect = useCallback((v: string) => {
    handleMethodChange(v)
    if (!hasScanned) handleScan()
  }, [handleMethodChange, hasScanned, handleScan])

  const handleOutputSelect = useCallback((path: string) => {
    setSelectedOutputPath(path)
    const output = ladyOutputs.find(o => o.path === path)
    if (output) {
      setMethod(output.type as Method)
      setSelectedTargets(new Set(output.targets))
    }
  }, [ladyOutputs])

  const handleTargetToggle = (targetValue: number) => {
    setSelectedTargets(prev => {
      const next = new Set(prev)
      if (next.has(targetValue)) next.delete(targetValue)
      else next.add(targetValue)
      return next
    })
  }

  const handleBatchAdd = useCallback(async () => {
    if (!selectedOutputPath) return
    const output = ladyOutputs.find(o => o.path === selectedOutputPath)
    if (!output) return

    setIsLoadingMetrics(true)
    try {
      const loadPromises = Array.from(selectedTargets).map(async (targetValue): Promise<DataEntry | null> => {
        try {
          const res = await fetch(`${PYTHON_API_URL}/api/read-lady-metrics`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ outputDir: selectedOutputPath, targetSize: targetValue }),
          })
          if (!res.ok) return null

          const apiData = await res.json()
          const ladyMetrics: Partial<Record<LadyMetricKey, MetricStat>> = apiData.metrics || {}
          const perRunLady: PerRunLadyMetrics[] | null = apiData.perRun || null

          const jobSource: JobSource = {
            type: 'job',
            jobId: output.name,
            jobName: `LADy: ${output.name}`,
            mdqaMetrics: null,
            perRunMetrics: null,
            perModelMetrics: null,
            ladyMetrics: Object.keys(ladyMetrics).length > 0 ? ladyMetrics : null,
            perRunLadyMetrics: perRunLady,
            nRuns: apiData.nRuns,
          }

          return {
            id: crypto.randomUUID(),
            method,
            size: targetValue,
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
      setSelectedOutputPath(null)
      setSelectedTargets(new Set())
    }
  }, [ladyOutputs, selectedOutputPath, selectedTargets, method, onAdd])

  // Filter outputs by method
  const filteredOutputs = ladyOutputs.filter(o => o.type === method)

  // Get target sizes for selected output
  const selectedOutput = ladyOutputs.find(o => o.path === selectedOutputPath)
  const availableTargets = selectedOutput?.targets ?? []

  const canAdd = selectedOutputPath && selectedTargets.size > 0 && !isLoadingMetrics

  return (
    <div className="border rounded-lg p-4 space-y-4 bg-background">
      <span className="text-sm font-medium">Add from LADy Output</span>

      {/* Method + Output row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs">Method</Label>
          <Select value={method} onValueChange={handleMethodSelect}>
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
          <Label className="text-xs">LADy Output</Label>
          <div className="flex gap-2">
            <Select
              value={selectedOutputPath || ''}
              onValueChange={handleOutputSelect}
            >
              <SelectTrigger className="flex-1">
                <SelectValue placeholder="Select LADy output..." />
              </SelectTrigger>
              <SelectContent className="max-h-60">
                {filteredOutputs.map(output => (
                  <SelectItem key={output.path} value={output.path}>
                    <span className="flex items-center gap-2">
                      {output.name}
                      <span className="text-muted-foreground ml-1">
                        ({output.targets.join(', ')} sents)
                      </span>
                    </span>
                  </SelectItem>
                ))}
                {filteredOutputs.length === 0 && hasScanned && (
                  <div className="px-2 py-4 text-sm text-muted-foreground text-center">
                    No LADy outputs found for {method.toUpperCase()}
                  </div>
                )}
                {!hasScanned && (
                  <div className="px-2 py-4 text-sm text-muted-foreground text-center">
                    Click Scan to discover LADy outputs
                  </div>
                )}
              </SelectContent>
            </Select>
            <Button
              variant="outline"
              size="sm"
              onClick={handleScan}
              disabled={isScanning}
              className="shrink-0"
            >
              {isScanning ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Scan'}
            </Button>
          </div>
        </div>
      </div>

      {/* Scanning state */}
      {isScanning && (
        <p className="text-xs text-muted-foreground flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" />
          Scanning LADy output directory...
        </p>
      )}

      {scanError && (
        <p className="text-xs text-destructive">{scanError}</p>
      )}

      {/* Target size checkboxes */}
      {availableTargets.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Target Sizes</Label>
            <span className="text-xs text-muted-foreground">
              {selectedTargets.size} of {availableTargets.length} selected
            </span>
          </div>
          <div className="flex flex-wrap gap-3">
            {availableTargets.map(target => (
              <label
                key={target}
                className="flex items-center gap-1.5 cursor-pointer"
              >
                <Checkbox
                  checked={selectedTargets.has(target)}
                  onCheckedChange={() => handleTargetToggle(target)}
                />
                <span className="text-sm font-mono">{target}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Add button */}
      {availableTargets.length > 0 && (
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
