import { useState, useCallback, useEffect, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'
import { DataSourcePanel } from '../data-source-panel'
import { BatchDataSourcePanel } from '../batch-data-source-panel'
import { ResearchTable } from '../research-table'
import { pairedTTest, cohensD, welchTTest, cohensDFromSummary, significanceStars, mean, stdev } from '../stats-utils'
import type { DataEntry, TableData, Method } from '../types'
import { MDQA_METRIC_KEYS, MDQA_METRIC_LABELS, METHOD_LABELS } from '../types'

const STORAGE_KEY = 'rq1c-entries'
const BASELINE_KEY = 'rq1c-baseline'

function loadEntries(): DataEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore corrupt data */ }
  return []
}

function loadBaseline(): Method | 'none' {
  try {
    const raw = localStorage.getItem(BASELINE_KEY)
    if (raw === 'real' || raw === 'cera' || raw === 'heuristic') return raw
  } catch { /* ignore */ }
  return 'none'
}

export function Table1cStats() {
  const [entries, setEntries] = useState<DataEntry[]>(loadEntries)
  const [baselineMethod, setBaselineMethod] = useState<Method | 'none'>(loadBaseline)
  const [tableResult, setTableResult] = useState<{ data: TableData; warnings: string[] } | null>(null)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  }, [entries])

  useEffect(() => {
    localStorage.setItem(BASELINE_KEY, baselineMethod)
  }, [baselineMethod])

  // Available methods from current entries
  const availableMethods = useMemo(() => {
    const methods = new Set(entries.map(e => e.method))
    return Array.from(methods)
  }, [entries])

  // Check if we can compute statistics (per-run data OR summary stats with std)
  const hasComparableData = useMemo(() => {
    if (baselineMethod === 'none') return false
    const baselineHasData = entries.some(e =>
      e.method === baselineMethod && e.source.mdqaMetrics !== null
    )
    const otherHasData = entries.some(e =>
      e.method !== baselineMethod && e.source.mdqaMetrics !== null
    )
    return baselineHasData && otherHasData
  }, [entries, baselineMethod])

  const canGenerate = baselineMethod !== 'none' && hasComparableData

  const generateTable = useCallback(() => {
    if (baselineMethod === 'none') return
    setTableResult(buildSignificanceTable(entries, baselineMethod))
  }, [entries, baselineMethod])

  return (
    <div className="space-y-4">
      <Tabs defaultValue="batch">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="batch">Batch Add (Multi-Target)</TabsTrigger>
          <TabsTrigger value="single">Single Add</TabsTrigger>
        </TabsList>

        <TabsContent value="batch" className="mt-4">
          <BatchDataSourcePanel
            entries={entries}
            onEntriesChange={setEntries}
          />
        </TabsContent>

        <TabsContent value="single" className="mt-4">
          <DataSourcePanel
            entries={entries}
            onEntriesChange={setEntries}
            metricType="mdqa"
            fileHint="Drop mdqa-results.json with per-run metrics (multi-run format)"
          />
        </TabsContent>
      </Tabs>

      {/* Baseline selector + Generate */}
      <div className="flex items-end gap-4">
        <div className="space-y-1.5">
          <Label className="text-xs">Compare Against (Baseline)</Label>
          <Select value={baselineMethod} onValueChange={(v) => setBaselineMethod(v as Method | 'none')}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Select baseline...</SelectItem>
              {availableMethods.map(m => (
                <SelectItem key={m} value={m}>{METHOD_LABELS[m]}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button onClick={generateTable} disabled={!canGenerate}>
          Generate Table
        </Button>
      </div>

      {/* Warnings when requirements not met */}
      {baselineMethod === 'none' && entries.length > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Select a baseline method above to compare against.
          </AlertDescription>
        </Alert>
      )}

      {baselineMethod !== 'none' && !hasComparableData && entries.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Need metric data for both the baseline ({METHOD_LABELS[baselineMethod]})
            and at least one other method at matching dataset sizes.
          </AlertDescription>
        </Alert>
      )}

      {/* Significance table warnings */}
      {tableResult && tableResult.warnings.length > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {tableResult.warnings.map((w, i) => <div key={i}>{w}</div>)}
          </AlertDescription>
        </Alert>
      )}

      {/* Main significance table */}
      {tableResult && tableResult.data.rowHeaders.length > 0 && (
        <ResearchTable
          data={tableResult.data}
          title="Table 1c: Statistical Significance"
          description={`t-test vs ${baselineMethod !== 'none' ? METHOD_LABELS[baselineMethod] : ''}. * p<0.05, ** p<0.01, *** p<0.001`}
          filename="rq1-table-1c-significance"
        />
      )}
    </div>
  )
}

// ──────────────────────────────────────────────
// Statistical significance table builder
// ──────────────────────────────────────────────

function buildSignificanceTable(
  entries: DataEntry[],
  baselineMethod: Method,
): { data: TableData; warnings: string[] } {
  const warnings: string[] = []
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)

  // Find all non-baseline methods that have any metrics
  const comparisonMethods = [...new Set(
    entries
      .filter(e => e.method !== baselineMethod && e.source.mdqaMetrics !== null)
      .map(e => e.method)
  )]

  interface SigResult {
    size: number
    metric: string
    comparisonMethod: string
    compMean: number
    baseMean: number
    delta: number
    tStat: number
    df: number
    pValue: number
    d: number
    sig: string
    testType: 'paired' | 'welch'
  }

  const results: SigResult[] = []

  for (const size of sizes) {
    const baseEntry = entries.find(e =>
      e.size === size && e.method === baselineMethod && e.source.mdqaMetrics !== null
    )
    if (!baseEntry) continue

    for (const compMethod of comparisonMethods) {
      const compEntry = entries.find(e =>
        e.size === size && e.method === compMethod && e.source.mdqaMetrics !== null
      )
      if (!compEntry) continue

      const baseHasRuns = baseEntry.source.perRunMetrics && baseEntry.source.perRunMetrics.length >= 2
      const compHasRuns = compEntry.source.perRunMetrics && compEntry.source.perRunMetrics.length >= 2
      const usePairedTest = baseHasRuns && compHasRuns

      for (const metricKey of MDQA_METRIC_KEYS) {
        if (usePairedTest) {
          // Paired t-test using per-run data
          const baseRuns = baseEntry.source.perRunMetrics!
          const compRuns = compEntry.source.perRunMetrics!
          const nPairs = Math.min(baseRuns.length, compRuns.length)

          const baseValues: number[] = []
          const compValues: number[] = []

          for (let i = 0; i < nPairs; i++) {
            const bv = baseRuns[i]?.metrics[metricKey]
            const cv = compRuns[i]?.metrics[metricKey]
            if (bv !== undefined && cv !== undefined) {
              baseValues.push(bv)
              compValues.push(cv)
            }
          }

          if (compValues.length < 2) {
            warnings.push(`Size n=${size}, ${MDQA_METRIC_LABELS[metricKey]}: Insufficient paired observations`)
            continue
          }

          const compMean = mean(compValues)
          const baseMean = mean(baseValues)
          const delta = compMean - baseMean
          const test = pairedTTest(compValues, baseValues)
          const d = cohensD(compValues, baseValues)
          const sig = significanceStars(test.p)

          results.push({
            size, metric: MDQA_METRIC_LABELS[metricKey],
            comparisonMethod: METHOD_LABELS[compMethod],
            compMean, baseMean, delta,
            tStat: test.t, df: test.df, pValue: test.p, d, sig,
            testType: 'paired',
          })
        } else {
          // Welch's t-test using summary statistics (mean, std, n)
          const compStat = compEntry.source.mdqaMetrics?.[metricKey]
          const baseStat = baseEntry.source.mdqaMetrics?.[metricKey]
          if (!compStat || !baseStat) continue

          const compN = compEntry.source.nRuns
            ?? compEntry.source.perRunMetrics?.length
            ?? 0
          const baseN = baseEntry.source.nRuns
            ?? baseEntry.source.perRunMetrics?.length
            ?? 0

          const compStd = compStat.std ?? (compEntry.source.perRunMetrics
            ? stdev(compEntry.source.perRunMetrics.map(r => r.metrics[metricKey] ?? 0).filter(v => v !== 0))
            : 0)
          const baseStd = baseStat.std ?? (baseEntry.source.perRunMetrics
            ? stdev(baseEntry.source.perRunMetrics.map(r => r.metrics[metricKey] ?? 0).filter(v => v !== 0))
            : 0)

          if (compN < 2 || baseN < 2) {
            warnings.push(`Size n=${size}, ${MDQA_METRIC_LABELS[metricKey]}: Need 2+ runs for both methods (got ${compN} vs ${baseN})`)
            continue
          }

          if (compStd === 0 && baseStd === 0) continue

          const delta = compStat.mean - baseStat.mean
          const test = welchTTest(compStat.mean, compStd, compN, baseStat.mean, baseStd, baseN)
          const d = cohensDFromSummary(compStat.mean, compStd, compN, baseStat.mean, baseStd, baseN)
          const sig = significanceStars(test.p)

          results.push({
            size, metric: MDQA_METRIC_LABELS[metricKey],
            comparisonMethod: METHOD_LABELS[compMethod],
            compMean: compStat.mean, baseMean: baseStat.mean, delta,
            tStat: test.t, df: test.df, pValue: test.p, d, sig,
            testType: 'welch',
          })
        }
      }
    }
  }

  if (results.length === 0) {
    return {
      data: { rowHeaders: [], columnHeaders: [], cells: [] },
      warnings: [...warnings, 'No statistical results could be computed'],
    }
  }

  // Check if mixed test types were used
  const testTypes = new Set(results.map(r => r.testType))
  if (testTypes.has('welch')) {
    warnings.push('Some comparisons used Welch\'s t-test (independent samples) because per-run data was not available for both methods. Use "Rerun Evaluation" on heuristic jobs to get per-run data for paired t-tests.')
  }

  // Determine if we have multiple comparison methods
  const hasMultipleMethods = comparisonMethods.length > 1
  const singleCompLabel = comparisonMethods.length === 1 ? METHOD_LABELS[comparisonMethods[0]] : ''

  const columnHeaders = hasMultipleMethods
    ? ['Size', 'Method', 'Metric', 'Mean', `${METHOD_LABELS[baselineMethod]} Mean`, '\u0394', 't(df)', 'p-value', "Cohen's d", 'Sig.']
    : ['Size', 'Metric', `${singleCompLabel} Mean`, `${METHOD_LABELS[baselineMethod]} Mean`, '\u0394', 't(df)', 'p-value', "Cohen's d", 'Sig.']

  const rowHeaders = results.map((_, i) => String(i + 1))
  const cells = results.map(r => [
    String(r.size),
    ...(hasMultipleMethods ? [r.comparisonMethod] : []),
    r.metric,
    r.compMean.toFixed(4),
    r.baseMean.toFixed(4),
    `${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(4)}`,
    `${r.tStat.toFixed(2)}(${r.df})`,
    r.pValue < 0.001 ? '<0.001' : r.pValue.toFixed(4),
    r.d.toFixed(2),
    r.sig,
  ])

  return {
    data: { rowHeaders, columnHeaders, cells },
    warnings,
  }
}
