import { useState, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'
import { DataSourcePanel } from '../data-source-panel'
import { ResearchTable } from '../research-table'
import { pairedTTest, cohensD, significanceStars, mean } from '../stats-utils'
import type { DataEntry, TableData, StatisticalResult } from '../types'
import { MDQA_METRIC_KEYS, MDQA_METRIC_LABELS } from '../types'

export function Table1cStats() {
  const [entries, setEntries] = useState<DataEntry[]>([])
  const [tableData, setTableData] = useState<TableData | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])

  // Need at least one CERA and one Heuristic entry with per-run data
  const hasPairableData = useMemo(() => {
    const ceraWithRuns = entries.some(e =>
      e.method === 'cera' && e.source.perRunMetrics && e.source.perRunMetrics.length >= 2
    )
    const heuristicWithRuns = entries.some(e =>
      e.method === 'heuristic' && e.source.perRunMetrics && e.source.perRunMetrics.length >= 2
    )
    return ceraWithRuns && heuristicWithRuns
  }, [entries])

  const generateTable = () => {
    const { data, warnings: w } = buildTable1c(entries)
    setTableData(data)
    setWarnings(w)
  }

  return (
    <div className="space-y-4">
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Statistical significance requires per-run metrics from both CERA and Heuristic
          at the same dataset size. Use jobs with 2+ runs (total_runs setting).
          Per-run data is needed for paired t-tests.
        </AlertDescription>
      </Alert>

      <DataSourcePanel
        entries={entries}
        onEntriesChange={setEntries}
        metricType="mdqa"
        fileHint="Drop mdqa-results.json with per-run metrics (multi-run format)"
      />

      {!hasPairableData && entries.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Need at least one CERA and one Heuristic entry with per-run metrics (2+ runs)
            at matching dataset sizes to compute statistical significance.
          </AlertDescription>
        </Alert>
      )}

      <Button onClick={generateTable} disabled={!hasPairableData}>
        Generate Table
      </Button>

      {warnings.length > 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {warnings.map((w, i) => <div key={i}>{w}</div>)}
          </AlertDescription>
        </Alert>
      )}

      {tableData && (
        <ResearchTable
          data={tableData}
          title="Table 1c: Statistical Significance (CERA vs Heuristic)"
          description="Paired t-test with Cohen's d effect size. * p<0.05, ** p<0.01, *** p<0.001"
          filename="rq1-table-1c-significance"
        />
      )}
    </div>
  )
}

function buildTable1c(entries: DataEntry[]): { data: TableData; warnings: string[] } {
  const warnings: string[] = []

  // Find all sizes where we have both CERA and Heuristic with per-run data
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)
  const results: StatisticalResult[] = []

  for (const size of sizes) {
    const ceraEntry = entries.find(e =>
      e.size === size && e.method === 'cera' && e.source.perRunMetrics && e.source.perRunMetrics.length >= 2
    )
    const heuristicEntry = entries.find(e =>
      e.size === size && e.method === 'heuristic' && e.source.perRunMetrics && e.source.perRunMetrics.length >= 2
    )

    if (!ceraEntry || !heuristicEntry) {
      if (entries.some(e => e.size === size)) {
        warnings.push(`Size n=${size}: Missing paired per-run data for both CERA and Heuristic`)
      }
      continue
    }

    const ceraRuns = ceraEntry.source.perRunMetrics!
    const heuristicRuns = heuristicEntry.source.perRunMetrics!
    const nPairs = Math.min(ceraRuns.length, heuristicRuns.length)

    for (const metricKey of MDQA_METRIC_KEYS) {
      const ceraValues: number[] = []
      const heuristicValues: number[] = []

      for (let i = 0; i < nPairs; i++) {
        const cv = ceraRuns[i]?.metrics[metricKey]
        const hv = heuristicRuns[i]?.metrics[metricKey]
        if (cv !== undefined && hv !== undefined) {
          ceraValues.push(cv)
          heuristicValues.push(hv)
        }
      }

      if (ceraValues.length < 2) {
        warnings.push(`Size n=${size}, ${MDQA_METRIC_LABELS[metricKey]}: Insufficient paired observations`)
        continue
      }

      const ceraMean = mean(ceraValues)
      const heuristicMean = mean(heuristicValues)
      const delta = ceraMean - heuristicMean
      const test = pairedTTest(ceraValues, heuristicValues)
      const d = cohensD(ceraValues, heuristicValues)
      const sig = significanceStars(test.p)

      results.push({
        metric: MDQA_METRIC_LABELS[metricKey],
        size,
        ceraMean,
        heuristicMean,
        delta,
        tStat: test.t,
        df: test.df,
        pValue: test.p,
        cohensD: d,
        significance: sig,
      })
    }
  }

  if (results.length === 0) {
    return {
      data: { rowHeaders: [], columnHeaders: [], cells: [] },
      warnings: [...warnings, 'No statistical results could be computed'],
    }
  }

  // Build table
  const columnHeaders = ['Size', 'Metric', 'CERA Mean', 'Heuristic Mean', '\u0394', 't(df)', 'p-value', "Cohen's d", 'Sig.']
  const rowHeaders = results.map((_, i) => String(i + 1))
  const cells = results.map(r => [
    String(r.size),
    r.metric,
    r.ceraMean.toFixed(4),
    r.heuristicMean.toFixed(4),
    `${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(4)}`,
    `${r.tStat.toFixed(2)}(${r.df})`,
    r.pValue < 0.001 ? '<0.001' : r.pValue.toFixed(4),
    r.cohensD.toFixed(2),
    r.significance,
  ])

  return {
    data: { rowHeaders, columnHeaders, cells },
    warnings,
  }
}
