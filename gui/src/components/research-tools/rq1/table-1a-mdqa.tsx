import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { DataSourcePanel } from '../data-source-panel'
import { ResearchTable } from '../research-table'
import type { DataEntry, TableData, MdqaMetricKey, Method } from '../types'
import { MDQA_METRIC_KEYS, MDQA_METRIC_LABELS, MDQA_METRIC_DIRECTION, METHOD_LABELS } from '../types'

export function Table1aMdqa() {
  const [entries, setEntries] = useState<DataEntry[]>([])
  const [tableData, setTableData] = useState<TableData | null>(null)

  const canGenerate = entries.length > 0

  const generateTable = () => {
    const data = buildTable1a(entries)
    setTableData(data)
  }

  return (
    <div className="space-y-4">
      <DataSourcePanel
        entries={entries}
        onEntriesChange={setEntries}
        metricType="mdqa"
        fileHint="Drop mdqa-results.json from a CERA/Heuristic job"
      />

      <Button onClick={generateTable} disabled={!canGenerate}>
        Generate Table
      </Button>

      {tableData && (
        <ResearchTable
          data={tableData}
          title="Table 1a: MDQA Metrics (Intrinsic Quality)"
          description="Mean ± SD across runs. Arrows indicate desired direction."
          filename="rq1-table-1a-mdqa"
        />
      )}
    </div>
  )
}

/**
 * Build Table 1a from data entries.
 * Rows: metrics (BLEU, ROUGE-L, etc.)
 * Columns: grouped by size, then by method within each size.
 * Supports multiple entries per (method, size) for per-model comparisons.
 */
function buildTable1a(entries: DataEntry[]): TableData {
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)
  const methodOrder: Method[] = ['real', 'cera', 'heuristic']

  // Build ordered entry list per size (sorted by method priority, then model slug)
  const columnEntries: DataEntry[] = []
  const columnHeaders: string[] = []
  const columnSubLabels: (string | null)[] = []
  const columnGroups: { label: string; span: number }[] = []

  for (const size of sizes) {
    const entriesForSize = entries
      .filter(e => e.size === size)
      .sort((a, b) => {
        const mi = methodOrder.indexOf(a.method) - methodOrder.indexOf(b.method)
        if (mi !== 0) return mi
        return (a.modelSlug || '').localeCompare(b.modelSlug || '')
      })

    columnGroups.push({ label: `n=${size}`, span: entriesForSize.length })

    for (const entry of entriesForSize) {
      columnEntries.push(entry)
      columnHeaders.push(METHOD_LABELS[entry.method])
      // Always show model slug when present so user can tell which model produced the results
      columnSubLabels.push(entry.modelSlug || null)
    }
  }

  // Build row headers with direction arrows
  const rowHeaders = MDQA_METRIC_KEYS.map(key => {
    const arrow = MDQA_METRIC_DIRECTION[key] === 'up' ? '\u2191' : '\u2193'
    return `${MDQA_METRIC_LABELS[key]} ${arrow}`
  })

  // Build cells
  const cells: string[][] = MDQA_METRIC_KEYS.map(metricKey => {
    return columnEntries.map(entry => formatMetricValue(entry, metricKey))
  })

  const hasAnySub = columnSubLabels.some(l => l !== null)
  return { rowHeaders, columnHeaders, cells, columnGroups, columnSubLabels: hasAnySub ? columnSubLabels : undefined }
}

/**
 * Format a metric value as "mean ± std" or just "mean".
 */
function formatMetricValue(entry: DataEntry, metricKey: MdqaMetricKey): string {
  const metrics = entry.source.type === 'job'
    ? entry.source.mdqaMetrics
    : entry.source.mdqaMetrics

  if (!metrics) return '---'

  const stat = metrics[metricKey]
  if (!stat) return '---'

  if (stat.std !== undefined && stat.std > 0) {
    return `${stat.mean.toFixed(4)} \u00B1 ${stat.std.toFixed(4)}`
  }
  return stat.mean.toFixed(4)
}
