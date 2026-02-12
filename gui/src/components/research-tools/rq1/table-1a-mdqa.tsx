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
 * Columns: grouped by size, then by method within each size
 */
function buildTable1a(entries: DataEntry[]): TableData {
  // Get unique sizes and methods, sorted
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)
  const methods: Method[] = ['real', 'cera', 'heuristic']

  // Build column headers and groups
  const columnHeaders: string[] = []
  const columnGroups: { label: string; span: number }[] = []

  for (const size of sizes) {
    const methodsForSize = methods.filter(m =>
      entries.some(e => e.size === size && e.method === m)
    )
    columnGroups.push({ label: `n=${size}`, span: methodsForSize.length })
    for (const m of methodsForSize) {
      columnHeaders.push(METHOD_LABELS[m])
    }
  }

  // Build row headers with direction arrows
  const rowHeaders = MDQA_METRIC_KEYS.map(key => {
    const arrow = MDQA_METRIC_DIRECTION[key] === 'up' ? '\u2191' : '\u2193'
    return `${MDQA_METRIC_LABELS[key]} ${arrow}`
  })

  // Build cells
  const cells: string[][] = MDQA_METRIC_KEYS.map(metricKey => {
    const row: string[] = []
    for (const size of sizes) {
      const methodsForSize = methods.filter(m =>
        entries.some(e => e.size === size && e.method === m)
      )
      for (const m of methodsForSize) {
        const entry = entries.find(e => e.size === size && e.method === m)
        if (!entry) {
          row.push('---')
          continue
        }
        row.push(formatMetricValue(entry, metricKey))
      }
    }
    return row
  })

  return { rowHeaders, columnHeaders, cells, columnGroups }
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
