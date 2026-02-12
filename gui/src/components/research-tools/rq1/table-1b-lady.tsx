import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { AlertCircle } from 'lucide-react'
import { DataSourcePanel } from '../data-source-panel'
import { ResearchTable } from '../research-table'
import type { DataEntry, TableData, LadyMetricKey, Method } from '../types'
import { LADY_METRIC_KEYS, LADY_METRIC_LABELS, METHOD_LABELS } from '../types'

export function Table1bLady() {
  const [entries, setEntries] = useState<DataEntry[]>([])
  const [tableData, setTableData] = useState<TableData | null>(null)

  const canGenerate = entries.length > 0

  const generateTable = () => {
    const data = buildTable1b(entries)
    setTableData(data)
  }

  return (
    <div className="space-y-4">
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          LADy-kap metrics parser is a placeholder. Drop a JSON file with keys like
          "P@5", "MAP@5", "NDCG@5", "R@5", "S@5" (or snake_case variants).
          The format will be finalized once LADy-kap evaluation outputs are available.
        </AlertDescription>
      </Alert>

      <DataSourcePanel
        entries={entries}
        onEntriesChange={setEntries}
        metricType="lady"
        fileHint="Drop LADy-kap evaluation results (JSON)"
      />

      <Button onClick={generateTable} disabled={!canGenerate}>
        Generate Table
      </Button>

      {tableData && (
        <ResearchTable
          data={tableData}
          title="Table 1b: LADy-kap Metrics (Extrinsic Utility)"
          description="Information retrieval metrics from LADy-kap downstream evaluation"
          filename="rq1-table-1b-ladykap"
        />
      )}
    </div>
  )
}

function buildTable1b(entries: DataEntry[]): TableData {
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)
  const methods: Method[] = ['real', 'cera', 'heuristic']

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

  const rowHeaders = LADY_METRIC_KEYS.map(key => `${LADY_METRIC_LABELS[key]} \u2191`)

  const cells: string[][] = LADY_METRIC_KEYS.map(metricKey => {
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
        row.push(formatLadyValue(entry, metricKey))
      }
    }
    return row
  })

  return { rowHeaders, columnHeaders, cells, columnGroups }
}

function formatLadyValue(entry: DataEntry, metricKey: LadyMetricKey): string {
  if (entry.source.type !== 'file') return '---'
  const metrics = entry.source.ladyMetrics
  if (!metrics) return '---'

  const stat = metrics[metricKey]
  if (!stat) return '---'

  if (stat.std !== undefined && stat.std > 0) {
    return `${stat.mean.toFixed(4)} \u00B1 ${stat.std.toFixed(4)}`
  }
  return stat.mean.toFixed(4)
}
