import { useState, useCallback, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { DataSourcePanel } from '../data-source-panel'
import { LadyBatchPanel } from '../lady-batch-panel'
import { ResearchTable } from '../research-table'
import type { DataEntry, TableData, LadyMetricKey, Method } from '../types'
import { LADY_METRIC_KEYS, LADY_METRIC_LABELS, LADY_METRIC_DIRECTION, METHOD_LABELS } from '../types'

const STORAGE_KEY = 'rq1b-entries'

function loadEntries(): DataEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore corrupt data */ }
  return []
}

export function Table1bLady() {
  const [entries, setEntries] = useState<DataEntry[]>(loadEntries)
  const [tableData, setTableData] = useState<TableData | null>(() => {
    const initial = loadEntries()
    return initial.length > 0 ? buildTable1b(initial) : null
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  }, [entries])

  const canGenerate = entries.length > 0

  const generateTable = useCallback(() => {
    const data = buildTable1b(entries)
    setTableData(data)
  }, [entries])

  return (
    <div className="space-y-4">
      <Tabs defaultValue="batch">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="batch">Batch Add (LADy Output)</TabsTrigger>
          <TabsTrigger value="single">Single Add</TabsTrigger>
        </TabsList>

        <TabsContent value="batch" className="mt-4">
          <LadyBatchPanel
            entries={entries}
            onEntriesChange={setEntries}
          />
        </TabsContent>

        <TabsContent value="single" className="mt-4">
          <DataSourcePanel
            entries={entries}
            onEntriesChange={setEntries}
            metricType="lady"
            fileHint="Drop LADy aggregate.csv or JSON results"
          />
        </TabsContent>
      </Tabs>

      <Button onClick={generateTable} disabled={!canGenerate}>
        Generate Table
      </Button>

      {tableData && (
        <ResearchTable
          data={tableData}
          title="Table 1b: LADy Metrics (Extrinsic Utility)"
          description="Mean ± SD across runs. All metrics: higher is better."
          filename="rq1-table-1b-lady"
        />
      )}
    </div>
  )
}

/**
 * Build Table 1b from data entries.
 * Rows: LADy metrics (P@5, MAP@5, NDCG@5, R@5, S@5)
 * Columns: grouped by size, then by method within each size.
 */
function buildTable1b(entries: DataEntry[]): TableData {
  const sizes = [...new Set(entries.map(e => e.size))].sort((a, b) => a - b)
  const methodOrder: Method[] = ['real', 'cera', 'heuristic']

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
      columnSubLabels.push(entry.modelSlug || null)
    }
  }

  // Build row headers with direction arrows
  const rowHeaders = LADY_METRIC_KEYS.map(key => {
    const arrow = LADY_METRIC_DIRECTION[key] === 'up' ? '\u2191' : '\u2193'
    return `${LADY_METRIC_LABELS[key]} ${arrow}`
  })

  // Build cells and raw numeric values
  const cells: string[][] = LADY_METRIC_KEYS.map(metricKey => {
    return columnEntries.map(entry => formatLadyValue(entry, metricKey))
  })
  const cellValues: (number | null)[][] = LADY_METRIC_KEYS.map(metricKey => {
    return columnEntries.map(entry => extractNumericValue(entry, metricKey))
  })
  const metricDirections = LADY_METRIC_KEYS.map(key => LADY_METRIC_DIRECTION[key])

  const hasAnySub = columnSubLabels.some(l => l !== null)
  return { rowHeaders, columnHeaders, cells, columnGroups, columnSubLabels: hasAnySub ? columnSubLabels : undefined, cellValues, metricDirections }
}

/**
 * Extract the raw numeric mean value for color coding.
 */
function extractNumericValue(entry: DataEntry, metricKey: LadyMetricKey): number | null {
  const metrics = getLadyMetrics(entry)
  if (!metrics) return null
  const stat = metrics[metricKey]
  return stat ? stat.mean : null
}

/**
 * Get LADy metrics from either JobSource or FileSource.
 */
function getLadyMetrics(entry: DataEntry) {
  return entry.source.ladyMetrics
}

/**
 * Format a metric value as "mean ± std" or just "mean".
 */
function formatLadyValue(entry: DataEntry, metricKey: LadyMetricKey): string {
  const metrics = getLadyMetrics(entry)
  if (!metrics) return '---'

  const stat = metrics[metricKey]
  if (!stat) return '---'

  if (stat.std !== undefined && stat.std > 0) {
    return `${stat.mean.toFixed(4)} \u00B1 ${stat.std.toFixed(4)}`
  }
  return stat.mean.toFixed(4)
}
