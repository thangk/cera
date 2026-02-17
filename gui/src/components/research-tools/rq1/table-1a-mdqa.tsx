import { useState, useCallback, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { DataSourcePanel } from '../data-source-panel'
import { BatchDataSourcePanel } from '../batch-data-source-panel'
import { ResearchTable } from '../research-table'
import type { DataEntry, TableData, MdqaMetricKey, Method } from '../types'
import { MDQA_METRIC_KEYS, MDQA_METRIC_LABELS, MDQA_METRIC_DIRECTION, METHOD_LABELS } from '../types'

const STORAGE_KEY = 'rq1a-entries'

function loadEntries(): DataEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore corrupt data */ }
  return []
}

export function Table1aMdqa() {
  const [entries, setEntries] = useState<DataEntry[]>(loadEntries)
  const [tableData, setTableData] = useState<TableData | null>(() => {
    const initial = loadEntries()
    return initial.length > 0 ? buildTable1a(initial) : null
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  }, [entries])

  const canGenerate = entries.length > 0

  const generateTable = useCallback(() => {
    const data = buildTable1a(entries)
    setTableData(data)
  }, [entries])

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
            fileHint="Drop mdqa-results.json from a CERA/Heuristic job"
          />
        </TabsContent>
      </Tabs>

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

  // Build cells and raw numeric values
  const cells: string[][] = MDQA_METRIC_KEYS.map(metricKey => {
    return columnEntries.map(entry => formatMetricValue(entry, metricKey))
  })
  const cellValues: (number | null)[][] = MDQA_METRIC_KEYS.map(metricKey => {
    return columnEntries.map(entry => extractNumericValue(entry, metricKey))
  })
  const metricDirections = MDQA_METRIC_KEYS.map(key => MDQA_METRIC_DIRECTION[key])

  const hasAnySub = columnSubLabels.some(l => l !== null)
  return { rowHeaders, columnHeaders, cells, columnGroups, columnSubLabels: hasAnySub ? columnSubLabels : undefined, cellValues, metricDirections }
}

/**
 * Extract the raw numeric mean value for color coding.
 */
function extractNumericValue(entry: DataEntry, metricKey: MdqaMetricKey): number | null {
  const metrics = entry.source.mdqaMetrics
  if (!metrics) return null
  const stat = metrics[metricKey]
  return stat ? stat.mean : null
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
