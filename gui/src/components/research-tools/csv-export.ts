/** CSV export utility for Research Tools tables */

import type { TableData } from './types'

export interface CsvExportOptions {
  showSubLabels?: boolean
  showSd?: boolean
}

/**
 * Convert TableData to CSV string.
 * Handles column groups by adding a header row.
 * Respects view options (sub-labels, SD visibility).
 */
export function tableDataToCsv(data: TableData, options?: CsvExportOptions): string {
  const showSubLabels = options?.showSubLabels ?? true
  const showSd = options?.showSd ?? true
  const rows: string[][] = []

  // Add column group header row if present
  if (data.columnGroups && data.columnGroups.length > 0) {
    const groupRow = ['']
    for (const group of data.columnGroups) {
      groupRow.push(group.label)
      // Fill remaining columns in this group with empty strings
      for (let i = 1; i < group.span; i++) {
        groupRow.push('')
      }
    }
    rows.push(groupRow)
  }

  // Add column header row (merge sub-labels into headers for CSV if visible)
  if (showSubLabels && data.columnSubLabels?.some(l => l !== null)) {
    rows.push(['', ...data.columnHeaders.map((h, i) => {
      const sub = data.columnSubLabels?.[i]
      return sub ? `${h} (${sub})` : h
    })])
  } else {
    rows.push(['', ...data.columnHeaders])
  }

  // Add data rows (strip SD if hidden)
  for (let i = 0; i < data.rowHeaders.length; i++) {
    const cells = (data.cells[i] || []).map(cell =>
      showSd ? cell : cell.replace(/\s*\u00B1\s*\S+/, '')
    )
    rows.push([data.rowHeaders[i], ...cells])
  }

  // Convert to CSV with proper escaping
  return rows
    .map(row => row.map(cell => escapeCsvCell(cell)).join(','))
    .join('\n')
}

/**
 * Escape a CSV cell value.
 * Wraps in quotes if the value contains commas, quotes, or newlines.
 */
function escapeCsvCell(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`
  }
  return value
}

/**
 * Trigger a CSV file download in the browser.
 * Adds UTF-8 BOM so Excel correctly renders ± and other unicode.
 */
export function downloadCsv(data: TableData, filename: string, options?: CsvExportOptions): void {
  const csv = tableDataToCsv(data, options)
  const bom = '\uFEFF'
  const blob = new Blob([bom + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.endsWith('.csv') ? filename : `${filename}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

/**
 * Transpose a TableData structure (swap rows and columns).
 */
export function transposeTableData(data: TableData): TableData {
  // Merge sub-labels into headers for transposed row labels
  const effectiveHeaders = data.columnHeaders.map((h, i) => {
    const sub = data.columnSubLabels?.[i]
    return sub ? `${h} (${sub})` : h
  })

  if (data.cells.length === 0 || data.cells[0].length === 0) {
    return {
      rowHeaders: effectiveHeaders,
      columnHeaders: data.rowHeaders,
      cells: [],
    }
  }

  const transposedCells = data.cells[0].map((_, colIdx) =>
    data.cells.map(row => row[colIdx])
  )

  // Transpose cellValues if present
  const transposedCellValues = data.cellValues && data.cellValues.length > 0 && data.cellValues[0].length > 0
    ? data.cellValues[0].map((_, colIdx) =>
        data.cellValues!.map(row => row[colIdx])
      )
    : undefined

  return {
    rowHeaders: effectiveHeaders,
    columnHeaders: data.rowHeaders,
    cells: transposedCells,
    cellValues: transposedCellValues,
    // When swapped: each row is a (size, method) combo, each column is a metric
    // metricDirections maps to columns now (was rows before)
    metricDirections: data.metricDirections,
    // Column groups and sub-labels don't apply when transposed
  }
}
