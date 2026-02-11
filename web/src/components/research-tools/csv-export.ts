/** CSV export utility for Research Tools tables */

import type { TableData } from './types'

/**
 * Convert TableData to CSV string.
 * Handles column groups by adding a header row.
 */
export function tableDataToCsv(data: TableData): string {
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

  // Add column header row
  rows.push(['', ...data.columnHeaders])

  // Add data rows
  for (let i = 0; i < data.rowHeaders.length; i++) {
    rows.push([data.rowHeaders[i], ...(data.cells[i] || [])])
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
 */
export function downloadCsv(data: TableData, filename: string): void {
  const csv = tableDataToCsv(data)
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
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
  if (data.cells.length === 0 || data.cells[0].length === 0) {
    return {
      rowHeaders: data.columnHeaders,
      columnHeaders: data.rowHeaders,
      cells: [],
    }
  }

  return {
    rowHeaders: data.columnHeaders,
    columnHeaders: data.rowHeaders,
    cells: data.cells[0].map((_, colIdx) =>
      data.cells.map(row => row[colIdx])
    ),
    // Column groups don't apply when transposed
  }
}
