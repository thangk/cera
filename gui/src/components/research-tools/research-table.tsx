import { useState, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ArrowUpDown, Download, Eye, Palette } from 'lucide-react'
import type { TableData, AxisOrientation } from './types'
import { transposeTableData, downloadCsv } from './csv-export'

type ComparisonScope = 'by-target' | 'all-targets'
type ViewOption = 'genllm' | 'sd'

interface ResearchTableProps {
  data: TableData
  title: string
  description?: string
  filename?: string
}

/**
 * Compute green background color for a cell based on its RANK within a comparison group.
 * Rank-based: worst = white (no color), each rank above gets a distinct green step.
 * Even very close values get visually distinct shades.
 */
function computeCellColor(
  value: number | null,
  groupValues: (number | null)[],
  direction: 'up' | 'down',
): string | undefined {
  if (value === null) return undefined

  const validValues = groupValues.filter((v): v is number => v !== null)
  if (validValues.length <= 1) return undefined // single value: no coloring needed

  // Sort to determine ranks (best first)
  const sorted = [...validValues].sort((a, b) => direction === 'up' ? b - a : a - b)

  // Find this value's rank (0 = best, n-1 = worst)
  const rank = sorted.indexOf(value)
  const worst = validValues.length - 1

  // Worst rank = no color (white), best = strongest green
  if (rank === worst) return undefined

  // Map rank to opacity: best (rank 0) = 0.40, second-worst = 0.15
  // Evenly spaced steps between ranks
  const opacity = 0.40 - (rank / worst) * 0.25
  return `rgba(34, 197, 94, ${opacity.toFixed(3)})`
}

/**
 * Build a color map for all cells based on comparison scope.
 * Returns a 2D array matching cells shape, with rgba strings or undefined.
 */
function buildColorMap(
  data: TableData,
  scope: ComparisonScope,
  orientation: AxisOrientation,
): (string | undefined)[][] {
  const { cellValues, metricDirections, columnGroups } = data
  if (!cellValues || !metricDirections || cellValues.length === 0) {
    return []
  }

  if (orientation === 'default') {
    // Default: rows = metrics, columns = (size × method)
    return cellValues.map((row, rowIdx) => {
      const direction = metricDirections[rowIdx] || 'up'

      if (scope === 'all-targets') {
        // Compare across ALL columns in this row
        return row.map(val => computeCellColor(val, row, direction))
      }

      // By-target: compare within each column group
      const colors: (string | undefined)[] = []
      let colOffset = 0
      const groups = columnGroups || [{ label: '', span: row.length }]
      for (const group of groups) {
        const groupSlice = row.slice(colOffset, colOffset + group.span)
        for (let i = 0; i < group.span; i++) {
          colors.push(computeCellColor(row[colOffset + i], groupSlice, direction))
        }
        colOffset += group.span
      }
      return colors
    })
  }

  // Swapped: rows = (size × method), columns = metrics
  // metricDirections maps to columns (each column is a metric)
  return cellValues.map(row => {
    return row.map((val, colIdx) => {
      const direction = metricDirections[colIdx] || 'up'

      if (scope === 'all-targets') {
        // Compare this metric across ALL rows
        const columnValues = cellValues.map(r => r[colIdx])
        return computeCellColor(val, columnValues, direction)
      }

      // By-target in swapped mode: we don't have column groups, so fall back to all
      const columnValues = cellValues.map(r => r[colIdx])
      return computeCellColor(val, columnValues, direction)
    })
  })
}

/**
 * Strip "± X.XXXX" from a cell string to hide standard deviation.
 */
function stripSd(cell: string): string {
  return cell.replace(/\s*\u00B1\s*\S+/, '')
}

export function ResearchTable({ data, title, description, filename }: ResearchTableProps) {
  const [orientation, setOrientation] = useState<AxisOrientation>('default')
  const [comparisonScope, setComparisonScope] = useState<ComparisonScope>('by-target')
  const [viewOptions, setViewOptions] = useState<Set<ViewOption>>(new Set(['sd']))

  const showGenLlm = viewOptions.has('genllm')
  const showSd = viewOptions.has('sd')

  const toggleViewOption = (option: ViewOption) => {
    setViewOptions(prev => {
      const next = new Set(prev)
      if (next.has(option)) next.delete(option)
      else next.add(option)
      return next
    })
  }

  const displayData = useMemo(() => {
    return orientation === 'swapped' ? transposeTableData(data) : data
  }, [data, orientation])

  const colorMap = useMemo(() => {
    return buildColorMap(displayData, comparisonScope, orientation)
  }, [displayData, comparisonScope, orientation])

  // Compute column indices that are the last column of each group (for vertical borders)
  const groupBorderCols = useMemo(() => {
    const set = new Set<number>()
    if (displayData.columnGroups && displayData.columnGroups.length > 1) {
      let offset = 0
      for (let i = 0; i < displayData.columnGroups.length - 1; i++) {
        offset += displayData.columnGroups[i].span
        set.add(offset - 1) // last column index of this group
      }
    }
    return set
  }, [displayData.columnGroups])

  const handleExport = () => {
    downloadCsv(displayData, filename || title.toLowerCase().replace(/\s+/g, '-'), {
      showSubLabels: showGenLlm,
      showSd,
    })
  }

  if (data.rowHeaders.length === 0 || data.columnHeaders.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="flex items-center gap-2">
            {/* Comparison scope */}
            {displayData.cellValues && (
              <Select value={comparisonScope} onValueChange={(v) => setComparisonScope(v as ComparisonScope)}>
                <SelectTrigger className="h-8 w-[140px] text-xs">
                  <Palette className="h-3.5 w-3.5 mr-1" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="by-target">By Target</SelectItem>
                  <SelectItem value="all-targets">All Targets</SelectItem>
                </SelectContent>
              </Select>
            )}

            {/* View options */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8">
                  <Eye className="h-3.5 w-3.5 mr-1.5" />
                  View
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel className="text-xs">Show / Hide</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuCheckboxItem
                  checked={showGenLlm}
                  onCheckedChange={() => toggleViewOption('genllm')}
                >
                  GenLLM Name
                </DropdownMenuCheckboxItem>
                <DropdownMenuCheckboxItem
                  checked={showSd}
                  onCheckedChange={() => toggleViewOption('sd')}
                >
                  SD Values
                </DropdownMenuCheckboxItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <Button
              variant="outline"
              size="sm"
              className="h-8"
              onClick={() => setOrientation(o => o === 'default' ? 'swapped' : 'default')}
            >
              <ArrowUpDown className="h-3.5 w-3.5 mr-1.5" />
              Swap Axes
            </Button>
            <Button variant="outline" size="sm" className="h-8" onClick={handleExport}>
              <Download className="h-3.5 w-3.5 mr-1.5" />
              CSV
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative w-full overflow-auto">
          <Table className="w-auto">
            <TableHeader>
              {/* Column group headers */}
              {displayData.columnGroups && displayData.columnGroups.length > 0 && (
                <TableRow>
                  <TableHead className="border-r px-2" />
                  {displayData.columnGroups.map((group, i) => (
                    <TableHead
                      key={i}
                      colSpan={group.span}
                      className="text-center font-semibold border-r last:border-r-0 px-2"
                    >
                      {group.label}
                    </TableHead>
                  ))}
                </TableRow>
              )}
              {/* Column headers */}
              <TableRow>
                <TableHead className="border-r font-semibold whitespace-nowrap">Metric</TableHead>
                {displayData.columnHeaders.map((header, i) => (
                  <TableHead key={i} className={`text-center whitespace-nowrap px-2${groupBorderCols.has(i) ? ' border-r' : ''}`}>
                    <div>{header}</div>
                    {showGenLlm && displayData.columnSubLabels?.[i] && (
                      <div className="text-[10px] text-muted-foreground font-normal leading-tight mt-0.5">
                        {displayData.columnSubLabels[i]}
                      </div>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayData.rowHeaders.map((rowHeader, rowIdx) => (
                <TableRow key={rowIdx}>
                  <TableCell className="border-r font-medium whitespace-nowrap">
                    {rowHeader}
                  </TableCell>
                  {displayData.cells[rowIdx]?.map((cell, colIdx) => {
                    const bg = colorMap[rowIdx]?.[colIdx]
                    const displayCell = cell
                      ? (showSd ? cell : stripSd(cell))
                      : '---'
                    return (
                      <TableCell
                        key={colIdx}
                        className={`text-center tabular-nums text-sm whitespace-nowrap px-2${groupBorderCols.has(colIdx) ? ' border-r' : ''}`}
                        style={bg ? { backgroundColor: bg } : undefined}
                      >
                        {displayCell}
                      </TableCell>
                    )
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
