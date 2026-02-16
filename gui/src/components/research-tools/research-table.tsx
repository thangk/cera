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
import { ArrowUpDown, Download } from 'lucide-react'
import type { TableData, AxisOrientation } from './types'
import { transposeTableData, downloadCsv } from './csv-export'

interface ResearchTableProps {
  data: TableData
  title: string
  description?: string
  filename?: string
}

export function ResearchTable({ data, title, description, filename }: ResearchTableProps) {
  const [orientation, setOrientation] = useState<AxisOrientation>('default')

  const displayData = useMemo(() => {
    return orientation === 'swapped' ? transposeTableData(data) : data
  }, [data, orientation])

  const handleExport = () => {
    downloadCsv(displayData, filename || title.toLowerCase().replace(/\s+/g, '-'))
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
            <Button
              variant="outline"
              size="sm"
              onClick={() => setOrientation(o => o === 'default' ? 'swapped' : 'default')}
            >
              <ArrowUpDown className="h-4 w-4 mr-1.5" />
              Swap Axes
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-1.5" />
              CSV
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative w-full overflow-auto">
          <Table>
            <TableHeader>
              {/* Column group headers */}
              {displayData.columnGroups && displayData.columnGroups.length > 0 && (
                <TableRow>
                  <TableHead className="border-r" />
                  {displayData.columnGroups.map((group, i) => (
                    <TableHead
                      key={i}
                      colSpan={group.span}
                      className="text-center font-semibold border-r last:border-r-0"
                    >
                      {group.label}
                    </TableHead>
                  ))}
                </TableRow>
              )}
              {/* Column headers */}
              <TableRow>
                <TableHead className="border-r font-semibold min-w-[120px]">Metric</TableHead>
                {displayData.columnHeaders.map((header, i) => (
                  <TableHead key={i} className="text-center min-w-[120px]">
                    <div>{header}</div>
                    {displayData.columnSubLabels?.[i] && (
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
                  {displayData.cells[rowIdx]?.map((cell, colIdx) => (
                    <TableCell key={colIdx} className="text-center tabular-nums text-sm">
                      {cell || '---'}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}
