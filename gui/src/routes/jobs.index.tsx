import { useState, useMemo } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation } from 'convex/react'
import { api } from 'convex/_generated/api'
import { toast } from 'sonner'
import { PYTHON_API_URL } from '../lib/api-urls'
import {
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Trash2,
  MoreVertical,
  Sparkles,
  Search,
  Filter,
  Eye,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronDown,
  Zap,
  Cpu,
} from 'lucide-react'

import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Skeleton } from '../components/ui/skeleton'
import { Input } from '../components/ui/input'
import { Checkbox } from '../components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../components/ui/dropdown-menu'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../components/ui/popover'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table'
import { Card, CardContent } from '../components/ui/card'

export const Route = createFileRoute('/jobs/')({
  component: JobsPage,
})

// Status configuration
const STATUS_CONFIG = {
  pending: { icon: Clock, color: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30', label: 'Pending' },
  composing: { icon: Loader2, color: 'bg-purple-500/10 text-purple-500 border-purple-500/30', label: 'Composing', spin: true },
  composed: { icon: CheckCircle, color: 'bg-indigo-500/10 text-indigo-500 border-indigo-500/30', label: 'Composed' },
  running: { icon: Loader2, color: 'bg-blue-500/10 text-blue-500 border-blue-500/30', label: 'Running', spin: true },
  evaluating: { icon: Loader2, color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/30', label: 'Evaluating', spin: true },
  paused: { icon: Clock, color: 'bg-orange-500/10 text-orange-500 border-orange-500/30', label: 'Paused' },
  completed: { icon: CheckCircle, color: 'bg-green-500/10 text-green-500 border-green-500/30', label: 'Completed' },
  terminated: { icon: XCircle, color: 'bg-gray-500/10 text-gray-500 border-gray-500/30', label: 'Terminated' },
  failed: { icon: XCircle, color: 'bg-red-500/10 text-red-500 border-red-500/30', label: 'Failed' },
}

// Column definitions
type ColumnKey = 'status' | 'name' | 'method' | 'phases' | 'compute' | 'progress' | 'created' | 'actions'

const COLUMNS: { key: ColumnKey; label: string; sortable: boolean; hideable: boolean }[] = [
  { key: 'status', label: 'Status', sortable: true, hideable: false },
  { key: 'name', label: 'Name', sortable: true, hideable: false },
  { key: 'method', label: 'Method', sortable: true, hideable: true },
  { key: 'phases', label: 'Phases', sortable: false, hideable: true },
  { key: 'compute', label: 'Compute', sortable: true, hideable: true },
  { key: 'progress', label: 'Progress', sortable: true, hideable: true },
  { key: 'created', label: 'Created', sortable: true, hideable: true },
  { key: 'actions', label: '', sortable: false, hideable: false },
]

type SortDirection = 'asc' | 'desc' | null

function JobsPage() {
  const jobs = useQuery(api.jobs.list)
  const removeJob = useMutation(api.jobs.remove)

  // Search state
  const [searchQuery, setSearchQuery] = useState('')

  // Sort state
  const [sortColumn, setSortColumn] = useState<ColumnKey | null>('created')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  // Filter state
  const [statusFilters, setStatusFilters] = useState<string[]>([])
  const [methodFilters, setMethodFilters] = useState<string[]>([])

  // View state (hidden columns)
  const [hiddenColumns, setHiddenColumns] = useState<Set<ColumnKey>>(new Set())

  // Handle column sort
  const handleSort = (column: ColumnKey) => {
    if (sortColumn === column) {
      // Cycle through: asc -> desc -> null
      if (sortDirection === 'asc') {
        setSortDirection('desc')
      } else if (sortDirection === 'desc') {
        setSortColumn(null)
        setSortDirection(null)
      }
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

  // Filter and sort jobs
  const filteredJobs = useMemo(() => {
    if (!jobs) return []

    let result = [...jobs]

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(job =>
        job.name.toLowerCase().includes(query) ||
        (job.config.subject_profile?.query || '').toLowerCase().includes(query)
      )
    }

    // Status filter
    if (statusFilters.length > 0) {
      result = result.filter(job => statusFilters.includes(job.status))
    }

    // Method filter
    if (methodFilters.length > 0) {
      result = result.filter(job => {
        const method = job.method || 'cera'
        return methodFilters.includes(method)
      })
    }

    // Sort
    if (sortColumn && sortDirection) {
      result.sort((a, b) => {
        let aVal: any, bVal: any

        switch (sortColumn) {
          case 'status':
            aVal = a.status
            bVal = b.status
            break
          case 'name':
            aVal = a.name.toLowerCase()
            bVal = b.name.toLowerCase()
            break
          case 'method':
            aVal = a.method || 'cera'
            bVal = b.method || 'cera'
            break
          case 'compute':
            aVal = a.evaluationDevice?.type || ''
            bVal = b.evaluationDevice?.type || ''
            break
          case 'progress':
            aVal = a.progress
            bVal = b.progress
            break
          case 'created':
            aVal = a.createdAt
            bVal = b.createdAt
            break
          default:
            return 0
        }

        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    }

    return result
  }, [jobs, searchQuery, statusFilters, methodFilters, sortColumn, sortDirection])

  const handleDelete = async (jobId: string, jobName: string, jobDir?: string) => {
    try {
      if (jobDir) {
        const pythonApiUrl = PYTHON_API_URL
        try {
          await fetch(`${pythonApiUrl}/api/delete-job-dir`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ jobDir }),
          })
        } catch (e) {
          console.warn('Failed to delete job directory:', e)
        }
      }
      await removeJob({ id: jobId as any })
      toast.success(`Deleted "${jobName}"`)
    } catch (error) {
      toast.error('Failed to delete job')
    }
  }

  // Toggle filter
  const toggleStatusFilter = (status: string) => {
    setStatusFilters(prev =>
      prev.includes(status) ? prev.filter(s => s !== status) : [...prev, status]
    )
  }

  const toggleMethodFilter = (method: string) => {
    setMethodFilters(prev =>
      prev.includes(method) ? prev.filter(m => m !== method) : [...prev, method]
    )
  }

  // Toggle column visibility
  const toggleColumn = (column: ColumnKey) => {
    setHiddenColumns(prev => {
      const next = new Set(prev)
      if (next.has(column)) {
        next.delete(column)
      } else {
        next.add(column)
      }
      return next
    })
  }

  // Get visible columns
  const visibleColumns = COLUMNS.filter(col => !hiddenColumns.has(col.key))

  // Render sort icon
  const renderSortIcon = (column: ColumnKey) => {
    if (sortColumn !== column) {
      return <ArrowUpDown className="ml-1 h-3 w-3 opacity-50" />
    }
    if (sortDirection === 'asc') {
      return <ArrowUp className="ml-1 h-3 w-3" />
    }
    return <ArrowDown className="ml-1 h-3 w-3" />
  }

  // Active filter count
  const activeFilterCount = statusFilters.length + methodFilters.length

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Jobs</h1>
          <p className="text-muted-foreground">
            Monitor and manage your generation jobs
          </p>
        </div>
        <Button asChild>
          <Link to="/create-job" search={{ reset: true }}>
            <Sparkles className="mr-2 h-4 w-4" />
            New Job
          </Link>
        </Button>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search jobs..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Filter Dropdown */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="gap-2">
              <Filter className="h-4 w-4" />
              Filter
              {activeFilterCount > 0 && (
                <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                  {activeFilterCount}
                </Badge>
              )}
              <ChevronDown className="h-3 w-3 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-56 p-3">
            <div className="space-y-4">
              {/* Status filters */}
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Status</p>
                <div className="space-y-2">
                  {Object.entries(STATUS_CONFIG).map(([status, config]) => (
                    <label key={status} className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={statusFilters.includes(status)}
                        onCheckedChange={() => toggleStatusFilter(status)}
                      />
                      <span className="text-sm">{config.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Method filters */}
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Method</p>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                      checked={methodFilters.includes('cera')}
                      onCheckedChange={() => toggleMethodFilter('cera')}
                    />
                    <span className="text-sm">CERA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                      checked={methodFilters.includes('heuristic')}
                      onCheckedChange={() => toggleMethodFilter('heuristic')}
                    />
                    <span className="text-sm">Heuristic</span>
                  </label>
                </div>
              </div>

              {/* Clear filters */}
              {activeFilterCount > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full"
                  onClick={() => {
                    setStatusFilters([])
                    setMethodFilters([])
                  }}
                >
                  Clear all filters
                </Button>
              )}
            </div>
          </PopoverContent>
        </Popover>

        {/* View Dropdown */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="gap-2">
              <Eye className="h-4 w-4" />
              View
              <ChevronDown className="h-3 w-3 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="start" className="w-48 p-3">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Toggle columns</p>
            <div className="space-y-2">
              {COLUMNS.filter(col => col.hideable).map(col => (
                <label key={col.key} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox
                    checked={!hiddenColumns.has(col.key)}
                    onCheckedChange={() => toggleColumn(col.key)}
                  />
                  <span className="text-sm">{col.label}</span>
                </label>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {/* Data Table */}
      {jobs === undefined ? (
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      ) : filteredJobs.length === 0 ? (
        jobs.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Sparkles className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">No jobs yet</p>
              <p className="text-sm text-muted-foreground/70 mb-4">
                Create your first generation job to get started
              </p>
              <Button asChild>
                <Link to="/create-job">Create Job</Link>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Search className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">No matching jobs</p>
              <p className="text-sm text-muted-foreground/70 mb-4">
                Try adjusting your search or filters
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  setSearchQuery('')
                  setStatusFilters([])
                  setMethodFilters([])
                }}
              >
                Clear filters
              </Button>
            </CardContent>
          </Card>
        )
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                {visibleColumns.map(col => (
                  <TableHead key={col.key} className={col.key === 'actions' ? 'w-[50px]' : ''}>
                    {col.sortable ? (
                      <button
                        className="flex items-center hover:text-foreground transition-colors -ml-2 px-2 py-1 rounded"
                        onClick={() => handleSort(col.key)}
                      >
                        {col.label}
                        {renderSortIcon(col.key)}
                      </button>
                    ) : (
                      col.label
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredJobs.map(job => (
                <JobTableRow
                  key={job._id}
                  job={job}
                  visibleColumns={visibleColumns.map(c => c.key)}
                  onDelete={() => handleDelete(job._id, job.name, job.jobDir)}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* Footer stats */}
      {jobs && jobs.length > 0 && (
        <p className="text-sm text-muted-foreground">
          Showing {filteredJobs.length} of {jobs.length} jobs
        </p>
      )}
    </div>
  )
}

function JobTableRow({
  job,
  visibleColumns,
  onDelete,
}: {
  job: any
  visibleColumns: ColumnKey[]
  onDelete: () => void
}) {
  const status = STATUS_CONFIG[job.status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.pending
  const StatusIcon = status.icon
  const createdAt = new Date(job.createdAt).toLocaleDateString()
  const method = job.method || 'cera'

  // Determine phases - handle both CERA and Heuristic jobs
  const getPhases = () => {
    // Check if phases array exists
    if (job.phases && job.phases.length > 0) {
      return job.phases
    }

    // Heuristic jobs: check heuristicConfig
    if (method === 'heuristic' && job.heuristicConfig) {
      // Heuristic jobs do generation + evaluation
      return ['generation', 'evaluation']
    }

    // CERA jobs: infer from config
    const phases: string[] = []
    if (job.config.subject_profile) phases.push('composition')
    if (job.config.generation) phases.push('generation')
    if (job.evaluationConfig || job.datasetFile) phases.push('evaluation')

    // If only evaluation (external dataset)
    if (phases.length === 0 && job.datasetFile) {
      return ['evaluation']
    }

    return phases.length > 0 ? phases : ['evaluation']
  }

  const phases = getPhases()

  // Phase badge colors
  const phaseColors: Record<string, string> = {
    composition: 'bg-blue-500/10 text-blue-600 border-blue-500/30',
    generation: 'bg-orange-500/10 text-orange-600 border-orange-500/30',
    evaluation: 'bg-green-500/10 text-green-600 border-green-500/30',
  }

  // Phase short labels
  const phaseLabels: Record<string, string> = {
    composition: 'Comp',
    generation: 'Gen',
    evaluation: 'Eval',
  }

  return (
    <TableRow className="cursor-pointer" onClick={() => window.location.href = `/jobs/${job._id}`}>
      {/* Status */}
      {visibleColumns.includes('status') && (
        <TableCell>
          <div className="flex items-center gap-2">
            <div className={`rounded-full p-1.5 ${status.color}`}>
              <StatusIcon className={`h-3.5 w-3.5 ${status.spin ? 'animate-spin' : ''}`} />
            </div>
            <span className="text-sm font-medium hidden sm:inline">{status.label}</span>
          </div>
        </TableCell>
      )}

      {/* Name */}
      {visibleColumns.includes('name') && (
        <TableCell>
          <div className="min-w-0">
            <p className="font-medium truncate max-w-[200px]">{job.name}</p>
            <p className="text-xs text-muted-foreground truncate max-w-[200px]">
              {job.config.subject_profile?.query || (job.heuristicConfig ? 'Heuristic generation' : 'External dataset')}
            </p>
          </div>
        </TableCell>
      )}

      {/* Method */}
      {visibleColumns.includes('method') && (
        <TableCell>
          <Badge
            variant="outline"
            className={
              method === 'cera'
                ? 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/30'
                : method === 'real'
                  ? 'bg-green-500/10 text-green-600 border-green-500/30'
                  : 'bg-purple-500/10 text-purple-500 border-purple-500/30'
            }
          >
            {method === 'cera' ? 'CERA' : method === 'real' ? 'Real' : 'Heuristic'}
          </Badge>
        </TableCell>
      )}

      {/* Phases */}
      {visibleColumns.includes('phases') && (
        <TableCell>
          <div className="flex gap-1">
            {phases.map(phase => (
              <Badge
                key={phase}
                variant="outline"
                className={`text-[10px] px-1.5 ${phaseColors[phase] || ''}`}
              >
                {phaseLabels[phase] || phase}
              </Badge>
            ))}
          </div>
        </TableCell>
      )}

      {/* Compute (GPU/CPU) */}
      {visibleColumns.includes('compute') && (
        <TableCell>
          {job.evaluationDevice ? (
            job.evaluationDevice.type === 'GPU' ? (
              <Badge
                variant="outline"
                className="text-[10px] px-1.5 border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-gradient-to-r from-emerald-500/20 to-teal-500/20 animate-pulse"
                title={job.evaluationDevice.name || 'GPU'}
              >
                <Zap className="h-3 w-3 mr-0.5" />
                GPU
              </Badge>
            ) : (
              <Badge
                variant="outline"
                className="text-[10px] px-1.5 border-gray-400 text-gray-500 dark:text-gray-400 bg-gray-100/50 dark:bg-gray-800/50"
              >
                <Cpu className="h-3 w-3 mr-0.5" />
                CPU
              </Badge>
            )
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </TableCell>
      )}

      {/* Progress */}
      {visibleColumns.includes('progress') && (
        <TableCell>
          {['running', 'composing', 'evaluating'].includes(job.status) ? (
            <div className="w-24">
              <div className="flex items-center gap-2">
                <Progress value={job.progress} className="h-1.5 flex-1" />
                <span className="text-xs text-muted-foreground w-8">{job.progress}%</span>
              </div>
            </div>
          ) : job.status === 'completed' ? (
            <span className="text-xs text-green-600">100%</span>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </TableCell>
      )}

      {/* Created */}
      {visibleColumns.includes('created') && (
        <TableCell>
          <span className="text-sm text-muted-foreground">{createdAt}</span>
        </TableCell>
      )}

      {/* Actions */}
      {visibleColumns.includes('actions') && (
        <TableCell onClick={e => e.stopPropagation()}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem asChild>
                <Link to="/jobs/$jobId" params={{ jobId: job._id }}>
                  View Details
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem
                className="text-destructive"
                onClick={onDelete}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </TableCell>
      )}
    </TableRow>
  )
}
