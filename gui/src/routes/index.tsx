import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from 'convex/react'
import { api } from 'convex/_generated/api'
import {
  Sparkles,
  Activity,
  Database,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  ArrowRight,
} from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Skeleton } from '../components/ui/skeleton'

export const Route = createFileRoute('/')({
  component: Dashboard,
})

function Dashboard() {
  const jobs = useQuery(api.jobs.list)
  const datasets = useQuery(api.datasets.list)

  const recentJobs = jobs?.slice(0, 5) ?? []
  const completedCount = jobs?.filter(j => j.status === 'completed').length ?? 0
  const runningCount = jobs?.filter(j => j.status === 'running').length ?? 0
  const totalReviews = datasets?.reduce((sum, d) => sum + d.reviewCount, 0) ?? 0

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Overview of your synthetic ABSA dataset generation
          </p>
        </div>
        <Button asChild>
          <Link to="/create-job">
            <Sparkles className="mr-2 h-4 w-4" />
            New Job
          </Link>
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Jobs"
          value={jobs?.length ?? 0}
          description="Generation jobs created"
          icon={Activity}
          loading={jobs === undefined}
        />
        <StatsCard
          title="Completed"
          value={completedCount}
          description="Successfully finished"
          icon={CheckCircle}
          loading={jobs === undefined}
          className="text-green-500"
        />
        <StatsCard
          title="Running"
          value={runningCount}
          description="Currently processing"
          icon={Loader2}
          loading={jobs === undefined}
          className="text-blue-500"
        />
        <StatsCard
          title="Total Reviews"
          value={totalReviews.toLocaleString()}
          description="Generated reviews"
          icon={Database}
          loading={datasets === undefined}
        />
      </div>

      {/* Recent Jobs */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>Your latest generation jobs</CardDescription>
          </div>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/jobs">
              View all
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          {jobs === undefined ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : recentJobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Sparkles className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">No jobs yet</p>
              <p className="text-sm text-muted-foreground/70">
                Create your first generation job to get started
              </p>
              <Button className="mt-4" asChild>
                <Link to="/create-job">Create Job</Link>
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              {recentJobs.map(job => (
                <JobCard key={job._id} job={job} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

    </div>
  )
}

function StatsCard({
  title,
  value,
  description,
  icon: Icon,
  loading,
  className,
}: {
  title: string
  value: string | number
  description: string
  icon: React.ComponentType<{ className?: string }>
  loading?: boolean
  className?: string
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className={`h-4 w-4 text-muted-foreground ${className ?? ''}`} />
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className="text-2xl font-bold">{value}</div>
        )}
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  )
}

function JobCard({ job }: { job: any }) {
  const statusConfig = {
    pending: { icon: Clock, color: 'bg-yellow-500/10 text-yellow-500', label: 'Pending' },
    composing: { icon: Loader2, color: 'bg-purple-500/10 text-purple-500', label: 'Composing' },
    composed: { icon: CheckCircle, color: 'bg-indigo-500/10 text-indigo-500', label: 'Composed' },
    running: { icon: Loader2, color: 'bg-blue-500/10 text-blue-500', label: 'Running' },
    paused: { icon: Clock, color: 'bg-orange-500/10 text-orange-500', label: 'Paused' },
    completed: { icon: CheckCircle, color: 'bg-green-500/10 text-green-500', label: 'Completed' },
    terminated: { icon: XCircle, color: 'bg-gray-500/10 text-gray-500', label: 'Terminated' },
    failed: { icon: XCircle, color: 'bg-red-500/10 text-red-500', label: 'Failed' },
  }

  const status = statusConfig[job.status as keyof typeof statusConfig] || statusConfig.pending
  const StatusIcon = status.icon

  return (
    <Link
      to="/jobs/$jobId"
      params={{ jobId: job._id }}
      className="flex items-center gap-4 rounded-lg border p-4 hover:bg-muted/50 transition-colors"
    >
      <div className={`rounded-full p-2 ${status.color}`}>
        <StatusIcon className={`h-4 w-4 ${job.status === 'running' ? 'animate-spin' : ''}`} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="font-medium truncate">{job.name}</p>
          {job.config.generation ? (
            <Badge variant="outline" className="text-xs">
              {job.config.generation.count} reviews
            </Badge>
          ) : (
            <Badge variant="secondary" className="text-xs">
              Evaluation only
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground truncate">
          {job.config.subject_profile?.query ?? 'External dataset'}
        </p>
      </div>
      <div className="flex flex-col items-end gap-1">
        <Badge className={status.color}>{status.label}</Badge>
        {job.status === 'running' && (
          <div className="w-24">
            <Progress value={job.progress} className="h-1" />
          </div>
        )}
      </div>
    </Link>
  )
}

