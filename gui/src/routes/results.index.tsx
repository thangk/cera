import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation } from 'convex/react'
import { api } from 'convex/_generated/api'
import { toast } from 'sonner'
import {
  Database,
  Sparkles,
  Trash2,
  MoreVertical,
  Download,
  BarChart3,
  FileText,
} from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Skeleton } from '../components/ui/skeleton'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../components/ui/dropdown-menu'

export const Route = createFileRoute('/results/')({
  component: ResultsPage,
})

function ResultsPage() {
  const datasets = useQuery(api.datasets.list)
  const removeDataset = useMutation(api.datasets.remove)

  const handleDelete = async (datasetId: string, datasetName: string) => {
    try {
      await removeDataset({ id: datasetId as any })
      toast.success(`Deleted "${datasetName}"`)
    } catch (error) {
      toast.error('Failed to delete dataset')
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Results</h1>
          <p className="text-muted-foreground">
            Browse and export your generated datasets
          </p>
        </div>
        <Button asChild>
          <Link to="/create-job">
            <Sparkles className="mr-2 h-4 w-4" />
            New Dataset
          </Link>
        </Button>
      </div>

      {/* Dataset Grid */}
      {datasets === undefined ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Skeleton key={i} className="h-64 w-full" />
          ))}
        </div>
      ) : datasets.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Database className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-muted-foreground">No datasets yet</p>
            <p className="text-sm text-muted-foreground/70 mb-4">
              Generate your first dataset to see it here
            </p>
            <Button asChild>
              <Link to="/create-job">Create Dataset</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map(dataset => (
            <DatasetCard
              key={dataset._id}
              dataset={dataset}
              onDelete={() => handleDelete(dataset._id, dataset.name)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function DatasetCard({ dataset, onDelete }: { dataset: any; onDelete: () => void }) {
  const createdAt = new Date(dataset.createdAt).toLocaleDateString()

  // Calculate overall quality score
  const overallScore = (
    (dataset.metrics.bertscore +
      dataset.metrics.distinct_1 +
      dataset.metrics.distinct_2 +
      (1 - dataset.metrics.self_bleu)) / 4 * 100
  ).toFixed(1)

  return (
    <Card className="hover:bg-muted/30 transition-colors">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base">
                <Link
                  to="/results/$id"
                  params={{ id: dataset._id }}
                  className="hover:underline"
                >
                  {dataset.name}
                </Link>
              </CardTitle>
              <CardDescription>{dataset.subject}</CardDescription>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem asChild>
                <Link to="/results/$id" params={{ id: dataset._id }}>
                  <FileText className="mr-2 h-4 w-4" />
                  View Details
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Download className="mr-2 h-4 w-4" />
                Export JSONL
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
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Reviews</span>
          <Badge variant="secondary">{dataset.reviewCount}</Badge>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Category</span>
          <Badge variant="outline">{dataset.category}</Badge>
        </div>

        {/* Metrics summary */}
        <div className="rounded-lg bg-muted/50 p-3 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1">
              <BarChart3 className="h-3 w-3" />
              Quality Score
            </span>
            <span className={`font-semibold ${
              parseFloat(overallScore) >= 70 ? 'text-green-500' :
              parseFloat(overallScore) >= 50 ? 'text-yellow-500' :
              'text-red-500'
            }`}>
              {overallScore}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">BERTScore</span>
              <span>{(dataset.metrics.bertscore * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Distinct-2</span>
              <span>{(dataset.metrics.distinct_2 * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Distinct-1</span>
              <span>{(dataset.metrics.distinct_1 * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Self-BLEU</span>
              <span>{(dataset.metrics.self_bleu * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <span>Created {createdAt}</span>
          <Button variant="link" size="sm" className="h-auto p-0" asChild>
            <Link to="/results/$id" params={{ id: dataset._id }}>
              View Details
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
