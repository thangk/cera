import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation } from 'convex/react'
import { api } from 'convex/_generated/api'
import { Id } from 'convex/_generated/dataModel'
import { toast } from 'sonner'
import {
  ArrowLeft,
  Database,
  Download,
  FileJson,
  FileSpreadsheet,
  FileCode,
  BarChart3,
  AlertCircle,
  Copy,
  ExternalLink,
} from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Skeleton } from '../components/ui/skeleton'
import { Progress } from '../components/ui/progress'
import { Separator } from '../components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'

export const Route = createFileRoute('/results/$id')({
  component: DatasetDetailPage,
})

function DatasetDetailPage() {
  const { id } = Route.useParams()
  const dataset = useQuery(api.datasets.get, { id: id as Id<'datasets'> })
  const job = useQuery(
    api.jobs.get,
    dataset ? { id: dataset.jobId } : 'skip'
  )

  if (dataset === undefined) {
    return (
      <div className="flex flex-col gap-6 p-6">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <Skeleton className="h-48 w-full" />
            <Skeleton className="h-64 w-full" />
          </div>
          <Skeleton className="h-96 w-full" />
        </div>
      </div>
    )
  }

  if (dataset === null) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <AlertCircle className="h-12 w-12 text-muted-foreground" />
        <p className="text-lg text-muted-foreground">Dataset not found</p>
        <Button asChild variant="outline">
          <Link to="/results">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Results
          </Link>
        </Button>
      </div>
    )
  }

  // Calculate overall quality score
  const overallScore = (
    (dataset.metrics.bertscore +
      dataset.metrics.distinct_1 +
      dataset.metrics.distinct_2 +
      (1 - dataset.metrics.self_bleu)) / 4 * 100
  )

  const handleCopyPath = () => {
    navigator.clipboard.writeText(dataset.outputPath)
    toast.success('Path copied to clipboard')
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link to="/results">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">{dataset.name}</h1>
              <p className="text-muted-foreground">{dataset.subject}</p>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleCopyPath}>
            <Copy className="mr-2 h-4 w-4" />
            Copy Path
          </Button>
          <Button>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Metrics Overview */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Quality Metrics (MDQA)
              </CardTitle>
              <CardDescription>
                Multi-Dimensional Quality Assessment results
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Overall score */}
              <div className="rounded-lg bg-muted/50 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Overall Quality Score</span>
                  <span className={`text-2xl font-bold ${
                    overallScore >= 70 ? 'text-green-500' :
                    overallScore >= 50 ? 'text-yellow-500' :
                    'text-red-500'
                  }`}>
                    {overallScore.toFixed(1)}%
                  </span>
                </div>
                <Progress value={overallScore} className="h-3" />
                <p className="text-xs text-muted-foreground mt-2">
                  Composite of BERTScore, Distinct-n, and Self-BLEU metrics
                </p>
              </div>

              {/* Individual metrics */}
              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  title="BERTScore"
                  value={dataset.metrics.bertscore}
                  description="Semantic similarity to reference texts"
                  higherIsBetter={true}
                />
                <MetricCard
                  title="Distinct-1"
                  value={dataset.metrics.distinct_1}
                  description="Unique unigram ratio"
                  higherIsBetter={true}
                />
                <MetricCard
                  title="Distinct-2"
                  value={dataset.metrics.distinct_2}
                  description="Unique bigram ratio"
                  higherIsBetter={true}
                />
                <MetricCard
                  title="Self-BLEU"
                  value={dataset.metrics.self_bleu}
                  description="Corpus diversity (lower is better)"
                  higherIsBetter={false}
                />
              </div>
            </CardContent>
          </Card>

          {/* Export Options */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="h-5 w-5" />
                Export Options
              </CardTitle>
              <CardDescription>
                Download your dataset in various formats
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="jsonl">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="jsonl">JSONL</TabsTrigger>
                  <TabsTrigger value="csv">CSV</TabsTrigger>
                  <TabsTrigger value="semeval">SemEval XML</TabsTrigger>
                </TabsList>

                <TabsContent value="jsonl" className="space-y-4">
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <FileJson className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">JSON Lines Format</span>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      One JSON object per line. Ideal for streaming and large datasets.
                    </p>
                    <pre className="text-xs bg-background rounded p-2 overflow-x-auto">
{`{"text": "Great food!", "aspects": [...], "sentiment": "positive"}
{"text": "Service was slow", "aspects": [...], "sentiment": "negative"}`}
                    </pre>
                  </div>
                  <Button className="w-full">
                    <FileJson className="mr-2 h-4 w-4" />
                    Download JSONL
                  </Button>
                </TabsContent>

                <TabsContent value="csv" className="space-y-4">
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <FileSpreadsheet className="h-4 w-4 text-green-500" />
                      <span className="font-medium">CSV Format</span>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      Comma-separated values. Compatible with Excel, Google Sheets.
                    </p>
                    <pre className="text-xs bg-background rounded p-2 overflow-x-auto">
{`text,aspects,sentiment,polarity
"Great food!","food:positive",positive,1
"Service was slow","service:negative",negative,-1`}
                    </pre>
                  </div>
                  <Button className="w-full">
                    <FileSpreadsheet className="mr-2 h-4 w-4" />
                    Download CSV
                  </Button>
                </TabsContent>

                <TabsContent value="semeval" className="space-y-4">
                  <div className="rounded-lg border bg-muted/30 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <FileCode className="h-4 w-4 text-orange-500" />
                      <span className="font-medium">SemEval XML Format</span>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">
                      Standard format for ABSA research and benchmarking.
                    </p>
                    <pre className="text-xs bg-background rounded p-2 overflow-x-auto">
{`<sentence id="1">
  <text>Great food!</text>
  <aspectTerms>
    <aspectTerm term="food" polarity="positive"/>
  </aspectTerms>
</sentence>`}
                    </pre>
                  </div>
                  <Button className="w-full">
                    <FileCode className="mr-2 h-4 w-4" />
                    Download SemEval XML
                  </Button>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Dataset Info */}
          <Card>
            <CardHeader>
              <CardTitle>Dataset Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Reviews</span>
                <Badge variant="secondary">{dataset.reviewCount}</Badge>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Category</span>
                <Badge variant="outline">{dataset.category}</Badge>
              </div>
              <Separator />
              <div className="flex justify-between">
                <span className="text-muted-foreground">Created</span>
                <span>{new Date(dataset.createdAt).toLocaleDateString()}</span>
              </div>
              <Separator />
              <div>
                <span className="text-muted-foreground">Output Path</span>
                <div className="flex items-center gap-2 mt-1">
                  <code className="text-xs bg-muted px-2 py-1 rounded truncate flex-1">
                    {dataset.outputPath}
                  </code>
                  <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleCopyPath}>
                    <Copy className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Source Job */}
          {job && (
            <Card>
              <CardHeader>
                <CardTitle>Source Job</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Job Name</span>
                  <span className="font-medium truncate max-w-[120px]">{job.name}</span>
                </div>
                <Separator />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model</span>
                  <span className="text-xs">{job.config.generation.model}</span>
                </div>
                <Separator />
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Batch Size</span>
                  <span>{job.config.generation.batch_size}</span>
                </div>
                <Separator />
                <Button variant="outline" className="w-full" asChild>
                  <Link to="/jobs/$jobId" params={{ jobId: job._id }}>
                    <ExternalLink className="mr-2 h-4 w-4" />
                    View Job Details
                  </Link>
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Quick Stats */}
          <Card>
            <CardHeader>
              <CardTitle>Quality Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Semantic Quality</span>
                  <span className={dataset.metrics.bertscore >= 0.7 ? 'text-green-500' : 'text-yellow-500'}>
                    {dataset.metrics.bertscore >= 0.7 ? 'High' : 'Medium'}
                  </span>
                </div>
                <Progress value={dataset.metrics.bertscore * 100} className="h-2" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Lexical Diversity</span>
                  <span className={dataset.metrics.distinct_2 >= 0.6 ? 'text-green-500' : 'text-yellow-500'}>
                    {dataset.metrics.distinct_2 >= 0.6 ? 'High' : 'Medium'}
                  </span>
                </div>
                <Progress value={dataset.metrics.distinct_2 * 100} className="h-2" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Corpus Diversity</span>
                  <span className={dataset.metrics.self_bleu <= 0.3 ? 'text-green-500' : 'text-yellow-500'}>
                    {dataset.metrics.self_bleu <= 0.3 ? 'High' : 'Medium'}
                  </span>
                </div>
                <Progress value={(1 - dataset.metrics.self_bleu) * 100} className="h-2" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function MetricCard({
  title,
  value,
  description,
  higherIsBetter,
}: {
  title: string
  value: number
  description: string
  higherIsBetter: boolean
}) {
  const percentage = value * 100
  const isGood = higherIsBetter ? percentage >= 60 : percentage <= 40

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">{title}</span>
        <span className={`text-lg font-bold ${isGood ? 'text-green-500' : 'text-yellow-500'}`}>
          {percentage.toFixed(1)}%
        </span>
      </div>
      <Progress value={higherIsBetter ? percentage : 100 - percentage} className="h-2 mb-2" />
      <p className="text-xs text-muted-foreground">{description}</p>
    </div>
  )
}
