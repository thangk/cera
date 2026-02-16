import { createFileRoute, useNavigate, useSearch } from '@tanstack/react-router'
import { useQuery } from 'convex/react'
import { api } from '../../convex/_generated/api'
import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Progress } from '@/components/ui/progress'
import { FileDropZone } from '@/components/ui/file-drop-zone'
import {
  Eye, Upload, FileText, BarChart3, Users, Loader2,
  ArrowUpDown, Filter, ChevronDown, ChevronRight, ChevronLeft,
  CheckCircle, XCircle, AlertCircle, Info, ClipboardCheck, Search, X,
  Plus, Trash2, RotateCcw, Save, Edit2, Coins,
} from 'lucide-react'
import { toast } from 'sonner'
import { z } from 'zod'
import { PYTHON_API_URL } from '../lib/api-urls'
import { useOpenRouterModels } from '../hooks/use-openrouter-models'

// Valid tool values for URL state
const toolValues = ['dataset', 'mav', 'conformity', 'metrics', 'tokens', 'domain-patterns'] as const
type ToolValue = typeof toolValues[number]

const searchSchema = z.object({
  tool: z.enum(toolValues).optional().catch('dataset'),
})

export const Route = createFileRoute('/viewer-tools')({
  component: ViewerTools,
  validateSearch: searchSchema,
})

// Types
interface Opinion {
  target: string
  category: string
  polarity: string
  from: number
  to: number
}

interface Sentence {
  id: string
  text: string
  opinions: Opinion[]
}

interface Review {
  id: string
  sentences: Sentence[]
  metadata?: {
    assigned_polarity?: string | Record<string, number>
    age?: number
    sex?: string
  }
}

/** Extract polarity string from metadata.assigned_polarity which may be a string or distribution object */
function resolvePolarity(review: Review): string {
  const raw = review.metadata?.assigned_polarity
  if (!raw) return inferPolarity(review)
  if (typeof raw === 'string') return raw.toLowerCase()
  // It's a distribution object like {positive: 65, neutral: 15, negative: 20}
  // Return the dominant polarity
  const entries = Object.entries(raw)
  if (entries.length === 0) return inferPolarity(review)
  entries.sort(([, a], [, b]) => b - a)
  return entries[0][0].toLowerCase()
}

interface TargetInfo {
  size: number
  datasetFiles: string[]
  hasMetrics: boolean
  hasConformity: boolean
}

interface JobInfo {
  dirName: string
  path: string
  jobId: string
  jobName: string
  hasDataset: boolean
  hasMavs: boolean
  hasMetrics: boolean
  hasConformity: boolean
  hasTokens: boolean
  datasetFiles: string[]
  mavModels: string[]
  targets: TargetInfo[]
}

interface MavModelData {
  understanding?: string
  queries?: { model: string; subject: string; queries: string[]; count: number }
  answers?: { model: string; subject: string; answers: Array<{ query_id: string; response: string; confidence: string }> }
}

interface MavReport {
  config?: { models: string[]; consensus_method: string; answer_similarity_threshold: number }
  summary?: { total_queries_generated: number; queries_after_dedup: number; queries_with_consensus: number; consensus_rate: number }
  per_query_consensus?: Array<{
    query_id: string
    query: string
    consensus_reached: boolean
    consensus_answer?: string
    answers: Array<{ model: string; response: string; confidence: string }>
    agreeing_models: string[]
    agreement_count: number
    pairwise_similarities?: Record<string, number>
  }>
}

interface MetricEntry {
  metric: string
  score: number
  description: string
}

// Polarity colors (use lowercase keys, lookup should normalize case)
const POLARITY_COLORS: Record<string, string> = {
  positive: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  negative: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  neutral: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
  conflict: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
}

// Helper to get polarity color (case-insensitive)
const getPolarityColor = (polarity: string): string => {
  return POLARITY_COLORS[polarity.toLowerCase()] || ''
}

const POLARITY_HIGHLIGHT: Record<string, string> = {
  positive: 'bg-green-200/50 dark:bg-green-800/30 underline decoration-green-500',
  negative: 'bg-red-200/50 dark:bg-red-800/30 underline decoration-red-500',
  neutral: 'bg-amber-200/50 dark:bg-amber-800/30 underline decoration-amber-500',
  conflict: 'bg-purple-200/50 dark:bg-purple-800/30 underline decoration-purple-500',
}

// Helper to get polarity highlight (case-insensitive)
const getPolarityHighlight = (polarity: string): string => {
  return POLARITY_HIGHLIGHT[polarity.toLowerCase()] || ''
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
  medium: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
  low: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
}

function ViewerTools() {
  const navigate = useNavigate()
  const { tool } = useSearch({ from: '/viewer-tools' })
  const currentTool = tool || 'dataset'

  const handleToolChange = (value: string) => {
    navigate({
      to: '/viewer-tools',
      search: { tool: value as ToolValue },
      replace: true,
    })
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">Viewer Tools</h1>
        <p className="text-muted-foreground">View and analyze datasets, MAV reports, conformity reports, evaluation metrics, and token usage</p>
      </div>

      <Tabs value={currentTool} onValueChange={handleToolChange} className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="dataset" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Dataset Viewer
          </TabsTrigger>
          <TabsTrigger value="mav" className="flex items-center gap-2">
            <Users className="h-4 w-4" />
            MAV Reports
          </TabsTrigger>
          <TabsTrigger value="conformity" className="flex items-center gap-2">
            <ClipboardCheck className="h-4 w-4" />
            Conformity
          </TabsTrigger>
          <TabsTrigger value="metrics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Metrics Viewer
          </TabsTrigger>
          <TabsTrigger value="tokens" className="flex items-center gap-2">
            <Coins className="h-4 w-4" />
            Tokens
          </TabsTrigger>
          <TabsTrigger value="domain-patterns" className="flex items-center gap-2">
            <Filter className="h-4 w-4" />
            Domain Patterns
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dataset" className="mt-6">
          <DatasetViewer />
        </TabsContent>
        <TabsContent value="mav" className="mt-6">
          <MavReportsViewer />
        </TabsContent>
        <TabsContent value="conformity" className="mt-6">
          <ConformityViewer />
        </TabsContent>
        <TabsContent value="metrics" className="mt-6">
          <MetricsViewer />
        </TabsContent>
        <TabsContent value="tokens" className="mt-6">
          <TokensViewer />
        </TabsContent>
        <TabsContent value="domain-patterns" className="mt-6">
          <DomainPatternsManager />
        </TabsContent>
      </Tabs>
    </div>
  )
}

// ========================================
// Dataset Viewer Tab
// ========================================

const DATASET_VIEWER_STORAGE_KEY = 'cera-dataset-viewer-state'

function DatasetViewer() {
  const settings = useQuery(api.settings.get)
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [selectedJob, setSelectedJob] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<string>('')
  const [reviews, setReviews] = useState<Review[]>([])
  const [loading, setLoading] = useState(false)
  const [jobsLoading, setJobsLoading] = useState(false)
  const [polarityFilter, setPolarityFilter] = useState<string>('all')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('id')
  const [sourceMode, setSourceMode] = useState<'job' | 'import'>('job')
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [reviewsPerPage, setReviewsPerPage] = useState(25)
  const [importedFileName, setImportedFileName] = useState<string>('')

  // Restore state from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(DATASET_VIEWER_STORAGE_KEY)
      if (saved) {
        const state = JSON.parse(saved)
        if (state.sourceMode) setSourceMode(state.sourceMode)
        if (state.importedFileName) setImportedFileName(state.importedFileName)
        if (state.reviews && state.reviews.length > 0) setReviews(state.reviews)
      }
    } catch {
      // Ignore parse errors
    }
  }, [])

  // Save imported file state to localStorage
  useEffect(() => {
    if (sourceMode === 'import' && reviews.length > 0 && importedFileName) {
      localStorage.setItem(DATASET_VIEWER_STORAGE_KEY, JSON.stringify({
        sourceMode,
        importedFileName,
        reviews,
      }))
    }
  }, [sourceMode, reviews, importedFileName])

  const closeImportedFile = () => {
    setReviews([])
    setImportedFileName('')
    localStorage.removeItem(DATASET_VIEWER_STORAGE_KEY)
  }

  const fetchJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const jobsDir = settings?.jobsDirectory || './jobs'
      const res = await fetch(`${PYTHON_API_URL}/api/jobs-list?jobs_directory=${encodeURIComponent(jobsDir)}`)
      const data = await res.json()
      const jobsList = data.jobs || []
      setJobs(jobsList)
      // Auto-select latest job with dataset if available
      const jobsWithDataset = jobsList.filter((j: JobInfo) => j.hasDataset)
      if (jobsWithDataset.length > 0 && !selectedJob) {
        setSelectedJob(jobsWithDataset[0].path)
      }
    } catch (err) {
      toast.error('Failed to load jobs list')
    } finally {
      setJobsLoading(false)
    }
  }, [settings?.jobsDirectory, selectedJob])

  // Auto-load jobs on initial mount when settings are available
  useEffect(() => {
    if (settings?.jobsDirectory && jobs.length === 0) {
      fetchJobs()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings?.jobsDirectory]) // Only run on mount and when settings change

  const loadDataset = async (jobDir: string, filename: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/read-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobDir, filename }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setReviews(data.reviews || [])
      toast.success(`Loaded ${data.count} reviews (${data.format.toUpperCase()})`)
    } catch (err) {
      toast.error('Failed to load dataset')
    } finally {
      setLoading(false)
    }
  }

  const processFile = async (file: File) => {
    setLoading(true)
    setCurrentPage(1) // Reset to first page on new import
    try {
      const text = await file.text()
      let parsedReviews: Review[] = []

      if (file.name.endsWith('.jsonl')) {
        parsedReviews = text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l))
      } else if (file.name.endsWith('.csv')) {
        // Parse CSV client-side
        const lines = text.split('\n')
        const headers = lines[0].split(',')
        const reviewMap: Record<string, Review> = {}
        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue
          const values = lines[i].split(',')
          const row: Record<string, string> = {}
          headers.forEach((h, idx) => row[h.trim()] = values[idx]?.trim() || '')
          const rid = row.review_id || '0'
          if (!reviewMap[rid]) reviewMap[rid] = { id: rid, sentences: [], metadata: {} }
          const sid = row.sentence_id || `${rid}:0`
          let sent = reviewMap[rid].sentences.find(s => s.id === sid)
          if (!sent) {
            sent = { id: sid, text: row.text || '', opinions: [] }
            reviewMap[rid].sentences.push(sent)
          }
          sent.opinions.push({
            target: row.target || 'NULL',
            category: row.category || '',
            polarity: row.polarity || '',
            from: parseInt(row.from || '0'),
            to: parseInt(row.to || '0'),
          })
        }
        parsedReviews = Object.values(reviewMap)
      } else if (file.name.endsWith('.xml')) {
        const parser = new DOMParser()
        const doc = parser.parseFromString(text, 'text/xml')
        // Support multiple SemEval XML formats: <Review>, <review>, or sentences directly
        let reviewElems = doc.querySelectorAll('Review')
        if (reviewElems.length === 0) {
          reviewElems = doc.querySelectorAll('review')
        }

        if (reviewElems.length > 0) {
          // Standard format with Review elements
          reviewElems.forEach(revEl => {
            const rid = revEl.getAttribute('rid') || revEl.getAttribute('id') || String(parsedReviews.length + 1)
            const sentences: Sentence[] = []
            // Support both <sentence> and <Sentence>
            const sentenceElems = revEl.querySelectorAll('sentence, Sentence')
            sentenceElems.forEach(sentEl => {
              const sid = sentEl.getAttribute('id') || ''
              const textEl = sentEl.querySelector('text, Text')
              const opinions: Opinion[] = []
              // Support both <Opinion> and <opinion>
              sentEl.querySelectorAll('Opinion, opinion, Opinions > Opinion').forEach(opEl => {
                opinions.push({
                  target: opEl.getAttribute('target') || 'NULL',
                  category: opEl.getAttribute('category') || '',
                  polarity: opEl.getAttribute('polarity') || '',
                  // Support both standard (from/to) and hotel format (target_from/target_to)
                  from: parseInt(opEl.getAttribute('from') || opEl.getAttribute('target_from') || '0'),
                  to: parseInt(opEl.getAttribute('to') || opEl.getAttribute('target_to') || '0'),
                })
              })
              sentences.push({ id: sid, text: textEl?.textContent || '', opinions })
            })
            parsedReviews.push({ id: rid, sentences, metadata: {} })
          })
        } else {
          // Fallback: sentences directly under root (SemEval-14 format)
          const sentenceElems = doc.querySelectorAll('sentence, Sentence')
          let currentReviewId = 0
          sentenceElems.forEach(sentEl => {
            currentReviewId++
            const sid = sentEl.getAttribute('id') || String(currentReviewId)
            const textEl = sentEl.querySelector('text, Text')
            const opinions: Opinion[] = []
            sentEl.querySelectorAll('Opinion, opinion, aspectTerm, aspectTerms > aspectTerm').forEach(opEl => {
              opinions.push({
                target: opEl.getAttribute('target') || opEl.getAttribute('term') || 'NULL',
                category: opEl.getAttribute('category') || opEl.getAttribute('aspect') || '',
                polarity: opEl.getAttribute('polarity') || '',
                // Support both standard (from/to) and hotel format (target_from/target_to)
                from: parseInt(opEl.getAttribute('from') || opEl.getAttribute('target_from') || '0'),
                to: parseInt(opEl.getAttribute('to') || opEl.getAttribute('target_to') || '0'),
              })
            })
            // Each sentence is a separate review in this format
            parsedReviews.push({
              id: String(currentReviewId),
              sentences: [{ id: sid, text: textEl?.textContent || '', opinions }],
              metadata: {}
            })
          })
        }
      }

      setReviews(parsedReviews)
      setImportedFileName(file.name)
      toast.success(`Imported ${parsedReviews.length} reviews from ${file.name}`)
    } catch (err) {
      console.error('Parse error:', err)
      toast.error('Failed to parse file')
    } finally {
      setLoading(false)
    }
  }

  // Get unique categories from reviews
  const allCategories = Array.from(new Set(
    reviews.flatMap(r => r.sentences.flatMap(s => s.opinions.map(o => o.category)))
  )).filter(Boolean).sort()

  // Get polarity counts (normalized to lowercase)
  const polarityCounts = reviews.reduce((acc, r) => {
    const pol = resolvePolarity(r)
    acc[pol] = (acc[pol] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  // Calculate totals for stats
  const totalSentences = reviews.reduce((sum, r) => sum + r.sentences.length, 0)
  const totalOpinions = reviews.reduce((sum, r) =>
    sum + r.sentences.reduce((sSum, s) => sSum + s.opinions.length, 0), 0)

  // Average sentences per review
  const avgSentencesPerReview = reviews.length > 0
    ? (totalSentences / reviews.length).toFixed(1)
    : '0'

  // Sentence-level polarity counts (from all opinions)
  const sentencePolarityCounts = reviews.reduce((acc, r) => {
    r.sentences.forEach(s => {
      s.opinions.forEach(o => {
        const pol = o.polarity?.toLowerCase() || 'neutral'
        acc[pol] = (acc[pol] || 0) + 1
      })
    })
    return acc
  }, {} as Record<string, number>)

  // Infer domain from categories
  const inferDomain = (categories: string[]): { value: string; confidence: number } => {
    if (!categories.length) return { value: 'General', confidence: 0.3 }
    const categoryStr = categories.join(' ').toUpperCase()

    const restaurantKeywords = ['FOOD', 'SERVICE', 'AMBIANCE', 'PRICE', 'DRINKS', 'MENU', 'RESTAURANT']
    const laptopKeywords = ['LAPTOP', 'DISPLAY', 'SCREEN', 'KEYBOARD', 'BATTERY', 'SUPPORT', 'OS', 'SOFTWARE', 'HARDWARE', 'MEMORY', 'GRAPHICS', 'PORTS', 'PERFORMANCE', 'DESIGN', 'PORTABILITY']
    const hotelKeywords = ['ROOM', 'ROOMS', 'STAFF', 'CLEANLINESS', 'VALUE', 'AMENITIES', 'HOTEL', 'LOBBY', 'BREAKFAST', 'BED', 'BATHROOM', 'POOL', 'GYM', 'RECEPTION', 'HOUSEKEEPING', 'FACILITIES', 'LOCATION', 'CHECK-IN']

    const scores: Record<string, number> = {
      Restaurant: restaurantKeywords.filter(kw => categoryStr.includes(kw)).length,
      Laptop: laptopKeywords.filter(kw => categoryStr.includes(kw)).length,
      Hotel: hotelKeywords.filter(kw => categoryStr.includes(kw)).length,
    }

    const maxScore = Math.max(...Object.values(scores))
    if (maxScore === 0) return { value: 'General', confidence: 0.3 }

    const domain = Object.entries(scores).find(([_, s]) => s === maxScore)?.[0] || 'General'
    return { value: domain, confidence: Math.min(1, maxScore / 5) }
  }

  // Infer region from text patterns
  const inferRegion = (reviewsToCheck: Review[]): { value: string | null; confidence: number } => {
    const allText = reviewsToCheck.flatMap(r => r.sentences.map(s => s.text)).join(' ')
    const allTextLower = allText.toLowerCase()

    const hasPound = allText.includes('£')
    const hasEuro = allText.includes('€')
    const hasDollar = allText.includes('$')

    const britishSpellings = ['colour', 'favour', 'honour', 'flavour', 'centre', 'theatre', 'metre']
    const americanSpellings = ['color', 'favor', 'honor', 'flavor', 'center', 'theater', 'meter']

    const britishCount = britishSpellings.filter(w => allTextLower.includes(w)).length
    const americanCount = americanSpellings.filter(w => allTextLower.includes(w)).length

    const scores: Record<string, number> = {
      'UK': (hasPound ? 5 : 0) + britishCount * 2,
      'Europe': hasEuro ? 5 : 0,
      'US/Canada': (hasDollar ? 2 : 0) + americanCount * 2,
    }

    const maxScore = Math.max(...Object.values(scores))
    if (maxScore < 2) return { value: null, confidence: 0 }

    const region = Object.entries(scores).find(([_, s]) => s === maxScore)?.[0] || null
    const confidence = Math.min(1, maxScore / 10)
    return { value: confidence >= 0.5 ? region : null, confidence }
  }

  const inferredDomain = inferDomain(allCategories)
  const inferredRegion = inferRegion(reviews)

  // Filter and sort reviews
  const filteredReviews = reviews
    .filter(r => {
      if (polarityFilter !== 'all') {
        const pol = resolvePolarity(r)
        if (pol !== polarityFilter) return false
      }
      if (categoryFilter !== 'all') {
        const hasCategory = r.sentences.some(s => s.opinions.some(o => o.category === categoryFilter))
        if (!hasCategory) return false
      }
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        const textMatch = r.sentences.some(s => s.text.toLowerCase().includes(query))
        const categoryMatch = r.sentences.some(s => s.opinions.some(o => o.category.toLowerCase().includes(query)))
        const targetMatch = r.sentences.some(s => s.opinions.some(o => o.target.toLowerCase().includes(query)))
        if (!textMatch && !categoryMatch && !targetMatch) return false
      }
      return true
    })
    .sort((a, b) => {
      if (sortBy === 'polarity') {
        const polA = resolvePolarity(a)
        const polB = resolvePolarity(b)
        return polA.localeCompare(polB)
      }
      return parseInt(a.id) - parseInt(b.id)
    })

  // Pagination
  const totalPages = Math.ceil(filteredReviews.length / reviewsPerPage)
  const paginatedReviews = filteredReviews.slice(
    (currentPage - 1) * reviewsPerPage,
    currentPage * reviewsPerPage
  )

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [polarityFilter, categoryFilter, searchQuery])

  return (
    <div className="space-y-4">
      {/* Source Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Data Source</CardTitle>
          <CardDescription>Select a job or import a dataset file</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button
              variant={sourceMode === 'job' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSourceMode('job')}
            >
              <FileText className="mr-2 h-4 w-4" />
              From Job
            </Button>
            <Button
              variant={sourceMode === 'import' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSourceMode('import')}
            >
              <Upload className="mr-2 h-4 w-4" />
              Import File
            </Button>
            {sourceMode === 'job' && (
              <Button
                variant="ghost"
                size="sm"
                onClick={fetchJobs}
                disabled={jobsLoading}
              >
                {jobsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Refresh Jobs'}
              </Button>
            )}
          </div>

          {sourceMode === 'job' && (
            <div className="flex gap-3">
              <div className="flex-1 space-y-2">
                <Label>Job</Label>
                <Select
                  value={selectedJob}
                  onValueChange={(val) => {
                    setSelectedJob(val)
                    setSelectedFile('')
                    setReviews([])
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={jobsLoading ? 'Loading...' : 'Select a job'} />
                  </SelectTrigger>
                  <SelectContent>
                    {jobs.filter(j => j.hasDataset).map(job => (
                      <SelectItem key={job.dirName} value={job.path}>
                        {job.jobName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {selectedJob && (
                <div className="flex-1 space-y-2">
                  <Label>Dataset File</Label>
                  <Select
                    value={selectedFile}
                    onValueChange={(val) => {
                      setSelectedFile(val)
                      loadDataset(selectedJob, val)
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select file" />
                    </SelectTrigger>
                    <SelectContent>
                      {(() => {
                        const job = jobs.find(j => j.path === selectedJob)
                        if (!job) return null
                        if (job.targets && job.targets.length > 0) {
                          return job.targets.map(t => (
                            <SelectGroup key={t.size}>
                              <SelectLabel className="text-xs font-semibold">{t.size} sentences</SelectLabel>
                              {t.datasetFiles.map(f => (
                                <SelectItem key={f} value={f}>
                                  {f.split('/').slice(1).join('/')}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          ))
                        }
                        return job.datasetFiles.map(f => (
                          <SelectItem key={f} value={f}>{f}</SelectItem>
                        ))
                      })()}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          )}

          {sourceMode === 'import' && (
            importedFileName && reviews.length > 0 ? (
              <div className="flex items-center justify-between p-4 border rounded-lg bg-muted/30">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{importedFileName}</p>
                    <p className="text-sm text-muted-foreground">{reviews.length} reviews loaded</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <FileDropZone
                    onFile={processFile}
                    placeholder="Load different file"
                    className="py-2 px-4 text-xs"
                  />
                  <Button variant="outline" size="sm" onClick={closeImportedFile}>
                    <X className="h-4 w-4 mr-1" />
                    Close File
                  </Button>
                </div>
              </div>
            ) : (
              <FileDropZone
                onFile={processFile}
                placeholder="Drop a .jsonl, .csv, or .xml file here, or click to browse"
                description="Supports JSONL, CSV, and SemEval XML formats"
              />
            )
          )}
        </CardContent>
      </Card>

      {/* Stats & Filters */}
      {reviews.length > 0 && (
        <Card>
          <CardContent className="pt-4 space-y-4">
            {/* Top row: Pagination controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">
                  Showing {(currentPage - 1) * reviewsPerPage + 1}-{Math.min(currentPage * reviewsPerPage, filteredReviews.length)} of {filteredReviews.length}
                  {filteredReviews.length !== reviews.length && ` (${reviews.length} total)`}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm px-2">
                  Page {currentPage} of {totalPages || 1}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage >= totalPages}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Filters row */}
            <div className="flex flex-wrap items-center gap-3">
              {/* Search - on the left */}
              <div className="relative flex-1 min-w-[200px]">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search text, category, or target..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-8 h-8"
                />
              </div>

              <Separator orientation="vertical" className="h-6" />

              {/* Reviews per page */}
              <Select value={String(reviewsPerPage)} onValueChange={(v) => { setReviewsPerPage(Number(v)); setCurrentPage(1) }}>
                <SelectTrigger className="w-[100px] h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 / page</SelectItem>
                  <SelectItem value="25">25 / page</SelectItem>
                  <SelectItem value="50">50 / page</SelectItem>
                </SelectContent>
              </Select>

              {/* Polarity filter */}
              <Select value={polarityFilter} onValueChange={setPolarityFilter}>
                <SelectTrigger className="w-[130px] h-8">
                  <Filter className="mr-1 h-3 w-3" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Polarity</SelectItem>
                  <SelectItem value="positive">Positive</SelectItem>
                  <SelectItem value="neutral">Neutral</SelectItem>
                  <SelectItem value="negative">Negative</SelectItem>
                </SelectContent>
              </Select>

              {/* Category filter */}
              <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                <SelectTrigger className="w-[160px] h-8">
                  <Filter className="mr-1 h-3 w-3" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  {allCategories.map(cat => (
                    <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Sort */}
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-[120px] h-8">
                  <ArrowUpDown className="mr-1 h-3 w-3" />
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="id">By ID</SelectItem>
                  <SelectItem value="polarity">By Polarity</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Stats row */}
            <div className="flex flex-col gap-3">
              {/* Basic stats */}
              <div className="flex items-center gap-3 flex-wrap">
                <Badge variant="secondary" className="text-sm">
                  {reviews.length} reviews
                </Badge>
                <Badge variant="outline" className="text-sm">
                  {totalSentences} sentences
                </Badge>
                <Badge variant="outline" className="text-sm">
                  {avgSentencesPerReview} avg sent/review
                </Badge>
                <Badge variant="outline" className="text-sm">
                  {totalOpinions} opinions
                </Badge>
                <Badge variant="outline" className="text-sm">
                  {allCategories.length} categories
                </Badge>
                {inferredDomain.value && (
                  <Badge variant="secondary" className="text-sm">
                    Domain: {inferredDomain.value} ({Math.round(inferredDomain.confidence * 100)}%)
                  </Badge>
                )}
                {inferredRegion.value && (
                  <Badge variant="secondary" className="text-sm">
                    Region: {inferredRegion.value}
                  </Badge>
                )}
              </div>

              {/* Polarity stats - side by side */}
              <div className="flex items-center gap-6 flex-wrap">
                {/* Document-level polarity */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground font-medium">By Document:</span>
                  {Object.entries(polarityCounts).map(([pol, count]) => (
                    <Badge key={pol} className={getPolarityColor(pol)}>
                      {pol}: {count}
                    </Badge>
                  ))}
                </div>

                <Separator orientation="vertical" className="h-6" />

                {/* Sentence-level polarity */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground font-medium">By Sentence:</span>
                  {Object.entries(sentencePolarityCounts).map(([pol, count]) => (
                    <Badge key={`sent-${pol}`} variant="outline" className={getPolarityColor(pol)}>
                      {pol}: {count}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading reviews...</span>
        </div>
      )}

      {/* Reviews List */}
      {!loading && paginatedReviews.length > 0 && (
        <div className="space-y-3">
          {paginatedReviews.map(review => (
            <ReviewCard key={review.id} review={review} searchQuery={searchQuery} />
          ))}

          {/* Bottom pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 pt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>
              <span className="text-sm px-4">
                Page {currentPage} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage >= totalPages}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ReviewCard({ review, searchQuery }: { review: Review; searchQuery?: string }) {
  const polarity = resolvePolarity(review)
  const allCategories = Array.from(new Set(
    review.sentences.flatMap(s => s.opinions.map(o => o.category))
  ))

  return (
    <Card className="overflow-hidden">
      <CardContent className="pt-4">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <Badge variant="outline" className="text-xs">#{review.id}</Badge>
          <Badge className={getPolarityColor(polarity)}>{polarity}</Badge>
          {review.metadata?.age && (
            <span className="text-xs text-muted-foreground">Age: {review.metadata.age}</span>
          )}
          {review.metadata?.sex && (
            <span className="text-xs text-muted-foreground capitalize">{review.metadata.sex}</span>
          )}
        </div>

        {/* Sentences with highlighted opinions */}
        <div className="space-y-2">
          {review.sentences.map(sentence => (
            <HighlightedSentence key={sentence.id} sentence={sentence} searchQuery={searchQuery} />
          ))}
        </div>

        {/* Footer: categories */}
        {allCategories.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3 pt-3 border-t">
            {allCategories.map(cat => (
              <Badge key={cat} variant="outline" className="text-xs">{cat}</Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function HighlightedSentence({ sentence, searchQuery }: { sentence: Sentence; searchQuery?: string }) {
  // Helper to highlight search matches in blue within text
  const highlightSearch = (text: string, query: string | undefined) => {
    if (!query || !query.trim()) return [{ text, isMatch: false }]

    const lowerQuery = query.toLowerCase()
    const result: Array<{ text: string; isMatch: boolean }> = []
    let remaining = text
    let lowerRemaining = remaining.toLowerCase()

    while (lowerRemaining.includes(lowerQuery)) {
      const idx = lowerRemaining.indexOf(lowerQuery)
      if (idx > 0) {
        result.push({ text: remaining.slice(0, idx), isMatch: false })
      }
      result.push({ text: remaining.slice(idx, idx + query.length), isMatch: true })
      remaining = remaining.slice(idx + query.length)
      lowerRemaining = remaining.toLowerCase()
    }

    if (remaining) {
      result.push({ text: remaining, isMatch: false })
    }

    return result
  }

  // Render text with optional search highlighting
  const renderText = (text: string, polarityClass?: string) => {
    const segments = highlightSearch(text, searchQuery)
    return segments.map((seg, idx) => {
      if (seg.isMatch) {
        return (
          <span key={idx} className={`bg-blue-200 dark:bg-blue-800 rounded px-0.5 ${polarityClass || ''}`}>
            {seg.text}
          </span>
        )
      }
      if (polarityClass) {
        return <span key={idx} className={polarityClass}>{seg.text}</span>
      }
      return <span key={idx}>{seg.text}</span>
    })
  }

  if (!sentence.opinions.length) {
    return <p className="text-sm">{renderText(sentence.text)}</p>
  }

  // Sort opinions by 'from' position
  const sortedOpinions = [...sentence.opinions].sort((a, b) => a.from - b.from)

  // Build highlighted text
  const parts: Array<{ text: string; polarity?: string }> = []
  let lastIdx = 0

  for (const op of sortedOpinions) {
    if (op.from > lastIdx) {
      parts.push({ text: sentence.text.slice(lastIdx, op.from) })
    }
    if (op.from < op.to && op.to <= sentence.text.length) {
      parts.push({ text: sentence.text.slice(op.from, op.to), polarity: op.polarity })
      lastIdx = op.to
    } else {
      // Invalid offsets, skip highlight
      if (op.from >= lastIdx) {
        lastIdx = op.from
      }
    }
  }
  if (lastIdx < sentence.text.length) {
    parts.push({ text: sentence.text.slice(lastIdx) })
  }

  return (
    <p className="text-sm leading-relaxed">
      {parts.map((part, i) => (
        part.polarity ? (
          <span key={i} className={`rounded px-0.5 ${getPolarityHighlight(part.polarity)}`} title={part.polarity}>
            {renderText(part.text)}
          </span>
        ) : (
          <React.Fragment key={i}>{renderText(part.text)}</React.Fragment>
        )
      ))}
    </p>
  )
}

function inferPolarity(review: Review): string {
  const polarities = review.sentences.flatMap(s => s.opinions.map(o => o.polarity?.toLowerCase()))
  if (!polarities.length) return 'neutral'
  const counts: Record<string, number> = {}
  polarities.forEach(p => { if (p) counts[p] = (counts[p] || 0) + 1 })
  if (!Object.keys(counts).length) return 'neutral'
  return Object.entries(counts).sort(([, a], [, b]) => b - a)[0][0]
}


// ========================================
// MAV Reports Viewer Tab
// ========================================

function MavReportsViewer() {
  const settings = useQuery(api.settings.get)
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [selectedJob, setSelectedJob] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [jobsLoading, setJobsLoading] = useState(false)
  const [modelsData, setModelsData] = useState<Record<string, MavModelData>>({})
  const [report, setReport] = useState<MavReport | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [showConsensus, setShowConsensus] = useState(true)

  const fetchJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const jobsDir = settings?.jobsDirectory || './jobs'
      const res = await fetch(`${PYTHON_API_URL}/api/jobs-list?jobs_directory=${encodeURIComponent(jobsDir)}`)
      const data = await res.json()
      const jobsList = data.jobs || []
      setJobs(jobsList)
      // Auto-select latest job with MAV data if available
      const jobsWithMav = jobsList.filter((j: JobInfo) => j.hasMavs)
      if (jobsWithMav.length > 0 && !selectedJob) {
        setSelectedJob(jobsWithMav[0].path)
        loadMavData(jobsWithMav[0].path)
      }
    } catch {
      toast.error('Failed to load jobs list')
    } finally {
      setJobsLoading(false)
    }
  }, [settings?.jobsDirectory, selectedJob])

  // Auto-load jobs on mount when settings are available
  useEffect(() => {
    if (settings?.jobsDirectory) {
      fetchJobs()
    }
  }, [settings?.jobsDirectory, fetchJobs])

  const loadMavData = async (jobDir: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/read-mav-reports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobDir }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setModelsData(data.models || {})
      setReport(data.report || null)
      const modelNames = Object.keys(data.models || {})
      if (modelNames.length > 0) setSelectedModel(modelNames[0])
      toast.success(`Loaded MAV data (${modelNames.length} models)`)
    } catch {
      toast.error('Failed to load MAV reports')
    } finally {
      setLoading(false)
    }
  }

  const modelNames = Object.keys(modelsData)

  return (
    <div className="space-y-4">
      {/* Job Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Select Job</CardTitle>
          <CardDescription>Pick a job with MAV data to view reports</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <div className="flex-1">
              <Select
                value={selectedJob}
                onValueChange={(val) => {
                  setSelectedJob(val)
                  loadMavData(val)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder={jobsLoading ? 'Loading...' : 'Select a job'} />
                </SelectTrigger>
                <SelectContent>
                  {jobs.filter(j => j.hasMavs).map(job => (
                    <SelectItem key={job.dirName} value={job.path}>
                      {job.jobName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button variant="outline" onClick={fetchJobs} disabled={jobsLoading}>
              {jobsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Refresh'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading MAV data...</span>
        </div>
      )}

      {/* Summary Card */}
      {!loading && report && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">MAV Consensus Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Models</p>
                <p className="text-xl font-bold">{report.config?.models?.length || modelNames.length}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Total Queries</p>
                <p className="text-xl font-bold">{report.summary?.queries_after_dedup || 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Consensus Reached</p>
                <p className="text-xl font-bold">{report.summary?.queries_with_consensus || 0}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Consensus Rate</p>
                <p className="text-xl font-bold">{((report.summary?.consensus_rate || 0) * 100).toFixed(0)}%</p>
              </div>
            </div>
            {report.config && (
              <div className="mt-4 pt-4 border-t flex flex-wrap gap-2">
                <Badge variant="outline">Method: {report.config.consensus_method}</Badge>
                <Badge variant="outline">Threshold: {report.config.answer_similarity_threshold}</Badge>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Consensus / Per-Model View Toggle */}
      {!loading && modelNames.length > 0 && (
        <>
          <div className="flex gap-2">
            <Button
              variant={showConsensus ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowConsensus(true)}
            >
              Consensus View
            </Button>
            <Button
              variant={!showConsensus ? 'default' : 'outline'}
              size="sm"
              onClick={() => setShowConsensus(false)}
            >
              Per-Model View
            </Button>
          </div>

          {showConsensus ? (
            <ConsensusView report={report} />
          ) : (
            <PerModelView
              modelsData={modelsData}
              modelNames={modelNames}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          )}
        </>
      )}
    </div>
  )
}

function ConsensusView({ report }: { report: MavReport | null }) {
  if (!report?.per_query_consensus?.length) {
    return <p className="text-muted-foreground text-sm">No consensus data available</p>
  }

  return (
    <div className="space-y-3">
      {report.per_query_consensus.map((item) => (
        <Card key={item.query_id}>
          <CardContent className="pt-4">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                {item.consensus_reached ? (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-500" />
                )}
              </div>
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">{item.query_id}</Badge>
                  <Badge className={item.consensus_reached ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'}>
                    {item.consensus_reached ? 'Consensus' : 'No Consensus'}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {item.agreement_count}/{item.answers.length} agree
                  </span>
                </div>
                <p className="text-sm font-medium">{item.query}</p>
                {item.consensus_answer && (
                  <p className="text-sm text-muted-foreground bg-muted/50 rounded p-2">
                    {item.consensus_answer}
                  </p>
                )}
                {/* Per-model answers */}
                <div className="space-y-1 pt-2">
                  {item.answers.map((ans, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-xs">
                      <Badge variant="outline" className="text-[10px] shrink-0">
                        {ans.model.split('/').pop()}
                      </Badge>
                      <span className="text-muted-foreground flex-1">{ans.response}</span>
                      <Badge className={`text-[10px] shrink-0 ${CONFIDENCE_COLORS[ans.confidence] || ''}`}>
                        {ans.confidence}
                      </Badge>
                    </div>
                  ))}
                </div>
                {/* Pairwise similarities */}
                {item.pairwise_similarities && Object.keys(item.pairwise_similarities).length > 0 && (
                  <div className="pt-2 border-t">
                    <p className="text-xs text-muted-foreground mb-1">Pairwise Similarity:</p>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(item.pairwise_similarities).map(([pair, sim]) => (
                        <Badge key={pair} variant="outline" className="text-[10px]">
                          {pair.split('-').filter((_, i) => i % 2 === 0).join(' vs ')}: {(sim as number).toFixed(2)}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function PerModelView({
  modelsData,
  modelNames,
  selectedModel,
  setSelectedModel,
}: {
  modelsData: Record<string, MavModelData>
  modelNames: string[]
  selectedModel: string
  setSelectedModel: (m: string) => void
}) {
  const currentModel = modelsData[selectedModel]

  return (
    <div className="space-y-4">
      {/* Model selector tabs */}
      <div className="flex flex-wrap gap-2">
        {modelNames.map(name => (
          <Button
            key={name}
            variant={selectedModel === name ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedModel(name)}
          >
            {name.replace(/-/g, ' ')}
          </Button>
        ))}
      </div>

      {currentModel && (
        <div className="space-y-4">
          {/* Understanding */}
          {currentModel.understanding && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Understanding</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm whitespace-pre-wrap font-sans">{currentModel.understanding}</pre>
              </CardContent>
            </Card>
          )}

          {/* Queries */}
          {currentModel.queries && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Generated Queries</CardTitle>
                  <Badge variant="secondary">{currentModel.queries.count} queries</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <ScrollArea className="max-h-[300px]">
                  <ol className="list-decimal list-inside space-y-1 text-sm">
                    {currentModel.queries.queries.map((q, i) => (
                      <li key={i} className="text-muted-foreground">{q}</li>
                    ))}
                  </ol>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {/* Answers */}
          {currentModel.answers && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Answers</CardTitle>
                  <Badge variant="secondary">{currentModel.answers.answers.length} answers</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <ScrollArea className="max-h-[400px]">
                  <div className="space-y-3">
                    {currentModel.answers.answers.map((ans, idx) => (
                      <div key={idx} className="flex items-start gap-3 p-2 rounded border">
                        <Badge variant="outline" className="text-xs shrink-0">{ans.query_id}</Badge>
                        <p className="text-sm flex-1">{ans.response}</p>
                        <Badge className={`shrink-0 ${CONFIDENCE_COLORS[ans.confidence] || ''}`}>
                          {ans.confidence}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}


// ========================================
// Conformity Viewer Tab
// ========================================

interface ConformityReport {
  polarity: number
  length: number
  noise: number
  validation?: number
  temperature?: number
}

interface JobWithConformity {
  _id: string
  name: string
  status: string
  createdAt: number
  completedAt?: number
  jobDir?: string
  conformityReport?: ConformityReport
  config?: {
    generation?: {
      targets?: Array<{ count_mode: string; target_value: number }>
    }
  }
  perTargetMetrics?: Array<{ targetIndex: number; targetValue: number; conformity?: ConformityReport }>
}

// Extended conformity report for multi-run
interface ExtendedConformityReport {
  isMultiRun?: boolean
  totalRuns?: number
  runs?: Array<{
    run: number
    polarity: number
    length: number
    noise: number
    validation: number
    temperature?: number
    reviewCount?: number
  }>
  average?: ConformityReport
  std?: ConformityReport
  // Single-run fields
  polarity?: number
  length?: number
  noise?: number
  validation?: number
  temperature?: number
  reviewCount?: number
}

// Conformity metric descriptions
const CONFORMITY_METRICS = {
  polarity: {
    label: 'Polarity Conformity',
    description: 'How well the sentence-level sentiment distribution matches the target (measures opinion polarity across all sentences)',
    color: '#f2aa84',
  },
  length: {
    label: 'Length Conformity',
    description: 'Fraction of reviews within the target sentence length range',
    color: '#4e95d9',
  },
  noise: {
    label: 'Noise Conformity',
    description: 'Success rate of noise injection (typos, grammar errors, colloquialisms)',
    color: '#a78bfa',
  },
  validation: {
    label: 'JSON Validation',
    description: 'Fraction of reviews with valid structured JSON output from LLM',
    color: '#8ed973',
  },
  temperature: {
    label: 'Temperature Conformity',
    description: 'Fraction of reviews generated with temperature within the configured range',
    color: '#f97316',
  },
}

function ConformityViewer() {
  const settings = useQuery(api.settings.get)
  const jobs = useQuery(api.jobs.list)
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [fileConformity, setFileConformity] = useState<ExtendedConformityReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [multiRunView, setMultiRunView] = useState<'summary' | 'per-run'>('summary')
  const [selectedRun, setSelectedRun] = useState<number>(1)
  const [selectedTarget, setSelectedTarget] = useState<number | null>(null)

  // Filter jobs that have conformity reports or jobDir (may have file-based conformity)
  const jobsWithConformity = (jobs || []).filter(
    (job: JobWithConformity) => job.conformityReport != null || job.jobDir
  ) as JobWithConformity[]

  // Get selected job
  const selectedJob = jobsWithConformity.find((j) => j._id === selectedJobId)

  // Detect targets for selected job
  const jobTargets = selectedJob?.config?.generation?.targets || []
  const isMultiTarget = jobTargets.length > 1

  // Load conformity from files when job is selected
  const loadConformityFromFile = useCallback(async (jobDir: string, targetSize?: number | null) => {
    setLoading(true)
    try {
      const body: Record<string, unknown> = { jobDir }
      if (targetSize != null) body.targetSize = targetSize
      const res = await fetch(`${PYTHON_API_URL}/api/read-conformity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        const data = await res.json()
        setFileConformity(data.conformity || null)
        if (data.conformity?.runs?.length > 0) {
          setSelectedRun(1)
        }
      } else {
        setFileConformity(null)
      }
    } catch {
      setFileConformity(null)
    } finally {
      setLoading(false)
    }
  }, [])

  // Auto-select first job with conformity if none selected
  useEffect(() => {
    if (!selectedJobId && jobsWithConformity.length > 0) {
      setSelectedJobId(jobsWithConformity[0]._id)
    }
  }, [jobsWithConformity, selectedJobId])

  // Load file-based conformity when job or target changes
  useEffect(() => {
    if (selectedJob?.jobDir) {
      loadConformityFromFile(selectedJob.jobDir, selectedTarget)
    } else {
      setFileConformity(null)
    }
  }, [selectedJob?.jobDir, selectedTarget, loadConformityFromFile])

  // Get effective conformity data (file-based takes precedence for multi-run info)
  const conformityData: ExtendedConformityReport | null = fileConformity || (selectedJob?.conformityReport ? {
    polarity: selectedJob.conformityReport.polarity,
    length: selectedJob.conformityReport.length,
    noise: selectedJob.conformityReport.noise,
    validation: selectedJob.conformityReport.validation,
  } : null)

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.95) return 'text-green-600 dark:text-green-400'
    if (score >= 0.85) return 'text-amber-600 dark:text-amber-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getScoreBgColor = (score: number) => {
    if (score >= 0.95) return '#22c55e'
    if (score >= 0.85) return '#f59e0b'
    return '#ef4444'
  }

  return (
    <div className="space-y-4">
      {/* Job Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Select Job</CardTitle>
          <CardDescription>Pick a completed job to view its conformity report</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            <div className="flex-1">
              <Select
                value={selectedJobId}
                onValueChange={(val) => {
                  setSelectedJobId(val)
                  setSelectedTarget(null)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder={jobs === undefined ? 'Loading...' : 'Select a job'} />
                </SelectTrigger>
                <SelectContent>
                  {jobsWithConformity.map((job) => (
                    <SelectItem key={job._id} value={job._id}>
                      <div className="flex items-center gap-2">
                        <span>{job.name}</span>
                        <Badge variant="outline" className="text-xs">
                          {job.status}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {/* Target selector for multi-target jobs */}
            {isMultiTarget && (
              <div className="w-40">
                <Select
                  value={selectedTarget?.toString() || ''}
                  onValueChange={(val) => setSelectedTarget(parseInt(val))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Target" />
                  </SelectTrigger>
                  <SelectContent>
                    {jobTargets.map((t: { target_value: number; count_mode: string }) => (
                      <SelectItem key={t.target_value} value={t.target_value.toString()}>
                        {t.target_value} {t.count_mode === 'sentences' ? 'sent' : 'rev'}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* No Jobs Message */}
      {jobs !== undefined && jobsWithConformity.length === 0 && (
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <AlertCircle className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg font-medium">No Conformity Data Available</p>
              <p className="text-sm">Complete a job with generation phase to see conformity reports</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading conformity data...</span>
        </div>
      )}

      {/* Conformity Report */}
      {!loading && conformityData && selectedJob && (
        <>
          {/* Multi-run view toggle */}
          {conformityData.isMultiRun && conformityData.runs && (
            <Card className="mb-4">
              <CardContent className="py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400">
                      Multi-Run Dataset
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {conformityData.totalRuns || conformityData.runs.length} runs evaluated
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant={multiRunView === 'summary' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setMultiRunView('summary')}
                    >
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Summary
                    </Button>
                    <Button
                      variant={multiRunView === 'per-run' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setMultiRunView('per-run')}
                    >
                      <Eye className="mr-2 h-4 w-4" />
                      Per Run
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Multi-run Summary View */}
          {conformityData.isMultiRun && multiRunView === 'summary' && conformityData.runs && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">{selectedJob.name}</CardTitle>
                    <CardDescription>
                      {selectedJob.completedAt ? formatDate(selectedJob.completedAt) : formatDate(selectedJob.createdAt)}
                    </CardDescription>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">
                      {(() => {
                        const avg = conformityData.average || conformityData
                        const scores = [avg.polarity, avg.length, avg.noise].filter(v => v != null) as number[]
                        if (avg.validation != null) scores.push(avg.validation)
                        if (avg.temperature != null) scores.push(avg.temperature)
                        const total = scores.reduce((a, b) => a + b, 0) / scores.length
                        return <span className={getScoreColor(total)}>{Math.round(total * 100)}%</span>
                      })()}
                    </div>
                    <p className="text-xs text-muted-foreground">Average Conformity</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 pr-4 font-medium">Metric</th>
                        {conformityData.runs.map(run => (
                          <th key={run.run} className="text-right py-2 px-2 font-medium text-muted-foreground">
                            Run {run.run}
                          </th>
                        ))}
                        <th className="text-right py-2 px-2 font-medium text-green-600 dark:text-green-400">Avg</th>
                        <th className="text-right py-2 pl-2 font-medium text-muted-foreground">Std</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(CONFORMITY_METRICS).map(([key, meta]) => {
                        const avgVal = (conformityData.average?.[key as keyof ConformityReport] ?? conformityData[key as keyof ExtendedConformityReport]) as number | undefined
                        const stdVal = conformityData.std?.[key as keyof ConformityReport]
                        if (avgVal == null) return null

                        return (
                          <tr key={key} className="border-b last:border-0">
                            <td className="py-2 pr-4 font-medium">{meta.label}</td>
                            {conformityData.runs!.map(run => {
                              const val = run[key as keyof typeof run] as number
                              return (
                                <td key={run.run} className="text-right py-2 px-2 font-mono text-xs" style={{ color: getScoreBgColor(val) }}>
                                  {Math.round(val * 100)}%
                                </td>
                              )
                            })}
                            <td className="text-right py-2 px-2 font-mono text-xs font-medium text-green-600 dark:text-green-400">
                              {Math.round(avgVal * 100)}%
                            </td>
                            <td className="text-right py-2 pl-2 font-mono text-xs text-muted-foreground">
                              ±{stdVal != null ? Math.round(stdVal * 100) : 0}%
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Multi-run Per-Run View */}
          {conformityData.isMultiRun && multiRunView === 'per-run' && conformityData.runs && (
            <>
              <Card className="mb-4">
                <CardContent className="py-4">
                  <div className="flex items-center gap-4">
                    <Label className="text-sm font-medium">Select Run:</Label>
                    <Select value={selectedRun.toString()} onValueChange={v => setSelectedRun(parseInt(v))}>
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="Select run" />
                      </SelectTrigger>
                      <SelectContent>
                        {conformityData.runs.map(run => (
                          <SelectItem key={run.run} value={run.run.toString()}>
                            Run {run.run}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
              {(() => {
                const runData = conformityData.runs!.find(r => r.run === selectedRun)
                if (!runData) return null
                return (
                  <div className="grid gap-4 md:grid-cols-2">
                    {Object.entries(CONFORMITY_METRICS).map(([key, meta]) => {
                      const value = runData[key as keyof typeof runData] as number
                      if (value == null) return null
                      const score = Math.round(value * 100)
                      return (
                        <Card key={key}>
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-base">{meta.label}</CardTitle>
                              <span className="text-2xl font-bold" style={{ color: getScoreBgColor(value) }}>
                                {score}%
                              </span>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <Progress value={score} className="h-3" indicatorColor={getScoreBgColor(value)} />
                            <p className="text-xs text-muted-foreground">{meta.description}</p>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                )
              })()}
            </>
          )}

          {/* Single-run view */}
          {!conformityData.isMultiRun && (
            <div className="grid gap-4 md:grid-cols-2">
              {/* Overview Card */}
              <Card className="md:col-span-2">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">{selectedJob.name}</CardTitle>
                      <CardDescription>
                        {selectedJob.completedAt ? formatDate(selectedJob.completedAt) : formatDate(selectedJob.createdAt)}
                      </CardDescription>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">
                        {(() => {
                          const scores = [conformityData.polarity, conformityData.length, conformityData.noise].filter(v => v != null) as number[]
                          if (conformityData.validation != null) scores.push(conformityData.validation)
                          if (conformityData.temperature != null) scores.push(conformityData.temperature)
                          const avg = scores.reduce((a, b) => a + b, 0) / scores.length
                          return <span className={getScoreColor(avg)}>{Math.round(avg * 100)}%</span>
                        })()}
                      </div>
                      <p className="text-xs text-muted-foreground">Average Conformity</p>
                    </div>
                  </div>
                </CardHeader>
              </Card>

              {/* Individual Metrics */}
              {Object.entries(CONFORMITY_METRICS).map(([key, meta]) => {
                const value = conformityData[key as keyof ExtendedConformityReport] as number | undefined
                if (value == null) return null
                const score = Math.round(value * 100)

                return (
                  <Card key={key}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base">{meta.label}</CardTitle>
                        <span className="text-2xl font-bold" style={{ color: getScoreBgColor(value) }}>
                          {score}%
                        </span>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <Progress value={score} className="h-3" indicatorColor={getScoreBgColor(value)} />
                      <p className="text-xs text-muted-foreground">{meta.description}</p>
                      {key === 'validation' && score < 100 && (
                        <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
                          <AlertCircle className="h-3 w-3" />
                          <span>{100 - score}% of reviews fell back to plain text</span>
                        </div>
                      )}
                      {key === 'polarity' && score < 95 && (
                        <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
                          <Info className="h-3 w-3" />
                          <span>Sentence-level sentiment mix differs from target distribution</span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </>
      )}

      {/* Interpretation Guide */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Info className="h-4 w-4" />
            Interpretation Guide
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-sm font-medium">Excellent (95-100%)</span>
              </div>
              <p className="text-xs text-muted-foreground">Near-perfect conformity to specifications</p>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-amber-500" />
                <span className="text-sm font-medium">Good (85-94%)</span>
              </div>
              <p className="text-xs text-muted-foreground">Minor deviations from target</p>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-sm font-medium">Needs Review (&lt;85%)</span>
              </div>
              <p className="text-xs text-muted-foreground">Significant deviation, check configuration</p>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-500" />
                <span className="text-sm font-medium">100% Validation</span>
              </div>
              <p className="text-xs text-muted-foreground">All reviews have valid structured output</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}


// ========================================
// Metrics Viewer Tab
// ========================================

const METRICS_VIEWER_STORAGE_KEY = 'cera-metrics-viewer-state'

// Types for multi-run metrics
interface PerRunMetrics {
  [category: string]: {
    [runKey: string]: MetricEntry[]  // e.g., run1, run2, etc.
  }
}

interface SummaryRow {
  metric: string
  [key: string]: string  // run1, run2, ..., average, std
}

interface SummaryMetrics {
  [category: string]: SummaryRow[]
}

function MetricsViewer() {
  const settings = useQuery(api.settings.get)
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [selectedJob, setSelectedJob] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [jobsLoading, setJobsLoading] = useState(false)
  const [metrics, setMetrics] = useState<Record<string, MetricEntry[]>>({})
  const [sourceMode, setSourceMode] = useState<'job' | 'import'>('job')
  const [importedFiles, setImportedFiles] = useState<Record<string, string>>({})
  // Multi-run state
  const [perRunMetrics, setPerRunMetrics] = useState<PerRunMetrics>({})
  const [summaryMetrics, setSummaryMetrics] = useState<SummaryMetrics>({})
  const [isMultiRun, setIsMultiRun] = useState(false)
  const [multiRunView, setMultiRunView] = useState<'summary' | 'per-run' | 'per-model'>('summary')
  const [selectedRun, setSelectedRun] = useState<string>('')
  const [selectedTarget, setSelectedTarget] = useState<number | null>(null)
  // Per-model metrics from mdqa-results.json
  const [perModelData, setPerModelData] = useState<Array<{
    model: string
    modelSlug: string
    metrics: Record<string, number>
    runs?: Array<{ run: number; metrics: Record<string, number> }>
  }>>([])
  const [isMultiModel, setIsMultiModel] = useState(false)

  // Restore state from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(METRICS_VIEWER_STORAGE_KEY)
      if (saved) {
        const state = JSON.parse(saved)
        if (state.sourceMode) setSourceMode(state.sourceMode)
        if (state.importedFiles) setImportedFiles(state.importedFiles)
        if (state.metrics && Object.keys(state.metrics).length > 0) setMetrics(state.metrics)
      }
    } catch {
      // Ignore parse errors
    }
  }, [])

  // Save imported file state to localStorage
  useEffect(() => {
    if (sourceMode === 'import' && Object.keys(metrics).length > 0 && Object.keys(importedFiles).length > 0) {
      localStorage.setItem(METRICS_VIEWER_STORAGE_KEY, JSON.stringify({
        sourceMode,
        importedFiles,
        metrics,
      }))
    }
  }, [sourceMode, metrics, importedFiles])

  const closeImportedFiles = () => {
    setMetrics({})
    setImportedFiles({})
    setPerRunMetrics({})
    setSummaryMetrics({})
    setIsMultiRun(false)
    setSelectedRun('')
    setPerModelData([])
    setIsMultiModel(false)
    localStorage.removeItem(METRICS_VIEWER_STORAGE_KEY)
  }

  const fetchJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const jobsDir = settings?.jobsDirectory || './jobs'
      const res = await fetch(`${PYTHON_API_URL}/api/jobs-list?jobs_directory=${encodeURIComponent(jobsDir)}`)
      const data = await res.json()
      const jobsList = data.jobs || []
      setJobs(jobsList)
      // Auto-select latest job with metrics if available
      const jobsWithMetrics = jobsList.filter((j: JobInfo) => j.hasMetrics)
      if (jobsWithMetrics.length > 0 && !selectedJob) {
        const firstJob = jobsWithMetrics[0]
        setSelectedJob(firstJob.path)
        const firstTarget = firstJob.targets?.find((t: TargetInfo) => t.hasMetrics)
        if (firstTarget) {
          setSelectedTarget(firstTarget.size)
          loadMetrics(firstJob.path, firstTarget.size)
        } else {
          loadMetrics(firstJob.path)
        }
      }
    } catch {
      toast.error('Failed to load jobs list')
    } finally {
      setJobsLoading(false)
    }
  }, [settings?.jobsDirectory, selectedJob])

  // Auto-load jobs on initial mount when settings are available
  useEffect(() => {
    if (settings?.jobsDirectory && jobs.length === 0) {
      fetchJobs()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings?.jobsDirectory]) // Only run on mount and when settings change

  const loadMetrics = async (jobDir: string, targetSize?: number | null) => {
    setLoading(true)
    try {
      const body: Record<string, unknown> = { jobDir }
      if (targetSize != null) body.targetSize = targetSize
      const res = await fetch(`${PYTHON_API_URL}/api/read-metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const metricsData = data.metrics || {}

      // Check for multi-run data (new consolidated format)
      if (metricsData.isMultiRun && metricsData.perRun) {
        setPerRunMetrics(metricsData.perRun)
        setIsMultiRun(true)
        // Set initial selected run
        const firstCategory = Object.keys(metricsData.perRun)[0]
        const runs = Object.keys(metricsData.perRun[firstCategory] || {}).sort(
          (a, b) => parseInt(a.replace('run', '')) - parseInt(b.replace('run', ''))
        )
        if (runs.length > 0) {
          setSelectedRun(runs[0])
        }

        // Build summary from average and std
        const summary: SummaryMetrics = {}
        const stdData = metricsData.std || {}
        for (const category of ['lexical', 'semantic', 'diversity']) {
          if (metricsData.average?.[category]) {
            summary[category] = metricsData.average[category].map((avgEntry: MetricEntry) => {
              const row: SummaryRow = { metric: avgEntry.metric }
              // Add per-run values
              for (const runKey of runs) {
                const runMetrics = metricsData.perRun[category]?.[runKey] || []
                const runEntry = runMetrics.find((m: MetricEntry) => m.metric === avgEntry.metric)
                row[runKey] = runEntry?.score?.toString() || '0'
              }
              // Add average and std
              row.average = avgEntry.score?.toString() || '0'
              const stdEntry = stdData[category]?.find((s: MetricEntry) => s.metric === avgEntry.metric)
              row.std = stdEntry?.score?.toString() || '0'
              return row
            })
          }
        }
        setSummaryMetrics(summary)

        // Set main metrics to averages for display
        const avgMetrics: Record<string, MetricEntry[]> = {}
        for (const category of ['lexical', 'semantic', 'diversity']) {
          if (metricsData.average?.[category]) {
            avgMetrics[category] = metricsData.average[category]
          } else if (metricsData[category]) {
            avgMetrics[category] = metricsData[category]
          }
        }
        setMetrics(avgMetrics)

        // Check for per-model data
        if (metricsData.perModel && Array.isArray(metricsData.perModel) && metricsData.perModel.length > 1) {
          setPerModelData(metricsData.perModel)
          setIsMultiModel(true)
        } else {
          setPerModelData([])
          setIsMultiModel(false)
        }

        toast.success(`Loaded metrics from ${metricsData.totalRuns || runs.length} runs`)
      } else {
        // Single-run data
        const { perRun, average, std, isMultiRun: _, totalRuns: __, ...singleRunMetrics } = metricsData
        setMetrics(singleRunMetrics)
        setPerRunMetrics({})
        setSummaryMetrics({})
        setIsMultiRun(false)
        setSelectedRun('')
        setPerModelData([])
        setIsMultiModel(false)
        toast.success('Loaded metrics data')
      }
    } catch {
      toast.error('Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  const processCsvFile = async (category: string, file: File) => {
    try {
      const text = await file.text()
      const lines = text.split('\n').filter(l => l.trim())
      if (lines.length < 2) return

      const entries: MetricEntry[] = []
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',')
        if (parts.length >= 2) {
          entries.push({
            metric: parts[0].trim(),
            score: parseFloat(parts[1].trim()),
            description: parts.slice(2).join(',').trim(),
          })
        }
      }
      setMetrics(prev => ({ ...prev, [category]: entries }))
      setImportedFiles(prev => ({ ...prev, [category]: file.name }))
      toast.success(`Imported ${category} metrics`)
    } catch {
      toast.error(`Failed to parse ${category} metrics file`)
    }
  }

  const METRIC_COLS = [
    { key: 'bleu', label: 'BLEU', higher: true },
    { key: 'rouge_l', label: 'ROUGE-L', higher: true },
    { key: 'bertscore', label: 'BERTScore', higher: true },
    { key: 'moverscore', label: 'MoverScore', higher: true },
    { key: 'distinct_1', label: 'Dist-1', higher: true },
    { key: 'distinct_2', label: 'Dist-2', higher: true },
    { key: 'self_bleu', label: 'Self-BLEU', higher: false },
  ]

  const CATEGORY_INFO: Record<string, { label: string; color: string; icon: typeof BarChart3 }> = {
    lexical: { label: 'Lexical Metrics', color: 'text-blue-500', icon: FileText },
    semantic: { label: 'Semantic Metrics', color: 'text-purple-500', icon: Eye },
    diversity: { label: 'Diversity Metrics', color: 'text-green-500', icon: BarChart3 },
  }

  // Metrics where lower is better
  const LOWER_IS_BETTER = new Set(['self_bleu'])

  // Metric thresholds for quality indicators (poor is the boundary between Fair and Poor)
  const METRIC_THRESHOLDS: Record<string, { poor: number; good: number; excellent: number; lowerIsBetter?: boolean }> = {
    bleu: { poor: 0.25, good: 0.40, excellent: 0.50 },
    rouge_l: { poor: 0.25, good: 0.40, excellent: 0.50 },
    bertscore: { poor: 0.55, good: 0.70, excellent: 0.85 },
    moverscore: { poor: 0.35, good: 0.50, excellent: 0.65 },
    distinct_1: { poor: 0.30, good: 0.50, excellent: 0.70 },
    distinct_2: { poor: 0.60, good: 0.80, excellent: 0.90 },
    self_bleu: { poor: 0.50, good: 0.30, excellent: 0.20, lowerIsBetter: true },
  }

  // Get quality indicator based on value and thresholds
  const getQualityIndicator = (metricKey: string, value: number) => {
    const threshold = METRIC_THRESHOLDS[metricKey]
    if (!threshold) return null

    if (threshold.lowerIsBetter) {
      if (value <= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400' }
      if (value <= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400' }
      if (value <= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400' }
      return { label: 'Poor', color: 'text-red-600 dark:text-red-400' }
    } else {
      if (value >= threshold.excellent) return { label: 'Excellent', color: 'text-green-600 dark:text-green-400' }
      if (value >= threshold.good) return { label: 'Good', color: 'text-blue-600 dark:text-blue-400' }
      if (value >= threshold.poor) return { label: 'Fair', color: 'text-amber-600 dark:text-amber-400' }
      return { label: 'Poor', color: 'text-red-600 dark:text-red-400' }
    }
  }

  // Get threshold text for a metric
  const getThresholdText = (metricKey: string) => {
    const threshold = METRIC_THRESHOLDS[metricKey]
    if (!threshold) return ''
    return threshold.lowerIsBetter
      ? `≤${threshold.good} good · ≤${threshold.excellent} excellent`
      : `≥${threshold.good} good · ≥${threshold.excellent} excellent`
  }

  return (
    <div className="space-y-4">
      {/* Source Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Data Source</CardTitle>
          <CardDescription>Select a job or import metric CSV files</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button
              variant={sourceMode === 'job' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSourceMode('job')}
            >
              <FileText className="mr-2 h-4 w-4" />
              From Job
            </Button>
            <Button
              variant={sourceMode === 'import' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSourceMode('import')}
            >
              <Upload className="mr-2 h-4 w-4" />
              Import Files
            </Button>
          </div>

          {sourceMode === 'job' && (
            <div className="flex gap-3">
              <div className="flex-1">
                <Select
                  value={selectedJob}
                  onValueChange={(val) => {
                    setSelectedJob(val)
                    setSelectedTarget(null)
                    const job = jobs.find(j => j.path === val)
                    const firstTarget = job?.targets?.find(t => t.hasMetrics)
                    if (firstTarget) {
                      setSelectedTarget(firstTarget.size)
                      loadMetrics(val, firstTarget.size)
                    } else {
                      loadMetrics(val)
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={jobsLoading ? 'Loading...' : 'Select a job'} />
                  </SelectTrigger>
                  <SelectContent>
                    {jobs.filter(j => j.hasMetrics).map(job => (
                      <SelectItem key={job.dirName} value={job.path}>
                        {job.jobName}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {/* Target selector for multi-target jobs */}
              {(() => {
                const job = jobs.find(j => j.path === selectedJob)
                const targetsWithMetrics = job?.targets?.filter(t => t.hasMetrics) || []
                if (targetsWithMetrics.length <= 1) return null
                return (
                  <div className="w-40">
                    <Select
                      value={selectedTarget?.toString() || ''}
                      onValueChange={(val) => {
                        const size = parseInt(val)
                        setSelectedTarget(size)
                        loadMetrics(selectedJob, size)
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Target" />
                      </SelectTrigger>
                      <SelectContent>
                        {targetsWithMetrics.map(t => (
                          <SelectItem key={t.size} value={t.size.toString()}>
                            {t.size} sent
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )
              })()}
              <Button variant="outline" onClick={fetchJobs} disabled={jobsLoading}>
                {jobsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Refresh'}
              </Button>
            </div>
          )}

          {sourceMode === 'import' && (
            <div className="space-y-4">
              {Object.keys(importedFiles).length > 0 && (
                <div className="flex items-center justify-between p-3 border rounded-lg bg-muted/30">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="font-medium">{Object.keys(importedFiles).length} metric file(s) loaded</p>
                      <p className="text-sm text-muted-foreground">
                        {Object.entries(importedFiles).map(([cat, name]) => `${cat}: ${name}`).join(' · ')}
                      </p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" onClick={closeImportedFiles}>
                    <X className="h-4 w-4 mr-1" />
                    Close Files
                  </Button>
                </div>
              )}
              <div className="grid gap-4 sm:grid-cols-3">
                {['lexical', 'semantic', 'diversity'].map(category => (
                  <div key={category} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium capitalize">{category} Metrics</p>
                      {metrics[category] && (
                        <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                          <CheckCircle className="mr-1 h-3 w-3" />
                          Loaded
                        </Badge>
                      )}
                    </div>
                    <FileDropZone
                      onFile={(f) => processCsvFile(category, f)}
                      accept=".csv"
                      placeholder={importedFiles[category] || `Drop ${category}-metrics.csv`}
                      className="py-4"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          <span className="ml-2 text-muted-foreground">Loading metrics...</span>
        </div>
      )}

      {/* Metrics Display */}
      {!loading && Object.keys(metrics).length > 0 && (
        <>
          {/* Multi-run view toggle */}
          {isMultiRun && (
            <Card className="mb-4">
              <CardContent className="py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400">
                      Multi-Run Dataset
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {Object.keys(perRunMetrics[Object.keys(perRunMetrics)[0]] || {}).length} runs
                      {isMultiModel && ` \u00d7 ${perModelData.length} models`} evaluated
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant={multiRunView === 'summary' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setMultiRunView('summary')}
                    >
                      <BarChart3 className="mr-2 h-4 w-4" />
                      Summary
                    </Button>
                    <Button
                      variant={multiRunView === 'per-run' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setMultiRunView('per-run')}
                    >
                      <Eye className="mr-2 h-4 w-4" />
                      Per Run
                    </Button>
                    {isMultiModel && (
                      <Button
                        variant={multiRunView === 'per-model' ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setMultiRunView('per-model')}
                      >
                        <Users className="mr-2 h-4 w-4" />
                        Per Model
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Multi-run Summary View */}
          {isMultiRun && multiRunView === 'summary' && Object.keys(summaryMetrics).length > 0 && (
            <div className="space-y-4">
              {['lexical', 'semantic', 'diversity'].map(category => {
                const catSummary = summaryMetrics[category]
                if (!catSummary || catSummary.length === 0) return null
                const info = CATEGORY_INFO[category]
                const Icon = info.icon

                // Get run columns from first row
                const firstRow = catSummary[0]
                const runColumns = Object.keys(firstRow)
                  .filter(k => k.startsWith('run'))
                  .sort((a, b) => parseInt(a.replace('run', '')) - parseInt(b.replace('run', '')))

                return (
                  <Card key={category}>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 ${info.color}`} />
                        <CardTitle className="text-base">{info.label}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left py-2 pr-4 font-medium">Metric</th>
                              {runColumns.map(run => (
                                <th key={run} className="text-right py-2 px-2 font-medium text-muted-foreground">
                                  {run.replace('run', 'Run ')}
                                </th>
                              ))}
                              <th className="text-right py-2 px-2 font-medium text-green-600 dark:text-green-400">Avg</th>
                              <th className="text-right py-2 pl-2 font-medium text-muted-foreground">Std</th>
                            </tr>
                          </thead>
                          <tbody>
                            {catSummary.map((row) => {
                              const avgValue = parseFloat(row.average) || 0
                              const quality = getQualityIndicator(row.metric, avgValue)
                              const isLowerBetter = LOWER_IS_BETTER.has(row.metric)

                              return (
                                <tr key={row.metric} className="border-b last:border-0">
                                  <td className="py-2 pr-4">
                                    <div className="flex items-center gap-2">
                                      <span className="font-medium">
                                        {row.metric.replace(/_/g, '-').replace(/\b\w/g, c => c.toUpperCase())}
                                      </span>
                                      {quality && (
                                        <span className={`text-[10px] font-medium ${quality.color}`}>
                                          {quality.label}
                                        </span>
                                      )}
                                      <span className={`text-[9px] ${isLowerBetter ? 'text-amber-500' : 'text-green-500'}`}>
                                        {isLowerBetter ? '↓' : '↑'}
                                      </span>
                                    </div>
                                  </td>
                                  {runColumns.map(run => (
                                    <td key={run} className="text-right py-2 px-2 font-mono text-xs text-muted-foreground">
                                      {parseFloat(row[run] || '0').toFixed(4)}
                                    </td>
                                  ))}
                                  <td className="text-right py-2 px-2 font-mono text-xs font-medium text-green-600 dark:text-green-400">
                                    {avgValue.toFixed(4)}
                                  </td>
                                  <td className="text-right py-2 pl-2 font-mono text-xs text-muted-foreground">
                                    ±{parseFloat(row.std || '0').toFixed(4)}
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}

          {/* Multi-run Per-Run View */}
          {isMultiRun && multiRunView === 'per-run' && (
            <>
              <Card className="mb-4">
                <CardContent className="py-4">
                  <div className="flex items-center gap-4">
                    <Label className="text-sm font-medium">Select Run:</Label>
                    <Select value={selectedRun} onValueChange={setSelectedRun}>
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="Select run" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.keys(perRunMetrics[Object.keys(perRunMetrics)[0]] || {})
                          .sort((a, b) => parseInt(a.replace('run', '')) - parseInt(b.replace('run', '')))
                          .map(run => (
                            <SelectItem key={run} value={run}>
                              {run.replace('run', 'Run ')}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
              <div className="grid gap-4 md:grid-cols-3">
                {['lexical', 'semantic', 'diversity'].map(category => {
                  const catRunMetrics = perRunMetrics[category]?.[selectedRun]
                  if (!catRunMetrics) return null
                  const info = CATEGORY_INFO[category]
                  const Icon = info.icon

                  return (
                    <Card key={category}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-2">
                          <Icon className={`h-4 w-4 ${info.color}`} />
                          <CardTitle className="text-base">{info.label}</CardTitle>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {catRunMetrics.map(entry => {
                          const isLowerBetter = LOWER_IS_BETTER.has(entry.metric)
                          const displayScore = Math.min(entry.score, 1) * 100
                          const quality = getQualityIndicator(entry.metric, entry.score)
                          const thresholdText = getThresholdText(entry.metric)

                          return (
                            <div key={entry.metric} className="space-y-1">
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium">
                                  {entry.metric.replace(/_/g, '-').replace(/\b\w/g, c => c.toUpperCase())}
                                </span>
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-mono">{entry.score.toFixed(4)}</span>
                                  {quality && (
                                    <span className={`text-[10px] font-medium ${quality.color}`}>
                                      {quality.label}
                                    </span>
                                  )}
                                </div>
                              </div>
                              <Progress value={displayScore} className="h-2" />
                              <div className="flex items-center justify-between">
                                <span className="text-xs text-muted-foreground">{entry.description}</span>
                                <span className={`text-[10px] ${isLowerBetter ? 'text-amber-600 dark:text-amber-400' : 'text-green-600 dark:text-green-400'}`}>
                                  {isLowerBetter ? '↓ lower is better' : '↑ higher is better'}
                                </span>
                              </div>
                              {thresholdText && (
                                <p className="text-[9px] text-muted-foreground/50 text-right">
                                  {thresholdText}
                                </p>
                              )}
                            </div>
                          )
                        })}
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </>
          )}

          {/* Multi-model Per-Model View */}
          {isMultiRun && multiRunView === 'per-model' && perModelData.length > 0 && (() => {
            // Find best value per metric for highlighting
            const bestValues: Record<string, number> = {}
            for (const col of METRIC_COLS) {
              const values = perModelData
                .map(pm => pm.metrics?.[col.key])
                .filter((v): v is number => v !== undefined)
              if (values.length > 0) {
                bestValues[col.key] = col.higher
                  ? Math.max(...values)
                  : Math.min(...values)
              }
            }
            const numRuns = Object.keys(perRunMetrics[Object.keys(perRunMetrics)[0]] || {}).length

            return (
              <div className="space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">Model Comparison</CardTitle>
                    <CardDescription>
                      Average metrics across {numRuns} runs per model
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2 text-xs font-semibold text-muted-foreground">Model</th>
                            {METRIC_COLS.map(col => (
                              <th key={col.key} className="text-center py-2 px-2 text-xs font-semibold text-muted-foreground">
                                {col.label}
                                <span className="block text-[9px] font-normal opacity-60">
                                  {col.higher ? '\u2191' : '\u2193'}
                                </span>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {perModelData.map(pm => (
                            <tr key={pm.model} className="border-b last:border-0">
                              <td className="py-2.5 px-2">
                                <span className="font-medium truncate max-w-[200px] block" title={pm.model}>
                                  {pm.modelSlug}
                                </span>
                              </td>
                              {METRIC_COLS.map(col => {
                                const value = pm.metrics?.[col.key]
                                const quality = getQualityIndicator(col.key, value ?? 0)
                                const isBest = value !== undefined && bestValues[col.key] === value && perModelData.length > 1
                                return (
                                  <td key={col.key} className={`text-center py-2.5 px-2 ${isBest ? 'font-bold' : ''}`}>
                                    {value !== undefined ? (
                                      <div className="flex flex-col items-center">
                                        <span className="font-mono text-xs">{value.toFixed(4)}</span>
                                        {quality && (
                                          <span className={`text-[9px] ${quality.color}`}>{quality.label}</span>
                                        )}
                                      </div>
                                    ) : (
                                      <span className="text-muted-foreground">&mdash;</span>
                                    )}
                                  </td>
                                )
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>

                {/* Per-model per-run breakdown */}
                {perModelData.some(pm => pm.runs && pm.runs.length > 1) && (
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">Per-Run Breakdown</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {perModelData.map(pm => {
                        if (!pm.runs || pm.runs.length <= 1) return null
                        return (
                          <div key={pm.model} className="space-y-2">
                            <h4 className="text-sm font-medium">{pm.modelSlug}</h4>
                            <div className="space-y-1.5 pl-2">
                              {pm.runs.map(runData => (
                                <div key={runData.run} className="rounded border p-2 bg-muted/20">
                                  <div className="flex justify-between items-center mb-1.5">
                                    <span className="text-xs font-medium">Run {runData.run}</span>
                                  </div>
                                  <div className="grid gap-1.5 grid-cols-2 sm:grid-cols-4 lg:grid-cols-7">
                                    {METRIC_COLS.map(col => {
                                      const value = runData.metrics?.[col.key]
                                      if (value === undefined) return null
                                      const quality = getQualityIndicator(col.key, value)
                                      return (
                                        <div key={col.key} className="text-center p-1.5 rounded bg-background">
                                          <p className="text-[9px] text-muted-foreground uppercase">{col.label}</p>
                                          <p className="text-xs font-semibold font-mono">{value.toFixed(4)}</p>
                                          {quality && <span className={`text-[8px] ${quality.color}`}>{quality.label}</span>}
                                        </div>
                                      )
                                    })}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )
                      })}
                    </CardContent>
                  </Card>
                )}
              </div>
            )
          })()}

          {/* Single-run view */}
          {!isMultiRun && (
            <div className="grid gap-4 md:grid-cols-3">
              {['lexical', 'semantic', 'diversity'].map(category => {
                const catMetrics = metrics[category]
                if (!catMetrics) return null
                const info = CATEGORY_INFO[category]
                const Icon = info.icon

                return (
                  <Card key={category}>
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <Icon className={`h-4 w-4 ${info.color}`} />
                        <CardTitle className="text-base">{info.label}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {catMetrics.map(entry => {
                        const isLowerBetter = LOWER_IS_BETTER.has(entry.metric)
                        const displayScore = Math.min(entry.score, 1) * 100
                        const quality = getQualityIndicator(entry.metric, entry.score)
                        const thresholdText = getThresholdText(entry.metric)

                        return (
                          <div key={entry.metric} className="space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">
                                {entry.metric.replace(/_/g, '-').replace(/\b\w/g, c => c.toUpperCase())}
                              </span>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-mono">{entry.score.toFixed(4)}</span>
                                {quality && (
                                  <span className={`text-[10px] font-medium ${quality.color}`}>
                                    {quality.label}
                                  </span>
                                )}
                              </div>
                            </div>
                            <Progress value={displayScore} className="h-2" />
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-muted-foreground">{entry.description}</span>
                              <span className={`text-[10px] ${isLowerBetter ? 'text-amber-600 dark:text-amber-400' : 'text-green-600 dark:text-green-400'}`}>
                                {isLowerBetter ? '↓ lower is better' : '↑ higher is better'}
                              </span>
                            </div>
                            {thresholdText && (
                              <p className="text-[9px] text-muted-foreground/50 text-right">
                                {thresholdText}
                              </p>
                            )}
                          </div>
                        )
                      })}
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </>
      )}

      {!loading && Object.keys(metrics).length === 0 && selectedJob && (
        <div className="flex items-center justify-center py-12 text-muted-foreground">
          <AlertCircle className="mr-2 h-5 w-5" />
          <span>No metrics data found for this job</span>
        </div>
      )}
    </div>
  )
}


// ========================================
// Tokens Viewer Tab
// ========================================

interface TokensReport {
  version: number
  job_id: string
  method: string
  generated_at: string
  totals: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    total_calls: number
    estimated_cost_usd: number
  }
  by_phase: Record<string, {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    total_calls: number
    estimated_cost_usd: number
    components: Record<string, {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      total_calls: number
      estimated_cost_usd: number
    }>
  }>
  by_model: Record<string, {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    total_calls: number
    estimated_cost_usd: number
    pricing?: { prompt_per_token: number; completion_per_token: number }
  }>
  by_target: Record<string, {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    total_calls: number
    estimated_cost_usd: number
    runs?: Record<string, {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      total_calls: number
      estimated_cost_usd: number
    }>
  }>
  records: Array<{
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    model: string
    component: string
    target: string
    run: string
    timestamp: string
    generation_id?: string
  }>
}

// Pipeline execution order for SIL components
const SIL_COMPONENT_ORDER = [
  'sil.research',
  'sil.search',
  'sil.queries',
  'sil.answers',
  'sil.judge',
  'sil.classify',
  'sil.cluster',
  'sil.cluster_judge',
]

// Component label display names
const COMPONENT_LABELS: Record<string, string> = {
  'sil.research': 'Research',
  'sil.search': 'Web Search',
  'sil.queries': 'Query Generation',
  'sil.answers': 'Query Answering',
  'sil.judge': 'Consensus Judging',
  'sil.classify': 'Fact Classification',
  'sil.cluster': 'Entity Clustering',
  'sil.cluster_judge': 'Cluster Judging',
  'aml': 'Review Generation',
  'aml.persona': 'Persona Generation',
  'aml.patterns': 'Writing Patterns',
  'aml.structures': 'Structure Variants',
  'rde.subject': 'Subject Context',
  'rde.reviewer': 'Reviewer Context',
  'rde.query': 'Subject Query',
  'rde.domain': 'Domain Extraction',
  'rde.region': 'Region Extraction',
  'heuristic': 'Heuristic Generation',
}

function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toLocaleString()
}

function formatCost(usd: number): string {
  if (usd === 0) return '$0.00'
  if (usd < 0.01) return `$${usd.toFixed(4)}`
  return `$${usd.toFixed(2)}`
}

function TokensViewer() {
  const settings = useQuery(api.settings.get)
  const [jobs, setJobs] = useState<JobInfo[]>([])
  const [selectedJob, setSelectedJob] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [jobsLoading, setJobsLoading] = useState(false)
  const [tokens, setTokens] = useState<TokensReport | null>(null)
  const [expandedComponents, setExpandedComponents] = useState<Set<string>>(new Set())
  const [costMode, setCostMode] = useState<'estimated' | 'actual'>('estimated')
  const [actualCosts, setActualCosts] = useState<Record<string, number>>({})
  const [actualCostsLoading, setActualCostsLoading] = useState(false)

  // Fetch OpenRouter models for pricing data (same endpoint the LLM selector uses)
  const { models: orModels } = useOpenRouterModels()

  // Build pricing lookup: { modelId: { prompt: perTokenCost, completion: perTokenCost } }
  const pricingMap = useMemo(() => {
    const map: Record<string, { prompt: number; completion: number }> = {}
    for (const m of orModels) {
      const promptPrice = parseFloat(m.pricing.prompt) || 0
      const completionPrice = parseFloat(m.pricing.completion) || 0
      if (promptPrice > 0 || completionPrice > 0) {
        map[m.id] = { prompt: promptPrice, completion: completionPrice }
      }
    }
    return map
  }, [orModels])

  // Compute estimated cost for a single record using pricing map
  const getEstimatedCost = useCallback((promptTokens: number, completionTokens: number, model: string) => {
    // Try exact match, then strip :online suffix
    const pricing = pricingMap[model] || pricingMap[model.replace(':online', '')]
    if (!pricing) return 0
    return promptTokens * pricing.prompt + completionTokens * pricing.completion
  }, [pricingMap])

  // Get cost for a record (actual if available, otherwise estimated)
  const getRecordCost = useCallback((r: { prompt_tokens: number; completion_tokens: number; model: string; generation_id?: string }) => {
    if (costMode === 'actual' && r.generation_id && actualCosts[r.generation_id] !== undefined) {
      return actualCosts[r.generation_id]
    }
    return getEstimatedCost(r.prompt_tokens, r.completion_tokens, r.model)
  }, [costMode, actualCosts, getEstimatedCost])

  // Fetch actual costs from OpenRouter generation details
  const fetchActualCosts = useCallback(async () => {
    if (!tokens) return
    const genIds = tokens.records.map(r => r.generation_id).filter((id): id is string => !!id)
    if (genIds.length === 0) {
      toast.error('No generation IDs found. Re-run job to capture them.')
      setCostMode('estimated')
      return
    }
    setActualCostsLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/fetch-actual-costs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ generationIds: genIds }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      if (data.error) {
        toast.error(`Failed to fetch actual costs: ${data.error}`)
        setCostMode('estimated')
        return
      }
      setActualCosts(data.costs || {})
      const found = Object.keys(data.costs || {}).length
      toast.success(`Fetched actual costs for ${found}/${genIds.length} generations`)
    } catch {
      toast.error('Failed to fetch actual costs from OpenRouter')
      setCostMode('estimated')
    } finally {
      setActualCostsLoading(false)
    }
  }, [tokens])

  // Enrich tokens report with computed costs (estimated from pricing or actual from OpenRouter)
  const enrichedTokens = useMemo(() => {
    if (!tokens) return tokens
    // In estimated mode, need pricing data; in actual mode, need actual costs
    if (costMode === 'estimated' && Object.keys(pricingMap).length === 0) return tokens
    // Compute total cost from raw records
    let totalCost = 0
    for (const r of tokens.records) {
      totalCost += getRecordCost(r)
    }
    // Compute by_phase costs
    const enrichedByPhase: typeof tokens.by_phase = {}
    for (const [phase, phaseData] of Object.entries(tokens.by_phase)) {
      const enrichedComponents: typeof phaseData.components = {}
      let phaseCost = 0
      for (const [comp, compData] of Object.entries(phaseData.components)) {
        let compCost = 0
        for (const r of tokens.records) {
          if (r.component === comp) {
            compCost += getRecordCost(r)
          }
        }
        enrichedComponents[comp] = { ...compData, estimated_cost_usd: compCost }
        phaseCost += compCost
      }
      enrichedByPhase[phase] = { ...phaseData, estimated_cost_usd: phaseCost, components: enrichedComponents }
    }
    // Compute by_model costs
    const enrichedByModel: typeof tokens.by_model = {}
    for (const [model, modelData] of Object.entries(tokens.by_model)) {
      const pricing = pricingMap[model] || pricingMap[model.replace(':online', '')]
      let cost = 0
      for (const r of tokens.records) {
        if (r.model === model) {
          cost += getRecordCost(r)
        }
      }
      enrichedByModel[model] = {
        ...modelData,
        estimated_cost_usd: cost,
        pricing: pricing ? { prompt_per_token: pricing.prompt, completion_per_token: pricing.completion } : undefined,
      }
    }
    // Compute by_target costs
    const enrichedByTarget: typeof tokens.by_target = {}
    for (const [target, targetData] of Object.entries(tokens.by_target)) {
      let targetCost = 0
      for (const r of tokens.records) {
        if (r.target === target) {
          targetCost += getRecordCost(r)
        }
      }
      const enrichedRuns: typeof targetData.runs = {}
      if (targetData.runs) {
        for (const [run, runData] of Object.entries(targetData.runs)) {
          let runCost = 0
          for (const r of tokens.records) {
            if (r.target === target && r.run === run) {
              runCost += getRecordCost(r)
            }
          }
          enrichedRuns[run] = { ...runData, estimated_cost_usd: runCost }
        }
      }
      enrichedByTarget[target] = { ...targetData, estimated_cost_usd: targetCost, runs: Object.keys(enrichedRuns).length > 0 ? enrichedRuns : targetData.runs }
    }
    return {
      ...tokens,
      totals: { ...tokens.totals, estimated_cost_usd: totalCost },
      by_phase: enrichedByPhase,
      by_model: enrichedByModel,
      by_target: enrichedByTarget,
    }
  }, [tokens, pricingMap, getRecordCost, costMode])

  const fetchJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const jobsDir = settings?.jobsDirectory || './jobs'
      const res = await fetch(`${PYTHON_API_URL}/api/jobs-list?jobs_directory=${encodeURIComponent(jobsDir)}`)
      const data = await res.json()
      const jobsList = data.jobs || []
      setJobs(jobsList)
      const jobsWithTokens = jobsList.filter((j: JobInfo) => j.hasTokens)
      if (jobsWithTokens.length > 0 && !selectedJob) {
        const firstJob = jobsWithTokens[0]
        setSelectedJob(firstJob.path)
        loadTokens(firstJob.path)
      }
    } catch {
      toast.error('Failed to load jobs list')
    } finally {
      setJobsLoading(false)
    }
  }, [settings?.jobsDirectory, selectedJob])

  useEffect(() => {
    if (settings?.jobsDirectory && jobs.length === 0) {
      fetchJobs()
    }
  }, [settings?.jobsDirectory])

  const loadTokens = async (jobDir: string) => {
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/read-tokens`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobDir }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      if (data.found && data.tokens) {
        setTokens(data.tokens)
      } else {
        setTokens(null)
        toast.error('No token data found for this job')
      }
    } catch (err) {
      toast.error('Failed to load token data')
      setTokens(null)
    } finally {
      setLoading(false)
    }
  }

  const toggleComponent = (key: string) => {
    setExpandedComponents(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  // Get per-component model breakdown from raw records (with live pricing or actual costs)
  const getComponentModelBreakdown = (component: string) => {
    if (!tokens) return {}
    const breakdown: Record<string, { prompt_tokens: number; completion_tokens: number; total_tokens: number; total_calls: number; estimated_cost_usd: number }> = {}
    for (const r of tokens.records) {
      if (r.component !== component) continue
      const model = r.model || 'unknown'
      if (!breakdown[model]) breakdown[model] = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, total_calls: 0, estimated_cost_usd: 0 }
      breakdown[model].prompt_tokens += r.prompt_tokens
      breakdown[model].completion_tokens += r.completion_tokens
      breakdown[model].total_tokens += r.total_tokens
      breakdown[model].total_calls += 1
      breakdown[model].estimated_cost_usd += getRecordCost(r)
    }
    return breakdown
  }

  const jobsWithTokens = jobs.filter(j => j.hasTokens)

  // Use enriched tokens (with computed costs) for display
  const displayTokens = enrichedTokens

  // Determine which phase sections to show
  const compositionPhase = displayTokens?.by_phase?.composition
  const generationPhase = displayTokens?.by_phase?.generation

  // Split composition components into SIL+MAV vs RDE
  const silComponents: [string, typeof compositionPhase extends undefined ? never : NonNullable<typeof compositionPhase>['components'][string]][] = []
  const rdeComponents: [string, typeof compositionPhase extends undefined ? never : NonNullable<typeof compositionPhase>['components'][string]][] = []
  if (compositionPhase?.components) {
    for (const [key, val] of Object.entries(compositionPhase.components)) {
      if (key.startsWith('sil.')) silComponents.push([key, val])
      else if (key.startsWith('rde.')) rdeComponents.push([key, val])
    }
  }

  // Sort SIL components by pipeline execution order
  silComponents.sort((a, b) => {
    const ai = SIL_COMPONENT_ORDER.indexOf(a[0])
    const bi = SIL_COMPONENT_ORDER.indexOf(b[0])
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi)
  })

  // Get all generation component keys
  const genComponentKeys = generationPhase ? Object.keys(generationPhase.components || {}) : []

  // Expand all / collapse all helpers
  const expandAllSil = () => {
    setExpandedComponents(prev => {
      const next = new Set(prev)
      for (const [key] of silComponents) next.add(key)
      return next
    })
  }
  const collapseAllSil = () => {
    setExpandedComponents(prev => {
      const next = new Set(prev)
      for (const [key] of silComponents) next.delete(key)
      return next
    })
  }
  const allSilExpanded = silComponents.length > 0 && silComponents.every(([key]) => expandedComponents.has(key))

  const expandAllGen = () => {
    setExpandedComponents(prev => {
      const next = new Set(prev)
      for (const key of genComponentKeys) next.add(key)
      return next
    })
  }
  const collapseAllGen = () => {
    setExpandedComponents(prev => {
      const next = new Set(prev)
      for (const key of genComponentKeys) next.delete(key)
      return next
    })
  }
  const allGenExpanded = genComponentKeys.length > 0 && genComponentKeys.every(key => expandedComponents.has(key))

  // Detect RDE model from records
  const rdeModel = useMemo(() => {
    if (!tokens) return null
    const rdeRecords = tokens.records.filter(r => r.component.startsWith('rde.'))
    if (rdeRecords.length === 0) return null
    const models = [...new Set(rdeRecords.map(r => r.model))]
    return models.length === 1 ? models[0] : models.join(', ')
  }, [tokens])

  return (
    <div className="space-y-6">
      {/* Job Selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Token Usage</CardTitle>
          <CardDescription>View LLM token usage and estimated costs per pipeline phase</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <Select
                value={selectedJob}
                onValueChange={(val) => {
                  setSelectedJob(val)
                  setTokens(null)
                  setActualCosts({})
                  setCostMode('estimated')
                  loadTokens(val)
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder={jobsLoading ? 'Loading jobs...' : 'Select a job'} />
                </SelectTrigger>
                <SelectContent>
                  {jobsWithTokens.map((job) => (
                    <SelectItem key={job.path} value={job.path}>
                      {job.jobName} ({job.dirName})
                    </SelectItem>
                  ))}
                  {jobsWithTokens.length === 0 && (
                    <SelectItem value="_none" disabled>No jobs with token data</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <Button variant="outline" size="sm" onClick={fetchJobs} disabled={jobsLoading}>
              {jobsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {!loading && !displayTokens && selectedJob && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No token data available for this job.
          </CardContent>
        </Card>
      )}

      {!loading && displayTokens && (
        <>
          {/* Cost Mode Toggle */}
          <div className="flex items-center gap-2 justify-end">
            <span className="text-xs text-muted-foreground">Cost Mode:</span>
            <div className="flex rounded-md border">
              <button
                className={`px-3 py-1 text-xs font-medium rounded-l-md transition-colors ${
                  costMode === 'estimated'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background hover:bg-muted'
                }`}
                onClick={() => setCostMode('estimated')}
              >
                Estimated
              </button>
              <button
                className={`px-3 py-1 text-xs font-medium rounded-r-md transition-colors ${
                  costMode === 'actual'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-background hover:bg-muted'
                }`}
                onClick={() => {
                  setCostMode('actual')
                  if (Object.keys(actualCosts).length === 0) fetchActualCosts()
                }}
                disabled={actualCostsLoading}
              >
                {actualCostsLoading ? <Loader2 className="h-3 w-3 animate-spin inline mr-1" /> : null}
                Actual
              </button>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{formatTokenCount(displayTokens.totals.total_tokens)}</div>
                <p className="text-xs text-muted-foreground mt-1">
                  {formatTokenCount(displayTokens.totals.prompt_tokens)} in / {formatTokenCount(displayTokens.totals.completion_tokens)} out
                </p>
                <p className="text-sm text-muted-foreground mt-1">Total Tokens</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{displayTokens.totals.total_calls.toLocaleString()}</div>
                <p className="text-sm text-muted-foreground mt-1">Total Calls</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{formatCost(displayTokens.totals.estimated_cost_usd)}</div>
                <p className="text-sm text-muted-foreground mt-1">
                  {costMode === 'actual' ? 'Actual Cost' : 'Estimated Cost'}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{Object.keys(displayTokens.by_model).length}</div>
                <p className="text-sm text-muted-foreground mt-1">Models Used</p>
              </CardContent>
            </Card>
          </div>

          {/* SIL+MAV Section */}
          {silComponents.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">SIL+MAV (Composition)</CardTitle>
                    <CardDescription>
                      {formatTokenCount(compositionPhase!.total_tokens)} tokens | {compositionPhase!.total_calls} calls | {formatCost(compositionPhase!.estimated_cost_usd)}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs"
                    onClick={allSilExpanded ? collapseAllSil : expandAllSil}
                  >
                    {allSilExpanded ? 'Collapse All' : 'Expand All'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left p-2 font-medium w-[280px]">Component</th>
                        <th className="text-right p-2 font-medium">Input</th>
                        <th className="text-right p-2 font-medium">Output</th>
                        <th className="text-right p-2 font-medium">Total</th>
                        <th className="text-right p-2 font-medium">Calls</th>
                        <th className="text-right p-2 font-medium">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {silComponents.map(([key, val]) => {
                        const isExpanded = expandedComponents.has(key)
                        const modelBreakdown = isExpanded ? getComponentModelBreakdown(key) : {}
                        return (
                          <React.Fragment key={key}>
                            <tr
                              className="border-b hover:bg-muted/30 cursor-pointer"
                              onClick={() => toggleComponent(key)}
                            >
                              <td className="p-2 flex items-center gap-1">
                                {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                {COMPONENT_LABELS[key] || key}
                              </td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.prompt_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.completion_tokens)}</td>
                              <td className="text-right p-2 tabular-nums font-medium">{formatTokenCount(val.total_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{val.total_calls}</td>
                              <td className="text-right p-2 tabular-nums">{formatCost(val.estimated_cost_usd)}</td>
                            </tr>
                            {isExpanded && Object.entries(modelBreakdown).map(([model, mval]) => (
                              <tr key={`${key}-${model}`} className="border-b bg-muted/10">
                                <td className="p-2 pl-8 text-muted-foreground text-xs">{model.split('/').pop()}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.prompt_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.completion_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.total_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{mval.total_calls}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatCost(mval.estimated_cost_usd)}</td>
                              </tr>
                            ))}
                          </React.Fragment>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* RDE Section */}
          {rdeComponents.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">RDE (Reference Dataset Extractor)</CardTitle>
                <CardDescription>
                  Context extraction from reference dataset
                  {rdeModel && (
                    <> &mdash; Model: <Badge variant="outline" className="font-mono text-xs ml-1">{rdeModel.split('/').pop()}</Badge></>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left p-2 font-medium w-[280px]">Component</th>
                        <th className="text-right p-2 font-medium">Input</th>
                        <th className="text-right p-2 font-medium">Output</th>
                        <th className="text-right p-2 font-medium">Total</th>
                        <th className="text-right p-2 font-medium">Calls</th>
                        <th className="text-right p-2 font-medium">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rdeComponents.map(([key, val]) => (
                        <tr key={key} className="border-b hover:bg-muted/30">
                          <td className="p-2">{COMPONENT_LABELS[key] || key}</td>
                          <td className="text-right p-2 tabular-nums">{formatTokenCount(val.prompt_tokens)}</td>
                          <td className="text-right p-2 tabular-nums">{formatTokenCount(val.completion_tokens)}</td>
                          <td className="text-right p-2 tabular-nums font-medium">{formatTokenCount(val.total_tokens)}</td>
                          <td className="text-right p-2 tabular-nums">{val.total_calls}</td>
                          <td className="text-right p-2 tabular-nums">{formatCost(val.estimated_cost_usd)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* GenLLM Section */}
          {generationPhase && Object.keys(generationPhase.components || {}).length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">
                      {displayTokens.method === 'heuristic' ? 'Heuristic Generation' : 'GenLLM (Generation)'}
                    </CardTitle>
                    <CardDescription>
                      {formatTokenCount(generationPhase.total_tokens)} tokens | {generationPhase.total_calls} calls | {formatCost(generationPhase.estimated_cost_usd)}
                    </CardDescription>
                  </div>
                  {genComponentKeys.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-xs"
                      onClick={allGenExpanded ? collapseAllGen : expandAllGen}
                    >
                      {allGenExpanded ? 'Collapse All' : 'Expand All'}
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left p-2 font-medium w-[280px]">Component</th>
                        <th className="text-right p-2 font-medium">Input</th>
                        <th className="text-right p-2 font-medium">Output</th>
                        <th className="text-right p-2 font-medium">Total</th>
                        <th className="text-right p-2 font-medium">Calls</th>
                        <th className="text-right p-2 font-medium">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(generationPhase.components).map(([key, val]) => {
                        const isExpanded = expandedComponents.has(key)
                        const modelBreakdown = isExpanded ? getComponentModelBreakdown(key) : {}
                        return (
                          <React.Fragment key={key}>
                            <tr
                              className="border-b hover:bg-muted/30 cursor-pointer"
                              onClick={() => toggleComponent(key)}
                            >
                              <td className="p-2 flex items-center gap-1">
                                {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                {COMPONENT_LABELS[key] || key}
                              </td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.prompt_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.completion_tokens)}</td>
                              <td className="text-right p-2 tabular-nums font-medium">{formatTokenCount(val.total_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{val.total_calls}</td>
                              <td className="text-right p-2 tabular-nums">{formatCost(val.estimated_cost_usd)}</td>
                            </tr>
                            {isExpanded && Object.entries(modelBreakdown).map(([model, mval]) => (
                              <tr key={`${key}-${model}`} className="border-b bg-muted/10">
                                <td className="p-2 pl-8 text-muted-foreground text-xs">{model.split('/').pop()}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.prompt_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.completion_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(mval.total_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{mval.total_calls}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatCost(mval.estimated_cost_usd)}</td>
                              </tr>
                            ))}
                          </React.Fragment>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Model Breakdown */}
          {Object.keys(displayTokens.by_model).length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Model Breakdown</CardTitle>
                <CardDescription>Token usage aggregated by model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left p-2 font-medium">Model</th>
                        <th className="text-right p-2 font-medium">Input</th>
                        <th className="text-right p-2 font-medium">Output</th>
                        <th className="text-right p-2 font-medium">Total</th>
                        <th className="text-right p-2 font-medium">Calls</th>
                        <th className="text-right p-2 font-medium">Cost</th>
                        <th className="text-right p-2 font-medium">Pricing (per 1M)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(displayTokens.by_model).map(([model, val]) => {
                        const p = val.pricing
                        const pricingStr = p
                          ? `$${(p.prompt_per_token * 1_000_000).toFixed(2)} / $${(p.completion_per_token * 1_000_000).toFixed(2)}`
                          : '—'
                        return (
                          <tr key={model} className="border-b hover:bg-muted/30">
                            <td className="p-2">
                              <Badge variant="outline" className="font-mono text-xs">{model.split('/').pop()}</Badge>
                            </td>
                            <td className="text-right p-2 tabular-nums">{formatTokenCount(val.prompt_tokens)}</td>
                            <td className="text-right p-2 tabular-nums">{formatTokenCount(val.completion_tokens)}</td>
                            <td className="text-right p-2 tabular-nums font-medium">{formatTokenCount(val.total_tokens)}</td>
                            <td className="text-right p-2 tabular-nums">{val.total_calls}</td>
                            <td className="text-right p-2 tabular-nums">{formatCost(val.estimated_cost_usd)}</td>
                            <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{pricingStr}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                    <tfoot>
                      <tr className="bg-muted/30 font-medium">
                        <td className="p-2">Total</td>
                        <td className="text-right p-2 tabular-nums">{formatTokenCount(displayTokens.totals.prompt_tokens)}</td>
                        <td className="text-right p-2 tabular-nums">{formatTokenCount(displayTokens.totals.completion_tokens)}</td>
                        <td className="text-right p-2 tabular-nums">{formatTokenCount(displayTokens.totals.total_tokens)}</td>
                        <td className="text-right p-2 tabular-nums">{displayTokens.totals.total_calls}</td>
                        <td className="text-right p-2 tabular-nums">{formatCost(displayTokens.totals.estimated_cost_usd)}</td>
                        <td className="text-right p-2"></td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Target Breakdown */}
          {Object.keys(displayTokens.by_target).length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Target Breakdown</CardTitle>
                <CardDescription>Token usage by dataset target size</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="text-left p-2 font-medium">Target</th>
                        <th className="text-right p-2 font-medium">Input</th>
                        <th className="text-right p-2 font-medium">Output</th>
                        <th className="text-right p-2 font-medium">Total</th>
                        <th className="text-right p-2 font-medium">Calls</th>
                        <th className="text-right p-2 font-medium">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(displayTokens.by_target).map(([target, val]) => {
                        const isExpanded = expandedComponents.has(`target-${target}`)
                        const runs = val.runs || {}
                        const hasRuns = Object.keys(runs).length > 1
                        return (
                          <React.Fragment key={target}>
                            <tr
                              className={`border-b hover:bg-muted/30 ${hasRuns ? 'cursor-pointer' : ''}`}
                              onClick={() => hasRuns && toggleComponent(`target-${target}`)}
                            >
                              <td className="p-2 flex items-center gap-1">
                                {hasRuns && (isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />)}
                                <Badge variant="secondary">{target}</Badge>
                              </td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.prompt_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{formatTokenCount(val.completion_tokens)}</td>
                              <td className="text-right p-2 tabular-nums font-medium">{formatTokenCount(val.total_tokens)}</td>
                              <td className="text-right p-2 tabular-nums">{val.total_calls}</td>
                              <td className="text-right p-2 tabular-nums">{formatCost(val.estimated_cost_usd)}</td>
                            </tr>
                            {isExpanded && Object.entries(runs).sort().map(([run, rval]) => (
                              <tr key={`${target}-${run}`} className="border-b bg-muted/10">
                                <td className="p-2 pl-8 text-muted-foreground text-xs">{run}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(rval.prompt_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(rval.completion_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatTokenCount(rval.total_tokens)}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{rval.total_calls}</td>
                                <td className="text-right p-2 tabular-nums text-xs text-muted-foreground">{formatCost(rval.estimated_cost_usd)}</td>
                              </tr>
                            ))}
                          </React.Fragment>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Metadata */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-6 text-xs text-muted-foreground">
                <span>Method: <Badge variant="outline" className="ml-1">{displayTokens.method}</Badge></span>
                <span>Job: {displayTokens.job_id}</span>
                <span>Generated: {new Date(displayTokens.generated_at).toLocaleString()}</span>
                <span>{displayTokens.records.length} raw records</span>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}


// ========================================
// Domain Patterns Manager Tab
// ========================================

interface DomainPattern {
  name: string
  keywords: string[]
}

function DomainPatternsManager() {
  const [patterns, setPatterns] = useState<DomainPattern[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [newPatternName, setNewPatternName] = useState('')
  const [newPatternKeywords, setNewPatternKeywords] = useState('')
  const [editName, setEditName] = useState('')
  const [editKeywords, setEditKeywords] = useState('')

  // Fetch patterns on mount
  useEffect(() => {
    const fetchPatterns = async () => {
      setLoading(true)
      try {
        const response = await fetch(`${PYTHON_API_URL}/api/domain-patterns`)
        if (response.ok) {
          const data = await response.json()
          setPatterns(data.domains || [])
        } else {
          toast.error('Failed to fetch domain patterns')
        }
      } catch (e) {
        toast.error('Failed to connect to API')
      } finally {
        setLoading(false)
      }
    }
    fetchPatterns()
  }, [])

  // Save patterns to API
  const savePatterns = async (newPatterns: DomainPattern[]) => {
    setSaving(true)
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/domain-patterns`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domains: newPatterns }),
      })
      if (response.ok) {
        const data = await response.json()
        setPatterns(data.domains || [])
        toast.success('Domain patterns saved')
      } else {
        toast.error('Failed to save domain patterns')
      }
    } catch (e) {
      toast.error('Failed to connect to API')
    } finally {
      setSaving(false)
    }
  }

  // Reset to defaults
  const resetToDefaults = async () => {
    setSaving(true)
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/domain-patterns/reset`, {
        method: 'POST',
      })
      if (response.ok) {
        const data = await response.json()
        setPatterns(data.domains || [])
        toast.success('Reset to defaults')
      } else {
        toast.error('Failed to reset patterns')
      }
    } catch (e) {
      toast.error('Failed to connect to API')
    } finally {
      setSaving(false)
    }
  }

  // Add new pattern
  const addPattern = () => {
    if (!newPatternName.trim()) {
      toast.error('Please enter a domain name')
      return
    }
    const keywords = newPatternKeywords.split(',').map(k => k.trim().toUpperCase()).filter(Boolean)
    if (keywords.length === 0) {
      toast.error('Please enter at least one keyword')
      return
    }
    const newPatterns = [...patterns, { name: newPatternName.trim(), keywords }]
    savePatterns(newPatterns)
    setNewPatternName('')
    setNewPatternKeywords('')
  }

  // Delete pattern
  const deletePattern = (index: number) => {
    const newPatterns = patterns.filter((_, i) => i !== index)
    savePatterns(newPatterns)
  }

  // Start editing
  const startEdit = (index: number) => {
    setEditingIndex(index)
    setEditName(patterns[index].name)
    setEditKeywords(patterns[index].keywords.join(', '))
  }

  // Save edit
  const saveEdit = () => {
    if (editingIndex === null) return
    if (!editName.trim()) {
      toast.error('Please enter a domain name')
      return
    }
    const keywords = editKeywords.split(',').map(k => k.trim().toUpperCase()).filter(Boolean)
    if (keywords.length === 0) {
      toast.error('Please enter at least one keyword')
      return
    }
    const newPatterns = [...patterns]
    newPatterns[editingIndex] = { name: editName.trim(), keywords }
    savePatterns(newPatterns)
    setEditingIndex(null)
  }

  // Cancel edit
  const cancelEdit = () => {
    setEditingIndex(null)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        <span className="ml-2 text-muted-foreground">Loading domain patterns...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Domain Patterns</CardTitle>
              <CardDescription>
                Manage domain inference patterns for Quick Stats. Keywords are matched against aspect categories.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={resetToDefaults}
              disabled={saving}
            >
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset to Defaults
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Existing patterns */}
          {patterns.map((pattern, index) => (
            <div key={index} className="flex items-start gap-3 p-3 rounded-lg border bg-muted/30">
              {editingIndex === index ? (
                // Edit mode
                <div className="flex-1 space-y-2">
                  <Input
                    placeholder="Domain name"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                  />
                  <Input
                    placeholder="Keywords (comma-separated)"
                    value={editKeywords}
                    onChange={(e) => setEditKeywords(e.target.value)}
                  />
                  <div className="flex gap-2">
                    <Button size="sm" onClick={saveEdit} disabled={saving}>
                      <Save className="h-4 w-4 mr-1" />
                      Save
                    </Button>
                    <Button size="sm" variant="outline" onClick={cancelEdit}>
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                // View mode
                <>
                  <div className="flex-1">
                    <div className="font-medium">{pattern.name}</div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {pattern.keywords.map((keyword, ki) => (
                        <Badge key={ki} variant="secondary" className="text-xs">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => startEdit(index)}
                      disabled={saving}
                    >
                      <Edit2 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deletePattern(index)}
                      disabled={saving}
                      className="text-red-500 hover:text-red-600"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </>
              )}
            </div>
          ))}

          {patterns.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              No domain patterns configured. Add one below or reset to defaults.
            </div>
          )}

          {/* Add new pattern */}
          <Separator />
          <div className="space-y-3">
            <Label className="text-base font-medium">Add New Pattern</Label>
            <div className="flex gap-3">
              <Input
                placeholder="Domain name (e.g., Restaurant)"
                value={newPatternName}
                onChange={(e) => setNewPatternName(e.target.value)}
                className="flex-1"
              />
              <Input
                placeholder="Keywords (comma-separated, e.g., FOOD, SERVICE, AMBIANCE)"
                value={newPatternKeywords}
                onChange={(e) => setNewPatternKeywords(e.target.value)}
                className="flex-[2]"
              />
              <Button onClick={addPattern} disabled={saving}>
                <Plus className="h-4 w-4 mr-1" />
                Add
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Keywords are automatically uppercased. A domain is matched if ANY keyword appears in the aspect categories.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
