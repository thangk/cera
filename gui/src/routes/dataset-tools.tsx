import { createFileRoute, useNavigate, useSearch } from '@tanstack/react-router'
import { useState } from 'react'
import { toast } from 'sonner'
import {
  ArrowRightLeft,
  Download,
  Search,
  Save,
  Filter,
  ChevronLeft,
  Layers,
  Scissors,
  Plus,
  X,
  FileText,
  AlertCircle,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Badge } from '../components/ui/badge'
import { FileDropZone } from '../components/ui/file-drop-zone'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import { Alert, AlertDescription } from '../components/ui/alert'
import { PYTHON_API_URL } from '../lib/api-urls'

type ToolId = 'extract-categories' | 'explicit-to-implicit' | 'convert-format' | 'merge' | 'subsample'

interface ToolDefinition {
  id: ToolId
  title: string
  description: string
  icon: LucideIcon
}

const TOOLS: ToolDefinition[] = [
  { id: 'extract-categories', title: 'Extract Aspect Categories', description: 'Scan a dataset to extract unique aspect categories', icon: Search },
  { id: 'explicit-to-implicit', title: 'Convert Explicit → Implicit', description: 'Strip target terms and offsets from explicit datasets', icon: ArrowRightLeft },
  { id: 'convert-format', title: 'Convert Format', description: 'Convert between JSONL, CSV, and SemEval XML', icon: ArrowRightLeft },
  { id: 'merge', title: 'Dataset Merger', description: 'Merge multiple datasets of the same format', icon: Layers },
  { id: 'subsample', title: 'Dataset Subsampler', description: 'Split a dataset into smaller non-overlapping parts', icon: Scissors },
]

export const Route = createFileRoute('/dataset-tools')({
  component: DatasetToolsPage,
  validateSearch: (search: Record<string, unknown>) => {
    const tool = search.tool as string | undefined
    const validTools: ToolId[] = ['extract-categories', 'explicit-to-implicit', 'convert-format', 'merge', 'subsample']
    return {
      tool: validTools.includes(tool as ToolId) ? (tool as ToolId) : undefined,
    }
  },
})

function DatasetToolsPage() {
  const navigate = useNavigate()
  const { tool: selectedTool } = useSearch({ from: '/dataset-tools' })

  const handleToolSelect = (toolId: ToolId) => {
    navigate({ to: '/dataset-tools', search: { tool: toolId } })
  }

  const handleBack = () => {
    navigate({ to: '/dataset-tools', search: { tool: undefined } })
  }

  return (
    <div className="flex flex-col gap-6 p-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dataset Tools</h1>
        <p className="text-muted-foreground">
          Utilities for converting, transforming, and analyzing ABSA datasets
        </p>
      </div>

      {selectedTool ? (
        <ToolContainer toolId={selectedTool} onBack={handleBack} />
      ) : (
        <ToolSelector onSelect={handleToolSelect} />
      )}
    </div>
  )
}

function ToolSelector({ onSelect }: { onSelect: (toolId: ToolId) => void }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {TOOLS.map((tool) => {
        const Icon = tool.icon
        return (
          <Card
            key={tool.id}
            className="cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => onSelect(tool.id)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Icon className="h-5 w-5 text-muted-foreground" />
                <CardTitle className="text-base">{tool.title}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">{tool.description}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

function ToolContainer({ toolId, onBack }: { toolId: ToolId; onBack: () => void }) {
  return (
    <div className="space-y-4">
      <Button variant="ghost" size="sm" onClick={onBack} className="gap-1">
        <ChevronLeft className="h-4 w-4" />
        Back to Tools
      </Button>

      {toolId === 'extract-categories' && <ExtractCategoriesSection />}
      {toolId === 'explicit-to-implicit' && <ExplicitToImplicitSection />}
      {toolId === 'convert-format' && <FormatConverterSection />}
      {toolId === 'merge' && <DatasetMergerSection />}
      {toolId === 'subsample' && <DatasetSubsamplerSection />}
    </div>
  )
}


function detectFormat(file: File, content: string): string {
  if (file.name.endsWith('.xml') || content.trimStart().startsWith('<?xml') || content.trimStart().startsWith('<Reviews')) {
    return 'xml'
  } else if (file.name.endsWith('.csv')) {
    return 'csv'
  }
  return 'jsonl'
}

// Pick a different target format than the source
function getDefaultTarget(source: string): string {
  if (source === 'xml') return 'jsonl'
  if (source === 'csv') return 'xml'
  return 'xml'
}

// Section 1: Convert Explicit → Implicit
function ExplicitToImplicitSection() {
  const [file, setFile] = useState<File | null>(null)
  const [content, setContent] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)
  const [detectedFormat, setDetectedFormat] = useState('')

  const loadFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    setContent(text)
    setResult('')
    setDetectedFormat(detectFormat(f, text))
  }

  const convert = async () => {
    if (!content) return
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/convert-explicit-to-implicit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, format: detectedFormat || 'auto' }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data.content)
      toast.success('Converted to implicit format')
    } catch (err: any) {
      toast.error(`Conversion failed: ${err.message}`)
    }
    setLoading(false)
  }

  const downloadResult = () => {
    if (!result) return
    const ext = detectedFormat === 'xml' ? '.xml' : detectedFormat === 'csv' ? '.csv' : '.jsonl'
    const blob = new Blob([result], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `reviews-implicit${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ArrowRightLeft className="h-5 w-5 text-muted-foreground" />
          <CardTitle>Convert Explicit → Implicit</CardTitle>
        </div>
        <CardDescription>
          Strip target terms and character offsets from an explicit dataset to create an implicit version
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileDropZone file={file} formatLabel={detectedFormat?.toUpperCase()} onFile={loadFile} />

        <div className="flex gap-2">
          <Button size="sm" onClick={convert} disabled={!content || loading}>
            {loading ? 'Converting...' : 'Convert to Implicit'}
          </Button>
          {result && (
            <Button variant="outline" size="sm" onClick={downloadResult}>
              <Download className="h-4 w-4 mr-1.5" />
              Download
            </Button>
          )}
        </div>

        {result && (
          <div className="rounded-md border bg-muted/30 p-3 max-h-40 overflow-auto">
            <pre className="text-xs whitespace-pre-wrap font-mono">{result.slice(0, 2000)}{result.length > 2000 ? '\n...' : ''}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Section 2: Convert Format
function FormatConverterSection() {
  const [file, setFile] = useState<File | null>(null)
  const [content, setContent] = useState('')
  const [sourceFormat, setSourceFormat] = useState('jsonl')
  const [targetFormat, setTargetFormat] = useState('xml')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)

  const loadFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    setContent(text)
    setResult('')

    const detected = detectFormat(f, text)
    setSourceFormat(detected)
    setTargetFormat(getDefaultTarget(detected))
  }

  const convert = async () => {
    if (!content) return
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/convert-format`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          source_format: sourceFormat,
          target_format: targetFormat,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data.content)
      toast.success(`Converted ${data.review_count} reviews to ${targetFormat.toUpperCase()}`)
    } catch (err: any) {
      toast.error(`Conversion failed: ${err.message}`)
    }
    setLoading(false)
  }

  const downloadResult = () => {
    if (!result) return
    const ext = targetFormat === 'xml' ? '.xml' : targetFormat === 'csv' ? '.csv' : '.jsonl'
    const blob = new Blob([result], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `reviews-converted${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ArrowRightLeft className="h-5 w-5 text-muted-foreground" />
          <CardTitle>Convert Format</CardTitle>
        </div>
        <CardDescription>
          Convert between JSONL, CSV, and SemEval XML formats. Use same-format conversion to normalize non-standard attributes.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileDropZone file={file} formatLabel={sourceFormat.toUpperCase()} onFile={loadFile} />

        <div className="flex items-end gap-3">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Source</Label>
            <Select value={sourceFormat} onValueChange={setSourceFormat}>
              <SelectTrigger className="w-[110px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="jsonl">JSONL</SelectItem>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="xml">XML</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center h-8">
            <ArrowRightLeft className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Target</Label>
            <Select value={targetFormat} onValueChange={setTargetFormat}>
              <SelectTrigger className="w-[110px] h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="jsonl">JSONL</SelectItem>
                <SelectItem value="csv">CSV</SelectItem>
                <SelectItem value="xml">XML</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex gap-2">
          <Button size="sm" onClick={convert} disabled={!content || loading}>
            {loading ? 'Converting...' : 'Convert'}
          </Button>
          {result && (
            <Button variant="outline" size="sm" onClick={downloadResult}>
              <Download className="h-4 w-4 mr-1.5" />
              Download
            </Button>
          )}
        </div>

        {result && (
          <div className="rounded-md border bg-muted/30 p-3 max-h-40 overflow-auto">
            <pre className="text-xs whitespace-pre-wrap font-mono">{result.slice(0, 2000)}{result.length > 2000 ? '\n...' : ''}</pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Section 3: Extract Aspect Categories
function ExtractCategoriesSection() {
  const [file, setFile] = useState<File | null>(null)
  const [content, setContent] = useState('')
  const [categories, setCategories] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [presetName, setPresetName] = useState('')
  const [downloadFormat, setDownloadFormat] = useState<'xml' | 'jsonl' | 'csv'>('jsonl')

  // Domain pattern creation state
  const [domainPatternName, setDomainPatternName] = useState('')
  const [creatingDomainPattern, setCreatingDomainPattern] = useState(false)

  // Extract unique keywords from categories (e.g., "FOOD#QUALITY" → ["FOOD", "QUALITY"])
  const extractKeywordsFromCategories = (cats: string[]): string[] => {
    const keywords = new Set<string>()
    cats.forEach(cat => {
      // Split by # and other common separators
      const parts = cat.toUpperCase().split(/[#_\-\/]/)
      parts.forEach(part => {
        if (part.trim()) keywords.add(part.trim())
      })
    })
    return Array.from(keywords).sort()
  }

  // Create domain pattern from extracted categories
  const createDomainPattern = async (forceOverwrite = false) => {
    if (!domainPatternName.trim() || categories.length === 0) {
      toast.error('Please enter a domain name and extract categories first')
      return
    }

    setCreatingDomainPattern(true)
    try {
      // First, fetch existing patterns
      const getRes = await fetch(`${PYTHON_API_URL}/api/domain-patterns`)
      if (!getRes.ok) throw new Error('Failed to fetch existing patterns')
      const existingData = await getRes.json()

      // Check if pattern with same name already exists
      const existingIndex = existingData.domains.findIndex(
        (d: { name: string }) => d.name.toLowerCase() === domainPatternName.trim().toLowerCase()
      )

      if (existingIndex !== -1 && !forceOverwrite) {
        // Ask user for confirmation
        const confirmed = window.confirm(
          `A domain pattern named "${existingData.domains[existingIndex].name}" already exists.\n\nDo you want to overwrite it with the new keywords?`
        )
        if (!confirmed) {
          setCreatingDomainPattern(false)
          return
        }
      }

      // Extract keywords from categories
      const keywords = extractKeywordsFromCategories(categories)

      // Build new domains list (replace if exists, add if new)
      let newDomains
      if (existingIndex !== -1) {
        // Overwrite existing pattern
        newDomains = [...existingData.domains]
        newDomains[existingIndex] = { name: domainPatternName.trim(), keywords }
      } else {
        // Add new pattern
        newDomains = [
          ...existingData.domains,
          { name: domainPatternName.trim(), keywords }
        ]
      }

      // Save updated patterns
      const putRes = await fetch(`${PYTHON_API_URL}/api/domain-patterns`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domains: newDomains }),
      })

      if (!putRes.ok) throw new Error('Failed to save domain pattern')

      const action = existingIndex !== -1 ? 'Updated' : 'Created'
      toast.success(`${action} domain pattern "${domainPatternName}" with ${keywords.length} keywords`)
      setDomainPatternName('')
    } catch (err: any) {
      toast.error(`Failed to create domain pattern: ${err.message}`)
    } finally {
      setCreatingDomainPattern(false)
    }
  }

  const loadFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    setContent(text)
    setCategories([])
    setPresetName(f.name.replace(/\.(jsonl|csv|xml)$/i, ''))
  }

  const extract = async () => {
    if (!content) return
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/extract-aspect-categories`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, format: 'auto' }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setCategories(data.categories)
      toast.success(`Extracted ${data.count} unique aspect categories`)
    } catch (err: any) {
      toast.error(`Extraction failed: ${err.message}`)
    }
    setLoading(false)
  }

  const savePreset = async () => {
    if (!presetName || categories.length === 0) return
    setSaving(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/save-aspect-preset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: presetName,
          source: file?.name || '',
          categories,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      toast.success(`Saved preset: ${data.filename}`)
    } catch (err: any) {
      toast.error(`Save failed: ${err.message}`)
    }
    setSaving(false)
  }

  const downloadCategories = () => {
    if (categories.length === 0) return

    let output: string
    let ext: string
    let mimeType: string

    if (downloadFormat === 'xml') {
      output = `<?xml version="1.0" encoding="UTF-8"?>\n<AspectCategories source="${file?.name || 'unknown'}">\n${categories.map(c => `  <Category>${c}</Category>`).join('\n')}\n</AspectCategories>`
      ext = '.xml'
      mimeType = 'application/xml'
    } else if (downloadFormat === 'csv') {
      output = `category\n${categories.join('\n')}`
      ext = '.csv'
      mimeType = 'text/csv'
    } else {
      output = categories.map(c => JSON.stringify({ category: c })).join('\n')
      ext = '.jsonl'
      mimeType = 'application/jsonl'
    }

    const blob = new Blob([output], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const filename = presetName || 'aspect-categories'
    a.download = `${filename.replace(/[^a-zA-Z0-9-_]/g, '-')}${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Search className="h-5 w-5 text-muted-foreground" />
          <CardTitle>Extract Aspect Categories</CardTitle>
        </div>
        <CardDescription>
          Scan a real dataset (e.g., SemEval 2015) to extract all unique aspect categories and save as a preset
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileDropZone file={file} onFile={loadFile} />

        <Button size="sm" onClick={extract} disabled={!content || loading}>
          {loading ? 'Scanning...' : 'Scan & Extract'}
        </Button>

        {categories.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Found {categories.length} categories:</span>
            </div>
            <div className="flex flex-wrap gap-1.5 p-3 rounded-md border bg-muted/20">
              {categories.map((cat) => (
                <Badge key={cat} variant="secondary" className="text-xs">
                  {cat}
                </Badge>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <Input
                placeholder="Preset name (e.g., SemEval 2015 Restaurant)"
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                className="h-8 text-sm max-w-xs"
              />
              <Button variant="outline" size="sm" onClick={savePreset} disabled={!presetName || saving}>
                <Save className="h-4 w-4 mr-1.5" />
                {saving ? 'Saving...' : 'Save as Preset'}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Saved presets appear in the "Load from preset" dropdown when creating jobs.
            </p>

            {/* Create Domain Pattern Section */}
            <div className="space-y-2 pt-3 border-t">
              <Label className="text-sm font-medium">Create Domain Pattern</Label>
              <p className="text-xs text-muted-foreground">
                Use extracted categories as keywords for Quick Stats domain inference.
                Keywords extracted: {extractKeywordsFromCategories(categories).join(', ')}
              </p>
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Domain name (e.g., Restaurant)"
                  value={domainPatternName}
                  onChange={(e) => setDomainPatternName(e.target.value)}
                  className="h-8 text-sm max-w-xs"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => createDomainPattern()}
                  disabled={!domainPatternName || creatingDomainPattern}
                >
                  <Filter className="h-4 w-4 mr-1.5" />
                  {creatingDomainPattern ? 'Creating...' : 'Create Pattern'}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Manage patterns in Viewer Tools → Domain Patterns tab.
              </p>
            </div>

            <div className="flex items-center gap-2 pt-2 border-t">
              <Select value={downloadFormat} onValueChange={(v) => setDownloadFormat(v as 'xml' | 'jsonl' | 'csv')}>
                <SelectTrigger className="w-[100px] h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="jsonl">JSONL</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="xml">XML</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={downloadCategories}>
                <Download className="h-4 w-4 mr-1.5" />
                Download
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Section 4: Dataset Merger
function DatasetMergerSection() {
  const [files, setFiles] = useState<File[]>([])
  const [fileContents, setFileContents] = useState<string[]>([])
  const [detectedFormat, setDetectedFormat] = useState<string>('')
  const [formatError, setFormatError] = useState<string>('')
  const [result, setResult] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<{ totalRecords: number; filesCount: number } | null>(null)

  const addFile = async (f: File) => {
    const text = await f.text()
    const format = detectFormat(f, text)

    // Check format compatibility
    if (files.length > 0 && format !== detectedFormat) {
      setFormatError(`Format mismatch: "${f.name}" is ${format.toUpperCase()}, but existing files are ${detectedFormat.toUpperCase()}`)
      return
    }

    setFormatError('')
    if (files.length === 0) {
      setDetectedFormat(format)
    }

    setFiles([...files, f])
    setFileContents([...fileContents, text])
    setResult('')
    setStats(null)
  }

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    const newContents = fileContents.filter((_, i) => i !== index)
    setFiles(newFiles)
    setFileContents(newContents)
    setResult('')
    setStats(null)
    if (newFiles.length === 0) {
      setDetectedFormat('')
    }
  }

  const merge = async () => {
    if (fileContents.length < 2) return
    setLoading(true)
    try {
      const res = await fetch(`${PYTHON_API_URL}/api/merge-datasets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: fileContents, format: detectedFormat }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data.content)
      setStats({ totalRecords: data.total_records, filesCount: data.files_merged })
      toast.success(`Merged ${data.files_merged} files with ${data.total_records} total records`)
    } catch (err: any) {
      toast.error(`Merge failed: ${err.message}`)
    }
    setLoading(false)
  }

  const downloadResult = () => {
    if (!result) return
    const ext = detectedFormat === 'xml' ? '.xml' : detectedFormat === 'csv' ? '.csv' : '.jsonl'
    const blob = new Blob([result], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `merged-dataset${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-muted-foreground" />
          <CardTitle>Dataset Merger</CardTitle>
        </div>
        <CardDescription>
          Merge multiple datasets of the same format into a single file
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileDropZone
          onFile={addFile}
          accept=".jsonl,.csv,.xml"
          placeholder="Drop files here to add to merge queue"
          description="All files must be the same format (JSONL, CSV, or XML)"
        />

        {files.length > 0 && (
          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">
              Files to merge ({files.length}):
            </Label>
            <div className="space-y-1">
              {files.map((file, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 rounded border bg-muted/20">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">{file.name}</span>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => removeFile(idx)} className="h-7 w-7 p-0">
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
            {detectedFormat && (
              <Badge variant="secondary" className="text-xs">
                Format: {detectedFormat.toUpperCase()}
              </Badge>
            )}
          </div>
        )}

        {formatError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{formatError}</AlertDescription>
          </Alert>
        )}

        <div className="flex gap-2">
          <Button size="sm" onClick={merge} disabled={files.length < 2 || loading}>
            {loading ? 'Merging...' : 'Merge Datasets'}
          </Button>
          {result && (
            <Button variant="outline" size="sm" onClick={downloadResult}>
              <Download className="h-4 w-4 mr-1.5" />
              Download
            </Button>
          )}
        </div>

        {stats && (
          <div className="flex gap-4 p-3 rounded-md bg-muted/30">
            <div>
              <span className="text-xs text-muted-foreground">Files merged</span>
              <p className="font-medium">{stats.filesCount}</p>
            </div>
            <div>
              <span className="text-xs text-muted-foreground">Total records</span>
              <p className="font-medium">{stats.totalRecords}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="rounded-md border bg-muted/30 p-3 max-h-40 overflow-auto">
            <pre className="text-xs whitespace-pre-wrap font-mono">
              {result.slice(0, 2000)}{result.length > 2000 ? '\n...' : ''}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Section 5: Dataset Subsampler
function DatasetSubsamplerSection() {
  const [file, setFile] = useState<File | null>(null)
  const [content, setContent] = useState('')
  const [detectedFormat, setDetectedFormat] = useState('')
  const [totalRecords, setTotalRecords] = useState(0)
  const [totalSentences, setTotalSentences] = useState(0)
  const [splitMode, setSplitMode] = useState<'equal' | 'custom' | 'sample'>('equal')
  const [numSplits, setNumSplits] = useState(2)
  const [customSizes, setCustomSizes] = useState<number[]>([100, 100])
  const [setSizes, setSetSizes] = useState<number[]>([100, 500, 1000])
  const [sampleUnit, setSampleUnit] = useState<'review' | 'sentence'>('review')
  const [namePrefix, setNamePrefix] = useState('set')
  const [namePostfix, setNamePostfix] = useState('')
  const [results, setResults] = useState<{ name: string; content: string; count: number; sentence_count?: number }[]>([])
  const [loading, setLoading] = useState(false)

  const countRecords = (text: string, format: string): { reviews: number; sentences: number } => {
    let reviews = 0
    let sentences = 0
    if (format === 'jsonl') {
      const lines = text.trim().split('\n').filter(line => line.trim())
      reviews = lines.length
      for (const line of lines) {
        try {
          const obj = JSON.parse(line)
          sentences += (obj.sentences || []).length
        } catch { /* skip */ }
      }
    } else if (format === 'csv') {
      const lines = text.trim().split('\n')
      // CSV rows are opinion-level; count unique review_ids and sentence_ids
      const reviewIds = new Set<string>()
      const sentenceIds = new Set<string>()
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',')
        if (cols[0]) reviewIds.add(cols[0])
        if (cols[1]) sentenceIds.add(cols[1])
      }
      reviews = reviewIds.size
      sentences = sentenceIds.size
    } else if (format === 'xml') {
      const reviewMatches = text.match(/<Review\s/g)
      const sentenceMatches = text.match(/<sentence\s/g)
      reviews = reviewMatches ? reviewMatches.length : 0
      sentences = sentenceMatches ? sentenceMatches.length : 0
    }
    return { reviews, sentences }
  }

  const loadFile = async (f: File) => {
    setFile(f)
    const text = await f.text()
    setContent(text)
    const format = detectFormat(f, text)
    setDetectedFormat(format)
    const counts = countRecords(text, format)
    setTotalRecords(counts.reviews)
    setTotalSentences(counts.sentences)
    setResults([])
  }

  const totalRequested = splitMode === 'custom' ? customSizes.reduce((a, b) => a + b, 0) : totalRecords
  const maxSampleSet = splitMode === 'sample'
    ? Math.max(...setSizes)
    : 0
  const sampleExceeds = splitMode === 'sample' && (
    sampleUnit === 'review'
      ? maxSampleSet > totalRecords
      : maxSampleSet > totalSentences
  )

  const subsample = async () => {
    if (!content) return
    setLoading(true)
    try {
      const body: Record<string, unknown> = {
        content,
        format: detectedFormat,
        mode: splitMode,
      }
      if (splitMode === 'equal') {
        body.num_splits = numSplits
      } else if (splitMode === 'custom') {
        body.split_sizes = customSizes
      } else if (splitMode === 'sample') {
        body.set_sizes = setSizes
        body.unit = sampleUnit
        body.name_prefix = namePrefix || 'set'
        body.name_postfix = namePostfix || undefined
      }

      const res = await fetch(`${PYTHON_API_URL}/api/subsample-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResults(data.splits)
      toast.success(`Created ${data.splits.length} ${splitMode === 'sample' ? 'sets' : 'splits'}`)
    } catch (err: any) {
      toast.error(`Subsample failed: ${err.message}`)
    }
    setLoading(false)
  }

  const downloadSplit = (index: number) => {
    const split = results[index]
    if (!split) return
    const ext = detectedFormat === 'xml' ? '.xml' : detectedFormat === 'csv' ? '.csv' : '.jsonl'
    const blob = new Blob([split.content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${split.name}${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadAllAsZip = async () => {
    const JSZip = (await import('jszip')).default
    const zip = new JSZip()
    const ext = detectedFormat === 'xml' ? '.xml' : detectedFormat === 'csv' ? '.csv' : '.jsonl'
    for (const split of results) {
      zip.file(`${split.name}${ext}`, split.content)
    }
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file?.name?.replace(/\.[^.]+$/, '') || 'dataset'}-${splitMode}-sets.zip`
    a.click()
    URL.revokeObjectURL(url)
  }

  const addCustomSplit = () => {
    setCustomSizes([...customSizes, 100])
  }

  const removeCustomSplit = (index: number) => {
    if (customSizes.length <= 2) return
    setCustomSizes(customSizes.filter((_, i) => i !== index))
  }

  const updateCustomSize = (index: number, value: number) => {
    const newSizes = [...customSizes]
    newSizes[index] = value
    setCustomSizes(newSizes)
  }

  const addSet = () => {
    setSetSizes([...setSizes, 100])
  }

  const removeSet = (index: number) => {
    if (setSizes.length <= 1) return
    setSetSizes(setSizes.filter((_, i) => i !== index))
  }

  const updateSetSize = (index: number, value: number) => {
    const newSizes = [...setSizes]
    newSizes[index] = value
    setSetSizes(newSizes)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Scissors className="h-5 w-5 text-muted-foreground" />
          <CardTitle>Dataset Subsampler</CardTitle>
        </div>
        <CardDescription>
          Split a dataset into non-overlapping parts, or create random sample sets of varying sizes
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FileDropZone file={file} formatLabel={detectedFormat?.toUpperCase()} onFile={loadFile} />

        {totalRecords > 0 && (
          <div className="flex gap-2">
            <Badge variant="secondary">
              {totalRecords} reviews
            </Badge>
            <Badge variant="secondary">
              {totalSentences} sentences
            </Badge>
          </div>
        )}

        {content && (
          <div className="space-y-4 p-4 rounded-lg border bg-muted/20">
            <div className="space-y-2">
              <Label>Mode</Label>
              <Select value={splitMode} onValueChange={(v) => setSplitMode(v as 'equal' | 'custom' | 'sample')}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="equal">Equal Splits</SelectItem>
                  <SelectItem value="custom">Custom Splits</SelectItem>
                  <SelectItem value="sample">Sample Sets</SelectItem>
                </SelectContent>
              </Select>
              {splitMode === 'sample' && (
                <p className="text-xs text-muted-foreground">
                  Each set is an independent random sample — items may appear in multiple sets
                </p>
              )}
            </div>

            {splitMode === 'equal' && (
              <div className="space-y-2">
                <Label>Number of Splits</Label>
                <Input
                  type="number"
                  min={2}
                  max={10}
                  value={numSplits}
                  onChange={(e) => setNumSplits(parseInt(e.target.value) || 2)}
                  className="w-24"
                />
                <p className="text-xs text-muted-foreground">
                  Each split will have ~{Math.floor(totalRecords / numSplits)} records
                </p>
              </div>
            )}

            {splitMode === 'custom' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Custom Split Sizes</Label>
                  <Button variant="outline" size="sm" onClick={addCustomSplit}>
                    <Plus className="h-4 w-4 mr-1" />
                    Add Split
                  </Button>
                </div>
                <div className="space-y-2">
                  {customSizes.map((size, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground w-16">
                        Split {idx + 1}:
                      </span>
                      <Input
                        type="number"
                        min={1}
                        value={size}
                        onChange={(e) => updateCustomSize(idx, parseInt(e.target.value) || 1)}
                        className="w-24"
                      />
                      {customSizes.length > 2 && (
                        <Button variant="ghost" size="sm" onClick={() => removeCustomSplit(idx)} className="h-8 w-8 p-0">
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  Total requested: {totalRequested} / {totalRecords} available
                </p>
                {totalRequested > totalRecords && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Requested size exceeds available records
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            )}

            {splitMode === 'sample' && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Unit</Label>
                  <Select value={sampleUnit} onValueChange={(v) => setSampleUnit(v as 'review' | 'sentence')}>
                    <SelectTrigger className="w-[200px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="review">By Review</SelectItem>
                      <SelectItem value="sentence">By Sentence</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {sampleUnit === 'sentence'
                      ? `Randomly pick N sentences (varied reviews). ${totalSentences} available.`
                      : `Randomly pick N complete reviews. ${totalRecords} available.`}
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Set Sizes</Label>
                    <Button variant="outline" size="sm" onClick={addSet}>
                      <Plus className="h-4 w-4 mr-1" />
                      Add Set
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {setSizes.map((size, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground w-14">
                          Set {idx + 1}:
                        </span>
                        <Input
                          type="number"
                          min={1}
                          value={size}
                          onChange={(e) => updateSetSize(idx, parseInt(e.target.value) || 1)}
                          className="w-24"
                        />
                        <span className="text-xs text-muted-foreground">
                          {sampleUnit === 'sentence' ? 'sentences' : 'reviews'}
                        </span>
                        {setSizes.length > 1 && (
                          <Button variant="ghost" size="sm" onClick={() => removeSet(idx)} className="h-8 w-8 p-0">
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                  {sampleExceeds && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        A set size exceeds the available {sampleUnit === 'sentence' ? `${totalSentences} sentences` : `${totalRecords} reviews`}
                      </AlertDescription>
                    </Alert>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>File Naming</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      value={namePrefix}
                      onChange={(e) => setNamePrefix(e.target.value)}
                      placeholder="prefix"
                      className="w-28"
                    />
                    <span className="text-sm text-muted-foreground">-[SIZE]-</span>
                    <Input
                      value={namePostfix}
                      onChange={(e) => setNamePostfix(e.target.value)}
                      placeholder="postfix (optional)"
                      className="w-40"
                    />
                    <span className="text-sm text-muted-foreground">
                      .{detectedFormat || 'xml'}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Preview: {namePrefix || 'set'}-{setSizes[0] || 100}{namePostfix ? `-${namePostfix}` : ''}.{detectedFormat || 'xml'}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-2">
          <Button
            size="sm"
            onClick={subsample}
            disabled={!content || loading || (splitMode === 'custom' && totalRequested > totalRecords) || sampleExceeds}
          >
            {loading ? 'Processing...' : splitMode === 'sample' ? 'Create Sets' : 'Create Splits'}
          </Button>
        </div>

        {results.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Generated {splitMode === 'sample' ? 'Sets' : 'Splits'}:</Label>
              {results.length > 1 && (
                <Button variant="outline" size="sm" onClick={downloadAllAsZip}>
                  <Download className="h-4 w-4 mr-1" />
                  Download All (.zip)
                </Button>
              )}
            </div>
            <div className="space-y-1">
              {results.map((split, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 rounded border bg-muted/20">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">{split.name}</span>
                    <Badge variant="secondary" className="text-xs">
                      {split.count} reviews
                    </Badge>
                    {split.sentence_count != null && (
                      <Badge variant="outline" className="text-xs">
                        {split.sentence_count} sentences
                      </Badge>
                    )}
                  </div>
                  <Button variant="outline" size="sm" onClick={() => downloadSplit(idx)}>
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
