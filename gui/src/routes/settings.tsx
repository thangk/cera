import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation } from 'convex/react'
import { api } from 'convex/_generated/api'
import { useState, useEffect } from 'react'
import { toast } from 'sonner'
import { Save, Eye, EyeOff, Key, FolderOutput, Loader2, RefreshCw, Gauge, AlertCircle, CheckCircle2, Search, Trash2, Database, ToggleLeft, Layers, Plus, Edit2, Star, StarOff, X, ChevronDown, Server } from 'lucide-react'
import type { Id } from 'convex/_generated/dataModel'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Alert, AlertDescription } from '../components/ui/alert'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../components/ui/dialog'
import { Separator } from '../components/ui/separator'
import { Badge } from '../components/ui/badge'
import { Switch } from '../components/ui/switch'
import { useOpenRouterLimits } from '../hooks/use-openrouter-limits'
import { useOpenRouterModels } from '../hooks/use-openrouter-models'
import { useLocalLlmModels } from '../hooks/use-local-llm-models'
import { LLMSelector } from '../components/llm-selector'

export const Route = createFileRoute('/settings')({
  component: SettingsPage,
})

// Type for preset from Convex
type LLMPreset = {
  _id: Id<"llm_presets">
  name: string
  isDefault: boolean
  rdeModel?: string
  mavModels?: string[]
  savModel?: string
  genModel?: string
  createdAt: number
  updatedAt: number
}

function SettingsPage() {
  const settings = useQuery(api.settings.get)
  const updateSettings = useMutation(api.settings.update)

  // LLM Presets
  const presets = useQuery(api.llmPresets.list)
  const createPreset = useMutation(api.llmPresets.create)
  const updatePreset = useMutation(api.llmPresets.update)
  const deletePreset = useMutation(api.llmPresets.remove)
  const setDefaultPreset = useMutation(api.llmPresets.setDefault)
  const clearDefaultPreset = useMutation(api.llmPresets.clearDefault)
  const { providers, groupedModels, processedModels, loading: modelsLoading } = useOpenRouterModels()
  const { models: localLlmModelsForPresets } = useLocalLlmModels()
  // Combine OpenRouter + local models for preset display/validation
  const allModels = [...processedModels, ...localLlmModelsForPresets]

  // Preset dialog state
  const [presetDialogOpen, setPresetDialogOpen] = useState(false)
  const [editingPreset, setEditingPreset] = useState<LLMPreset | null>(null)
  const [presetForm, setPresetForm] = useState({
    name: '',
    isDefault: false,
    rdeModel: '',
    mavModels: ['', '', ''],
    savModel: '',
    genModel: '',
  })
  const [savingPreset, setSavingPreset] = useState(false)
  const [expandedSection, setExpandedSection] = useState<'rde' | 'mav' | 'sav' | 'gen' | null>(null)

  const [apiKey, setApiKey] = useState('')
  const [tavilyApiKey, setTavilyApiKey] = useState('')
  const [tavilyEnabled, setTavilyEnabled] = useState(true)
  const [convexAdminKey, setConvexAdminKey] = useState('')
  const { keyInfo, loading: limitsLoading, error: limitsError, refetch: refetchLimits } = useOpenRouterLimits(apiKey)
  const [showApiKey, setShowApiKey] = useState(false)
  const [showTavilyApiKey, setShowTavilyApiKey] = useState(false)
  const [showConvexAdminKey, setShowConvexAdminKey] = useState(false)
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('dark')
  const [jobsDir, setJobsDir] = useState('./jobs')
  const [saving, setSaving] = useState(false)

  // Local LLMs state
  const [localLlmEnabled, setLocalLlmEnabled] = useState(false)
  const [localLlmEndpoint, setLocalLlmEndpoint] = useState('')
  const [localLlmApiKey, setLocalLlmApiKey] = useState('')
  const [showLocalLlmApiKey, setShowLocalLlmApiKey] = useState(false)
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [connectionError, setConnectionError] = useState('')
  const [localModels, setLocalModels] = useState<string[]>([])

  // Reset preset form when dialog opens/closes or editing preset changes
  const openPresetDialog = (preset?: LLMPreset) => {
    if (preset) {
      setEditingPreset(preset)
      setPresetForm({
        name: preset.name,
        isDefault: preset.isDefault,
        rdeModel: preset.rdeModel || '',
        mavModels: [
          preset.mavModels?.[0] || '',
          preset.mavModels?.[1] || '',
          preset.mavModels?.[2] || '',
        ],
        savModel: preset.savModel || '',
        genModel: preset.genModel || '',
      })
    } else {
      setEditingPreset(null)
      setPresetForm({
        name: '',
        isDefault: false,
        rdeModel: '',
        mavModels: ['', '', ''],
        savModel: '',
        genModel: '',
      })
    }
    setPresetDialogOpen(true)
  }

  const handleSavePreset = async () => {
    if (!presetForm.name.trim()) {
      toast.error('Preset name is required')
      return
    }

    setSavingPreset(true)
    try {
      // Filter out empty MAV models but keep the structure
      const mavModels = presetForm.mavModels.filter(m => m)

      if (editingPreset) {
        await updatePreset({
          id: editingPreset._id,
          name: presetForm.name,
          isDefault: presetForm.isDefault,
          rdeModel: presetForm.rdeModel || undefined,
          mavModels: mavModels.length > 0 ? mavModels : undefined,
          savModel: presetForm.savModel || undefined,
          genModel: presetForm.genModel || undefined,
        })
        toast.success('Preset updated')
      } else {
        await createPreset({
          name: presetForm.name,
          isDefault: presetForm.isDefault,
          rdeModel: presetForm.rdeModel || undefined,
          mavModels: mavModels.length > 0 ? mavModels : undefined,
          savModel: presetForm.savModel || undefined,
          genModel: presetForm.genModel || undefined,
        })
        toast.success('Preset created')
      }
      setPresetDialogOpen(false)
    } catch {
      toast.error('Failed to save preset')
    } finally {
      setSavingPreset(false)
    }
  }

  // Helper to get model display name
  const getModelDisplay = (modelId: string | undefined) => {
    if (!modelId) return null
    const model = allModels.find(m => m.id === modelId)
    if (model) return model.name
    // Model not found - show as unavailable
    return <span className="text-destructive">{modelId.split('/').pop()} (unavailable)</span>
  }

  useEffect(() => {
    if (settings) {
      setApiKey(settings.openrouterApiKey ?? '')
      setTavilyApiKey(settings.tavilyApiKey ?? '')
      setTavilyEnabled(settings.tavilyEnabled ?? true)
      setConvexAdminKey(settings.convexAdminKey ?? '')
      setTheme(settings.theme)
      setJobsDir(settings.jobsDirectory ?? './jobs')
      setLocalLlmEnabled(settings.localLlmEnabled ?? false)
      setLocalLlmEndpoint(settings.localLlmEndpoint ?? '')
      setLocalLlmApiKey(settings.localLlmApiKey ?? '')
    }
  }, [settings])

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateSettings({
        openrouterApiKey: apiKey || undefined,
        tavilyApiKey: tavilyApiKey || undefined,
        tavilyEnabled,
        convexAdminKey: convexAdminKey || undefined,
        theme,
        jobsDirectory: jobsDir,
        localLlmEnabled,
        localLlmEndpoint: localLlmEndpoint || undefined,
        localLlmApiKey: localLlmApiKey || undefined,
      })
      toast.success('Settings saved successfully')
    } catch (error) {
      toast.error('Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  const handleTestConnection = async () => {
    if (!localLlmEndpoint) {
      toast.error('Enter an endpoint URL first')
      return
    }
    setTestingConnection(true)
    setConnectionStatus('idle')
    try {
      const url = localLlmEndpoint.replace(/\/+$/, '') + '/v1/models'
      const headers: Record<string, string> = {}
      if (localLlmApiKey) {
        headers['Authorization'] = `Bearer ${localLlmApiKey}`
      }
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(10000) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      const models = (data.data || []).map((m: { id: string }) => m.id)
      setLocalModels(models)
      setConnectionStatus('success')
      toast.success(`Connected! ${models.length} model(s) available.`)
    } catch (err) {
      setConnectionStatus('error')
      setConnectionError(err instanceof Error ? err.message : 'Connection failed')
      setLocalModels([])
      toast.error('Failed to connect to local LLM server')
    } finally {
      setTestingConnection(false)
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6 max-w-2xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure your CERA preferences and API keys
        </p>
      </div>

      {/* API Key */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Key className="h-5 w-5 text-muted-foreground" />
            <CardTitle>OpenRouter API Key</CardTitle>
          </div>
          <CardDescription>
            Your OpenRouter API key for accessing LLM providers. Get one at{' '}
            <a
              href="https://openrouter.ai/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              openrouter.ai/keys
            </a>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showApiKey ? 'text' : 'password'}
                placeholder="sk-or-v1-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                onClick={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? (
                  <EyeOff className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <Eye className="h-4 w-4 text-muted-foreground" />
                )}
              </Button>
            </div>
          </div>

          {/* API Key Status & Rate Limits */}
          {apiKey && (
            <>
              {limitsLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Checking API key...</span>
                </div>
              ) : limitsError ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{limitsError}</AlertDescription>
                </Alert>
              ) : keyInfo ? (
                <div className="rounded-lg border bg-muted/50 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium">API Key Valid</span>
                      {keyInfo.is_free_tier && (
                        <Badge variant="secondary" className="text-[10px]">Free Tier</Badge>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={refetchLimits}
                      className="h-7 px-2"
                    >
                      <RefreshCw className="h-3 w-3" />
                    </Button>
                  </div>

                  <Separator />

                  <div className="grid gap-3 sm:grid-cols-3 text-sm">
                    <div className="space-y-1">
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <Gauge className="h-3 w-3" />
                        <span>Rate Limit</span>
                      </div>
                      <div className="font-medium">
                        {keyInfo.rate_limit.requests} req / {keyInfo.rate_limit.interval}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-muted-foreground">Usage</div>
                      <div className="font-medium">
                        ${keyInfo.usage.toFixed(4)}
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-muted-foreground">Credit Limit</div>
                      <div className="font-medium">
                        {keyInfo.limit ? `$${keyInfo.limit.toFixed(2)}` : 'Unlimited'}
                      </div>
                    </div>
                  </div>

                  {keyInfo.label && (
                    <div className="text-xs text-muted-foreground">
                      Key: {keyInfo.label}
                    </div>
                  )}
                </div>
              ) : null}
            </>
          )}
        </CardContent>
      </Card>

      {/* Tavily API Key (Optional) */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Search className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Web Search Provider</CardTitle>
            <Badge variant="outline" className="text-[10px]">Optional</Badge>
          </div>
          <CardDescription>
            Configure web search for MAV verification. You can use Tavily (dedicated search API)
            or OpenRouter's native web search plugin (enabled in your OpenRouter account).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Toggle between Tavily and OpenRouter native search */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <ToggleLeft className="h-4 w-4 text-muted-foreground" />
                <Label htmlFor="tavily-toggle" className="font-medium">Use Tavily for Web Search</Label>
              </div>
              <p className="text-sm text-muted-foreground">
                {tavilyEnabled
                  ? "Tavily API will be used for web search (requires API key below)"
                  : "OpenRouter's native web search will be used instead"}
              </p>
            </div>
            <Switch
              id="tavily-toggle"
              checked={tavilyEnabled}
              onCheckedChange={setTavilyEnabled}
            />
          </div>

          {/* Tavily API Key input - only show if Tavily is enabled */}
          {tavilyEnabled && (
            <div className="space-y-3">
              <Label>Tavily API Key</Label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    type={showTavilyApiKey ? 'text' : 'password'}
                    placeholder="tvly-..."
                    value={tavilyApiKey}
                    onChange={(e) => setTavilyApiKey(e.target.value)}
                    className="pr-10"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                    onClick={() => setShowTavilyApiKey(!showTavilyApiKey)}
                  >
                    {showTavilyApiKey ? (
                      <EyeOff className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <Eye className="h-4 w-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>
              </div>
              {tavilyApiKey ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span>Tavily API key configured</span>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  Get a key at{' '}
                  <a
                    href="https://tavily.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    tavily.com
                  </a>
                </p>
              )}
            </div>
          )}

          {/* OpenRouter native search info */}
          {!tavilyEnabled && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Make sure you've enabled the Web Search plugin in your{' '}
                <a
                  href="https://openrouter.ai/settings/plugins"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  OpenRouter settings
                </a>
                . Native search uses provider pricing for Claude/OpenAI/Perplexity/xAI, or Exa ($0.02/request) for other models.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Convex Admin Key (Required for Progress Updates) */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Convex Admin Key</CardTitle>
            <Badge variant="default" className="text-[10px]">Required</Badge>
          </div>
          <CardDescription>
            Required for real-time progress updates during generation.
            This is the same key you use for Convex Dashboard access.
            Generate it with: <code className="text-xs bg-muted px-1 py-0.5 rounded">docker exec convex ./generate_admin_key.sh</code>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                type={showConvexAdminKey ? 'text' : 'password'}
                placeholder="convex-self-hosted|..."
                value={convexAdminKey}
                onChange={(e) => setConvexAdminKey(e.target.value)}
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                onClick={() => setShowConvexAdminKey(!showConvexAdminKey)}
              >
                {showConvexAdminKey ? (
                  <EyeOff className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <Eye className="h-4 w-4 text-muted-foreground" />
                )}
              </Button>
            </div>
          </div>
          {convexAdminKey ? (
            <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <span>Convex Admin key configured - progress updates will work</span>
            </div>
          ) : (
            <Alert className="mt-3">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Without this key, job progress won't update in the UI during generation.
                You'll still see results when the job completes.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Local LLM Server */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Server className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Local LLM Server</CardTitle>
            <Badge variant="outline" className="text-[10px]">Optional</Badge>
          </div>
          <CardDescription>
            Connect to a self-hosted vLLM server for generation using open-source models.
            Only affects the Generation phase — SIL, MAV, and other pipeline stages continue using OpenRouter.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Enable Toggle */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <ToggleLeft className="h-4 w-4 text-muted-foreground" />
                <Label htmlFor="local-llm-toggle" className="font-medium">
                  Enable Local LLMs
                </Label>
              </div>
              <p className="text-sm text-muted-foreground">
                {localLlmEnabled
                  ? "Local models will appear in the generation model selector"
                  : "All generation uses OpenRouter models"}
              </p>
            </div>
            <Switch
              id="local-llm-toggle"
              checked={localLlmEnabled}
              onCheckedChange={setLocalLlmEnabled}
            />
          </div>

          {/* Configuration (only when enabled) */}
          {localLlmEnabled && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Endpoint URL</Label>
                <Input
                  placeholder="http://your-vm-ip:8100"
                  value={localLlmEndpoint}
                  onChange={(e) => {
                    setLocalLlmEndpoint(e.target.value)
                    setConnectionStatus('idle')
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  The vLLM server URL. Copy this from the cera-vLLM dashboard.
                </p>
              </div>

              <div className="space-y-2">
                <Label>API Key <span className="text-muted-foreground">(from dashboard)</span></Label>
                <div className="relative">
                  <Input
                    type={showLocalLlmApiKey ? 'text' : 'password'}
                    placeholder="cvllm-..."
                    value={localLlmApiKey}
                    onChange={(e) => setLocalLlmApiKey(e.target.value)}
                    className="pr-10"
                  />
                  <Button
                    type="button" variant="ghost" size="icon"
                    className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                    onClick={() => setShowLocalLlmApiKey(!showLocalLlmApiKey)}
                  >
                    {showLocalLlmApiKey
                      ? <EyeOff className="h-4 w-4 text-muted-foreground" />
                      : <Eye className="h-4 w-4 text-muted-foreground" />}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Auto-generated by the cera-vLLM dashboard. Copy it from the Connection section.
                </p>
              </div>

              {/* Test Connection Button + Status */}
              <div className="flex items-center gap-3">
                <Button
                  variant="outline" size="sm"
                  onClick={handleTestConnection}
                  disabled={!localLlmEndpoint || testingConnection}
                >
                  {testingConnection
                    ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Testing...</>
                    : 'Test Connection'}
                </Button>
                {connectionStatus === 'success' && (
                  <div className="flex items-center gap-2 text-sm text-green-500">
                    <CheckCircle2 className="h-4 w-4" />
                    <span>{localModels.length} model(s): {localModels.join(', ')}</span>
                  </div>
                )}
                {connectionStatus === 'error' && (
                  <div className="flex items-center gap-2 text-sm text-red-500">
                    <AlertCircle className="h-4 w-4" />
                    <span>{connectionError}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Storage Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <FolderOutput className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Storage Settings</CardTitle>
          </div>
          <CardDescription>
            Configure where job files and datasets are saved
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Jobs Directory</Label>
            <Input
              value={jobsDir}
              onChange={(e) => setJobsDir(e.target.value)}
              placeholder="./jobs"
            />
            <p className="text-xs text-muted-foreground">
              Directory for job files (AML prompts, MAV data, reports, datasets)
            </p>
          </div>
          <div className="space-y-2">
            <Label>Theme</Label>
            <Select value={theme} onValueChange={(v) => setTheme(v as typeof theme)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="system">System</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* LLM Selection Presets */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-muted-foreground" />
              <CardTitle>LLM Selection Presets</CardTitle>
            </div>
            <Button size="sm" onClick={() => openPresetDialog()}>
              <Plus className="h-4 w-4 mr-1" />
              New Preset
            </Button>
          </div>
          <CardDescription>
            Save model configurations for quick selection when creating jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!presets || presets.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No presets created yet. Create one to save your preferred model selections.
            </p>
          ) : (
            <div className="space-y-2">
              {presets.map((preset: LLMPreset) => {
                const configuredModels = [
                  preset.rdeModel && { label: 'RDE', value: getModelDisplay(preset.rdeModel) },
                  preset.mavModels?.filter(m => m).length ? {
                    label: 'MAV',
                    value: `${preset.mavModels.filter(m => m).length} model${preset.mavModels.filter(m => m).length > 1 ? 's' : ''}`
                  } : null,
                  preset.savModel && { label: 'SAV', value: getModelDisplay(preset.savModel) },
                  preset.genModel && { label: 'GEN', value: getModelDisplay(preset.genModel) },
                ].filter(Boolean) as { label: string; value: React.ReactNode }[]

                return (
                  <div key={preset._id} className="flex items-center justify-between p-3 rounded-lg border bg-card">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{preset.name}</span>
                        {preset.isDefault && (
                          <Badge variant="secondary" className="text-[10px]">Default</Badge>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
                        {configuredModels.map((item, i) => (
                          <span key={i} className="text-xs text-muted-foreground">
                            <span className="font-medium">{item.label}:</span> {item.value}
                          </span>
                        ))}
                        {configuredModels.length === 0 && (
                          <span className="text-xs text-muted-foreground italic">No models configured</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async () => {
                          try {
                            if (preset.isDefault) {
                              await clearDefaultPreset({ id: preset._id })
                              toast.success('Default cleared')
                            } else {
                              await setDefaultPreset({ id: preset._id })
                              toast.success('Set as default')
                            }
                          } catch {
                            toast.error('Failed to update default')
                          }
                        }}
                        title={preset.isDefault ? 'Clear default' : 'Set as default'}
                      >
                        {preset.isDefault ? (
                          <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                        ) : (
                          <StarOff className="h-4 w-4 text-muted-foreground" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => openPresetDialog(preset)}
                        title="Edit preset"
                      >
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async () => {
                          try {
                            await deletePreset({ id: preset._id })
                            toast.success('Preset deleted')
                          } catch {
                            toast.error('Failed to delete preset')
                          }
                        }}
                        title="Delete preset"
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preset Edit/Create Dialog */}
      <Dialog open={presetDialogOpen} onOpenChange={(open) => {
        setPresetDialogOpen(open)
        if (!open) setExpandedSection(null)
      }}>
        <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {editingPreset ? 'Edit Preset' : 'Create New Preset'}
            </DialogTitle>
            <DialogDescription>
              Configure which models to include in this preset. All model fields are optional.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Preset Name */}
            <div className="space-y-2">
              <Label>Preset Name *</Label>
              <Input
                value={presetForm.name}
                onChange={(e) => setPresetForm(f => ({ ...f, name: e.target.value }))}
                placeholder="e.g., High-Quality Production"
              />
            </div>

            {/* Default Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <Label>Set as Default</Label>
                <p className="text-xs text-muted-foreground">
                  Auto-apply when creating new jobs
                </p>
              </div>
              <Switch
                checked={presetForm.isDefault}
                onCheckedChange={(v) => setPresetForm(f => ({ ...f, isDefault: v }))}
              />
            </div>

            <Separator />

            {/* Model Sections - Accordion Style */}
            <div className="space-y-2">
              {/* RDE Model */}
              <div className="rounded-lg border overflow-hidden">
                <button
                  type="button"
                  className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors text-left"
                  onClick={() => setExpandedSection(expandedSection === 'rde' ? null : 'rde')}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">RDE Model</div>
                    <p className="text-xs text-muted-foreground truncate">
                      {presetForm.rdeModel
                        ? processedModels.find(m => m.id === presetForm.rdeModel)?.name || presetForm.rdeModel.split('/').pop()
                        : 'Not configured'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {presetForm.rdeModel && (
                      <Badge variant="secondary" className="text-[10px]">1 model</Badge>
                    )}
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${expandedSection === 'rde' ? 'rotate-180' : ''}`} />
                  </div>
                </button>
                {expandedSection === 'rde' && (
                  <div className="p-3 border-t bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-2">
                      Used for extracting context from reference datasets
                    </p>
                    <div className="flex items-center gap-2">
                      <LLMSelector
                        providers={providers}
                        groupedModels={groupedModels}
                        loading={modelsLoading}
                        value={presetForm.rdeModel}
                        onChange={(v) => setPresetForm(f => ({ ...f, rdeModel: v }))}
                        placeholder="Select RDE model (optional)..."
                        className="flex-1"
                      />
                      {presetForm.rdeModel && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0"
                          onClick={() => setPresetForm(f => ({ ...f, rdeModel: '' }))}
                          title="Clear"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* MAV Models */}
              <div className="rounded-lg border overflow-hidden">
                <button
                  type="button"
                  className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors text-left"
                  onClick={() => setExpandedSection(expandedSection === 'mav' ? null : 'mav')}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">MAV Models</div>
                    <p className="text-xs text-muted-foreground truncate">
                      {presetForm.mavModels.filter(m => m).length > 0
                        ? presetForm.mavModels.filter(m => m).map(m => processedModels.find(pm => pm.id === m)?.name || m.split('/').pop()).join(', ')
                        : 'Not configured'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {presetForm.mavModels.filter(m => m).length > 0 && (
                      <Badge variant="secondary" className="text-[10px]">
                        {presetForm.mavModels.filter(m => m).length} model{presetForm.mavModels.filter(m => m).length > 1 ? 's' : ''}
                      </Badge>
                    )}
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${expandedSection === 'mav' ? 'rotate-180' : ''}`} />
                  </div>
                </button>
                {expandedSection === 'mav' && (
                  <div className="p-3 border-t bg-muted/30 space-y-3">
                    <p className="text-xs text-muted-foreground">
                      For CERA method: 2-3 models for consensus verification
                    </p>
                    {[0, 1, 2].map((index) => (
                      <div key={index} className="space-y-1">
                        <Label className="text-xs text-muted-foreground">
                          Model {index + 1}{index < 2 ? '' : ' (optional)'}
                        </Label>
                        <div className="flex items-center gap-2">
                          <LLMSelector
                            providers={providers}
                            groupedModels={groupedModels}
                            loading={modelsLoading}
                            value={presetForm.mavModels[index] || ''}
                            onChange={(v) => {
                              const newMav = [...presetForm.mavModels]
                              newMav[index] = v
                              setPresetForm(f => ({ ...f, mavModels: newMav }))
                            }}
                            disabledModels={presetForm.mavModels.filter((_, i) => i !== index)}
                            placeholder={`Select MAV model ${index + 1} (optional)...`}
                            className="flex-1"
                          />
                          {presetForm.mavModels[index] && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="shrink-0"
                              onClick={() => {
                                const newMav = [...presetForm.mavModels]
                                newMav[index] = ''
                                setPresetForm(f => ({ ...f, mavModels: newMav }))
                              }}
                              title="Clear"
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* SAV Model */}
              <div className="rounded-lg border overflow-hidden">
                <button
                  type="button"
                  className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors text-left"
                  onClick={() => setExpandedSection(expandedSection === 'sav' ? null : 'sav')}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">SAV Model</div>
                    <p className="text-xs text-muted-foreground truncate">
                      {presetForm.savModel
                        ? processedModels.find(m => m.id === presetForm.savModel)?.name || presetForm.savModel.split('/').pop()
                        : 'Not configured'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {presetForm.savModel && (
                      <Badge variant="secondary" className="text-[10px]">1 model</Badge>
                    )}
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${expandedSection === 'sav' ? 'rotate-180' : ''}`} />
                  </div>
                </button>
                {expandedSection === 'sav' && (
                  <div className="p-3 border-t bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-2">
                      Used when MAV is disabled (single model for verification)
                    </p>
                    <div className="flex items-center gap-2">
                      <LLMSelector
                        providers={providers}
                        groupedModels={groupedModels}
                        loading={modelsLoading}
                        value={presetForm.savModel}
                        onChange={(v) => setPresetForm(f => ({ ...f, savModel: v }))}
                        placeholder="Select SAV model (optional)..."
                        className="flex-1"
                      />
                      {presetForm.savModel && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0"
                          onClick={() => setPresetForm(f => ({ ...f, savModel: '' }))}
                          title="Clear"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Generation Model */}
              <div className="rounded-lg border overflow-hidden">
                <button
                  type="button"
                  className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors text-left"
                  onClick={() => setExpandedSection(expandedSection === 'gen' ? null : 'gen')}
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm">Generation Model</div>
                    <p className="text-xs text-muted-foreground truncate">
                      {presetForm.genModel
                        ? processedModels.find(m => m.id === presetForm.genModel)?.name || presetForm.genModel.split('/').pop()
                        : 'Not configured'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {presetForm.genModel && (
                      <Badge variant="secondary" className="text-[10px]">1 model</Badge>
                    )}
                    <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform ${expandedSection === 'gen' ? 'rotate-180' : ''}`} />
                  </div>
                </button>
                {expandedSection === 'gen' && (
                  <div className="p-3 border-t bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-2">
                      Primary model for review generation (AML phase)
                    </p>
                    <div className="flex items-center gap-2">
                      <LLMSelector
                        providers={providers}
                        groupedModels={groupedModels}
                        loading={modelsLoading}
                        value={presetForm.genModel}
                        onChange={(v) => setPresetForm(f => ({ ...f, genModel: v }))}
                        placeholder="Select generation model (optional)..."
                        className="flex-1"
                      />
                      {presetForm.genModel && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0"
                          onClick={() => setPresetForm(f => ({ ...f, genModel: '' }))}
                          title="Clear"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setPresetDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleSavePreset}
              disabled={!presetForm.name.trim() || savingPreset}
            >
              {savingPreset ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                editingPreset ? 'Save Changes' : 'Create Preset'
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Clear Form Data */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Trash2 className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Clear Cached Data</CardTitle>
          </div>
          <CardDescription>
            Clear cached form data and extracted reference context. Use this if you're experiencing issues with stale data.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button
            variant="outline"
            onClick={() => {
              localStorage.removeItem('cera-generate-config')
              localStorage.removeItem('cera-ref-context-cache')
              toast.success('Form data and reference context cache cleared.')
            }}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Clear All Cached Data
          </Button>
          <p className="text-xs text-muted-foreground">
            This clears the Create Job form data and any cached reference dataset extractions.
          </p>
        </CardContent>
      </Card>

      <Separator />

      {/* Save Button */}
      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={saving}>
          <Save className="mr-2 h-4 w-4" />
          {saving ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </div>
  )
}
