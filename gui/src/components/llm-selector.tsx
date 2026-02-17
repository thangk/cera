import { useState, useMemo } from 'react'
import { Check, ChevronsUpDown, Search, Type, Image, Video, Mic, Loader2, X, CheckCircle2, BrainCircuit } from 'lucide-react'
import { cn } from '../lib/utils'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from './ui/popover'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from './ui/tooltip'
import { useOpenRouterModels, type ProcessedModel, type ProviderInfo } from '../hooks/use-openrouter-models'
import { useLocalLlmModels } from '../hooks/use-local-llm-models'

export type ValidationStatus = 'idle' | 'checking' | 'valid' | 'error'

// Clean model name by removing "(free)" suffix
function cleanModelName(name: string): string {
  return name.replace(/\s*\(free\)\s*$/i, '').trim()
}

interface LLMSelectorProps {
  value: string  // Full model ID like "anthropic/claude-sonnet-4"
  onChange: (modelId: string) => void
  disabledModels?: string[]
  placeholder?: string
  className?: string
  // Optional: pass in hook data to avoid re-fetching
  providers?: ProviderInfo[]
  groupedModels?: Record<string, { label: string; models: ProcessedModel[] }>
  loading?: boolean
  // Validation status
  validationStatus?: ValidationStatus
  validationError?: string
  // Thinking mode filter
  thinkingMode?: boolean
  onThinkingModeChange?: (enabled: boolean) => void
}

export function LLMSelector({
  value,
  onChange,
  disabledModels = [],
  placeholder = 'Select model...',
  className,
  providers: externalProviders,
  groupedModels: externalGroupedModels,
  loading: externalLoading,
  validationStatus = 'idle',
  validationError,
  thinkingMode,
  onThinkingModeChange,
}: LLMSelectorProps) {
  // Use external data if provided, otherwise fetch
  const hookData = useOpenRouterModels()
  const localData = useLocalLlmModels()
  const baseProviders = externalProviders ?? hookData.providers
  const baseGroupedModels = externalGroupedModels ?? hookData.groupedModels
  const loading = externalLoading ?? hookData.loading

  // Merge local models at the top when enabled and configured
  const providers = useMemo(() => {
    if (!localData.enabled || !localData.configured || localData.providers.length === 0) {
      return baseProviders
    }
    return [...localData.providers, ...baseProviders]
  }, [localData.enabled, localData.configured, localData.providers, baseProviders])

  const groupedModels = useMemo(() => {
    if (!localData.enabled || !localData.configured || Object.keys(localData.groupedModels).length === 0) {
      return baseGroupedModels
    }
    return { ...localData.groupedModels, ...baseGroupedModels }
  }, [localData.enabled, localData.configured, localData.groupedModels, baseGroupedModels])

  // Internal thinking mode state (used when no external control provided)
  const [internalThinkingMode, setInternalThinkingMode] = useState(false)
  const isThinkingMode = thinkingMode ?? internalThinkingMode
  const handleThinkingToggle = (enabled: boolean) => {
    if (onThinkingModeChange) {
      onThinkingModeChange(enabled)
    } else {
      setInternalThinkingMode(enabled)
    }
  }

  // Extract provider from value, or use pending provider when user just picked a provider
  const [pendingProvider, setPendingProvider] = useState<string | null>(null)
  const selectedProvider = pendingProvider || (value ? value.split('/')[0] : '')
  const selectedModel = value
    ? (groupedModels[value.split('/')[0]]?.models.find(m => m.id === value) ?? undefined)
    : undefined

  const [providerOpen, setProviderOpen] = useState(false)
  const [modelOpen, setModelOpen] = useState(false)
  const [providerSearch, setProviderSearch] = useState('')
  const [modelSearch, setModelSearch] = useState('')

  // Filter providers
  const filteredProviders = useMemo(() => {
    if (!providerSearch) return providers
    const searchLower = providerSearch.toLowerCase()
    return providers.filter(p =>
      p.label.toLowerCase().includes(searchLower) ||
      p.id.toLowerCase().includes(searchLower)
    )
  }, [providers, providerSearch])

  // Filter models for selected provider (with thinking mode filter)
  const availableModels = useMemo(() => {
    const models = groupedModels[selectedProvider]?.models ?? []
    if (!isThinkingMode) return models
    return models.filter(m => m.hasThinking)
  }, [groupedModels, selectedProvider, isThinkingMode])

  const filteredModels = useMemo(() => {
    if (!modelSearch) return availableModels
    const searchLower = modelSearch.toLowerCase()
    return availableModels.filter(m =>
      m.name.toLowerCase().includes(searchLower) ||
      m.id.toLowerCase().includes(searchLower)
    )
  }, [availableModels, modelSearch])

  const handleProviderChange = (providerId: string) => {
    // Set pending provider and open model dropdown for user to pick
    setPendingProvider(providerId)
    setProviderOpen(false)
    setProviderSearch('')
    // Open model dropdown after a tick so the popover transition completes
    setTimeout(() => setModelOpen(true), 100)
  }

  const handleModelChange = (modelId: string) => {
    setPendingProvider(null) // Clear pending since we have a full model ID now
    onChange(modelId)
    setModelOpen(false)
    setModelSearch('')
  }

  if (loading) {
    return (
      <div className={cn("flex gap-2", className)}>
        <Button variant="outline" className="flex-1" disabled>
          Loading...
        </Button>
        <Button variant="outline" className="flex-[2]" disabled>
          Loading...
        </Button>
      </div>
    )
  }

  const selectedProviderInfo = providers.find(p => p.id === selectedProvider)

  return (
    <div className={cn("flex gap-2", className)}>
      {/* Provider Dropdown */}
      <Popover open={providerOpen} onOpenChange={setProviderOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={providerOpen}
            className="w-[180px] justify-between h-auto min-h-10"
          >
            {selectedProviderInfo ? (
              <span className="truncate">{selectedProviderInfo.label}</span>
            ) : (
              <span className="text-muted-foreground">Provider</span>
            )}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[280px] p-0" align="start" onWheel={(e) => e.stopPropagation()}>
          <div className="flex items-center px-3 py-2">
            <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
            <input
              placeholder="Search providers..."
              value={providerSearch}
              onChange={(e) => setProviderSearch(e.target.value)}
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>
          <ScrollArea className="h-[300px]">
            {filteredProviders.length === 0 ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                No providers found.
              </div>
            ) : (
              <div className="p-1">
                {filteredProviders.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => handleProviderChange(provider.id)}
                    className={cn(
                      'relative flex w-full items-center gap-2 rounded-sm px-2 pr-3 py-2 text-left text-sm outline-none transition-colors',
                      'hover:bg-accent hover:text-accent-foreground cursor-pointer',
                      selectedProvider === provider.id && 'bg-accent'
                    )}
                  >
                    <Check
                      className={cn(
                        'h-4 w-4 shrink-0',
                        selectedProvider === provider.id ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    <span className="font-medium truncate flex-1 min-w-0">{provider.label}</span>
                    <div className="flex items-center gap-1 shrink-0">
                      {/* Order: Local, OSS, Free, Paid - show all applicable badges */}
                      {provider.id === 'local' && (
                        <Badge variant="secondary" className="bg-blue-500/20 text-blue-500 text-[9px] px-1 py-0">Local</Badge>
                      )}
                      {provider.isOpenSource && provider.id !== 'local' && (
                        <Badge variant="outline" className="text-[9px] px-1 py-0">OSS</Badge>
                      )}
                      {provider.hasFreeModels && (
                        <Badge variant="secondary" className="bg-green-500/20 text-green-600 text-[9px] px-1 py-0">Free</Badge>
                      )}
                      {provider.hasPaidModels && (
                        <Badge variant="secondary" className="bg-amber-500/20 text-amber-600 text-[9px] px-1 py-0">Paid</Badge>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground shrink-0 w-4 text-right">
                      ({provider.modelCount})
                    </span>
                  </button>
                ))}
              </div>
            )}
          </ScrollArea>
        </PopoverContent>
      </Popover>

      {/* Thinking Mode Toggle */}
      <Button
        variant="outline"
        onClick={() => handleThinkingToggle(!isThinkingMode)}
        className={cn(
          "h-auto min-h-10 w-10 shrink-0 px-0",
          isThinkingMode && "bg-purple-500/20 border-purple-500/50 text-purple-500 hover:bg-purple-500/30 hover:text-purple-500"
        )}
      >
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="flex items-center justify-center h-full w-full">
              <BrainCircuit className="h-4 w-4" />
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p className="text-xs">{isThinkingMode ? 'Thinking mode: ON' : 'Thinking mode: OFF'}</p>
          </TooltipContent>
        </Tooltip>
      </Button>

      {/* Model Dropdown */}
      <Popover open={modelOpen} onOpenChange={setModelOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={modelOpen}
            disabled={!selectedProvider}
            className={cn(
              "flex-1 justify-between h-auto min-h-10 overflow-hidden",
              validationStatus === 'error' && "border-red-500 border-2"
            )}
          >
            {selectedModel ? (
              <div className="flex flex-col items-start gap-0.5 text-left min-w-0 flex-1 overflow-hidden">
                <span className="text-sm font-medium truncate w-full">{cleanModelName(selectedModel.name)}</span>
                <div className="flex items-center gap-1">
                  {selectedModel.isFree ? (
                    <Badge variant="secondary" className="bg-green-500/20 text-green-600 text-[9px] px-1 py-0">Free</Badge>
                  ) : (
                    <span className="text-[10px] text-muted-foreground">{selectedModel.pricing}</span>
                  )}
                  {selectedModel.isOpenSource && (
                    <Badge variant="outline" className="text-[9px] px-1 py-0">OSS</Badge>
                  )}
                </div>
              </div>
            ) : (
              <span className="text-muted-foreground">{placeholder}</span>
            )}
            {/* Modality Icons - positioned with validation indicator for vertical centering */}
            {selectedModel && (
              <div className="flex items-center shrink-0">
                <ModalityIcons model={selectedModel} />
              </div>
            )}
            {/* Validation Status Indicator */}
            <div className="ml-2 flex items-center gap-1.5 shrink-0">
              {validationStatus === 'checking' && (
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span className="text-[10px]">checking</span>
                </div>
              )}
              {validationStatus === 'valid' && (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              )}
              {validationStatus === 'error' && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-1 text-red-500">
                      <X className="h-4 w-4" />
                      <span className="text-[10px]">unavailable</span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-xs">
                    <p className="text-xs">{validationError || 'Model not available'}</p>
                  </TooltipContent>
                </Tooltip>
              )}
              {validationStatus === 'idle' && (
                <ChevronsUpDown className="h-4 w-4 opacity-50" />
              )}
            </div>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[400px] p-0" align="start" onWheel={(e) => e.stopPropagation()}>
          <div className="flex items-center px-3 py-2">
            <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
            <input
              placeholder="Search models..."
              value={modelSearch}
              onChange={(e) => setModelSearch(e.target.value)}
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>
          <ScrollArea className="h-[300px]">
            {filteredModels.length === 0 ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                {selectedProvider ? 'No models found.' : 'Select a provider first.'}
              </div>
            ) : (
              <div className="p-1">
                {/* Free models first */}
                {filteredModels.some(m => m.isFree) && (
                  <div className="mb-2">
                    <div className="px-2 py-1.5 text-xs font-semibold text-green-600 uppercase tracking-wider">
                      Free
                    </div>
                    {filteredModels.filter(m => m.isFree).map((model) => (
                      <ModelItem
                        key={model.id}
                        model={model}
                        isSelected={value === model.id}
                        isDisabled={disabledModels.includes(model.id) && model.id !== value}
                        onClick={() => handleModelChange(model.id)}
                      />
                    ))}
                  </div>
                )}
                {/* Paid models */}
                {filteredModels.some(m => !m.isFree) && (
                  <div className="mb-2">
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Paid
                    </div>
                    {filteredModels.filter(m => !m.isFree).map((model) => (
                      <ModelItem
                        key={model.id}
                        model={model}
                        isSelected={value === model.id}
                        isDisabled={disabledModels.includes(model.id) && model.id !== value}
                        onClick={() => handleModelChange(model.id)}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </ScrollArea>
        </PopoverContent>
      </Popover>
    </div>
  )
}

// Modality icons component
// Order: text (leftmost), image, video, audio
function ModalityIcons({ model, className }: { model: ProcessedModel; className?: string }) {
  const hasText = model.inputModalities.includes('text')
  const hasModalities = hasText || model.hasVision || model.hasAudio || model.hasVideo || model.hasThinking

  if (!hasModalities) return null

  return (
    <div className={cn("flex items-center gap-0.5", className)}>
      {model.hasThinking && (
        <span title="Thinking / Reasoning">
          <BrainCircuit className="h-3 w-3 text-purple-500" />
        </span>
      )}
      {hasText && (
        <span title="Text input">
          <Type className="h-3 w-3 text-muted-foreground" />
        </span>
      )}
      {model.hasVision && (
        <span title="Vision (image input)">
          <Image className="h-3 w-3 text-muted-foreground" />
        </span>
      )}
      {model.hasVideo && (
        <span title="Video input">
          <Video className="h-3 w-3 text-muted-foreground" />
        </span>
      )}
      {model.hasAudio && (
        <span title="Audio support">
          <Mic className="h-3 w-3 text-muted-foreground" />
        </span>
      )}
    </div>
  )
}

// Individual model item in dropdown
function ModelItem({
  model,
  isSelected,
  isDisabled,
  onClick,
}: {
  model: ProcessedModel
  isSelected: boolean
  isDisabled: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      className={cn(
        'relative flex w-full items-start gap-2 rounded-sm px-2 pr-3 py-2 text-left text-sm outline-none transition-colors',
        isDisabled
          ? 'opacity-50 cursor-not-allowed'
          : 'hover:bg-accent hover:text-accent-foreground cursor-pointer',
        isSelected && 'bg-accent'
      )}
    >
      <Check
        className={cn(
          'h-4 w-4 mt-0.5 shrink-0',
          isSelected ? 'opacity-100' : 'opacity-0'
        )}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="font-medium truncate flex-1">{cleanModelName(model.name)}</span>
          <div className="flex items-center gap-1.5 shrink-0">
            <ModalityIcons model={model} />
          </div>
        </div>
        <div className="flex items-center gap-1 mt-0.5">
          {model.isFree ? (
            <Badge variant="secondary" className="bg-green-500/20 text-green-600 text-[9px] px-1 py-0">
              Free
            </Badge>
          ) : (
            <span className="text-[10px] text-muted-foreground">{model.pricing}</span>
          )}
          {model.isOpenSource && (
            <Badge variant="outline" className="text-[9px] px-1 py-0">OSS</Badge>
          )}
          <span className="text-[10px] text-muted-foreground">· {model.contextLength.toLocaleString()} ctx</span>
        </div>
      </div>
    </button>
  )
}
