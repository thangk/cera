import { useState, useEffect, useMemo } from 'react'

export interface OpenRouterModel {
  id: string
  name: string
  description?: string
  pricing: {
    prompt: string
    completion: string
  }
  context_length: number
  architecture?: {
    modality?: string
    input_modalities?: string[]
    output_modalities?: string[]
  }
  top_provider?: {
    is_moderated: boolean
  }
  supported_parameters?: string[]
}

interface OpenRouterModelsResponse {
  data: OpenRouterModel[]
}

export interface ProcessedModel {
  id: string
  name: string
  contextLength: number
  pricing: string
  isFree: boolean
  provider: string
  providerLabel: string
  isOpenSource: boolean
  // Modality info
  inputModalities: string[]
  outputModalities: string[]
  hasVision: boolean
  hasAudio: boolean
  hasVideo: boolean
  hasThinking: boolean
}

export interface GroupedModels {
  [provider: string]: {
    label: string
    models: ProcessedModel[]
  }
}

export interface ProviderInfo {
  id: string
  label: string
  modelCount: number
  isOpenSource: boolean
  hasFreeModels: boolean
  hasPaidModels: boolean
}

// Map provider IDs to display names and metadata
const PROVIDER_META: Record<string, { label: string; isOpenSource: boolean }> = {
  anthropic: { label: 'Anthropic', isOpenSource: false },
  openai: { label: 'OpenAI', isOpenSource: false },
  google: { label: 'Google', isOpenSource: false },
  'x-ai': { label: 'xAI', isOpenSource: false },
  xai: { label: 'xAI', isOpenSource: false },
  deepseek: { label: 'DeepSeek', isOpenSource: true },
  meta: { label: 'Meta', isOpenSource: true },
  'meta-llama': { label: 'Meta Llama', isOpenSource: true },
  mistralai: { label: 'Mistral AI', isOpenSource: true },
  cohere: { label: 'Cohere', isOpenSource: false },
  perplexity: { label: 'Perplexity', isOpenSource: false },
  microsoft: { label: 'Microsoft', isOpenSource: false },
  amazon: { label: 'Amazon', isOpenSource: false },
  'ai21': { label: 'AI21 Labs', isOpenSource: false },
  nvidia: { label: 'NVIDIA', isOpenSource: true },
  qwen: { label: 'Qwen', isOpenSource: true },
  databricks: { label: 'Databricks', isOpenSource: true },
}

function processModel(model: OpenRouterModel): ProcessedModel {
  const [providerId] = model.id.split('/')
  const meta = PROVIDER_META[providerId]
  const promptPrice = parseFloat(model.pricing.prompt) || 0
  const completionPrice = parseFloat(model.pricing.completion) || 0
  const isFree = promptPrice === 0 && completionPrice === 0

  // Get modalities from API
  let inputModalities = model.architecture?.input_modalities || []
  let outputModalities = model.architecture?.output_modalities || []

  // Infer from model name/id for special cases
  const modelLower = model.id.toLowerCase() + ' ' + model.name.toLowerCase()

  // Detect special model types that need modality override
  const isAudioOnlyModel = (modelLower.includes('audio') || modelLower.includes('tts') || modelLower.includes('speech'))
                           && !modelLower.includes('omni') && !modelLower.includes('4o')
  const isImageGenModel = modelLower.includes('dall-e') || modelLower.includes('image-gen')

  // Override modalities for known special models (OpenRouter may report incorrectly)
  if (isAudioOnlyModel) {
    inputModalities = ['audio']
    outputModalities = ['audio']
  } else if (isImageGenModel) {
    inputModalities = ['text'] // Image gen takes text prompts
    outputModalities = ['image']
  } else if (inputModalities.length === 0) {
    // Default to text for regular chat models
    inputModalities = ['text']
  }

  if (outputModalities.length === 0) {
    outputModalities = ['text']
  }

  return {
    id: model.id,
    name: model.name,
    contextLength: model.context_length,
    pricing: isFree ? 'Free' : `$${(promptPrice * 1000000).toFixed(2)}/$${(completionPrice * 1000000).toFixed(2)} per 1M tokens`,
    isFree,
    provider: providerId,
    providerLabel: meta?.label || providerId.charAt(0).toUpperCase() + providerId.slice(1),
    isOpenSource: meta?.isOpenSource ?? false,
    inputModalities,
    outputModalities,
    hasVision: inputModalities.includes('image'),
    hasAudio: inputModalities.includes('audio') || outputModalities.includes('audio'),
    hasVideo: inputModalities.includes('video'),
    hasThinking: model.supported_parameters?.includes('reasoning') || model.id.includes(':thinking'),
  }
}

export function useOpenRouterModels() {
  const [models, setModels] = useState<OpenRouterModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchModels() {
      try {
        const response = await fetch('https://openrouter.ai/api/v1/models')
        if (!response.ok) {
          throw new Error('Failed to fetch models')
        }
        const data: OpenRouterModelsResponse = await response.json()
        setModels(data.data || [])
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchModels()
  }, [])

  // Process all models with enriched data
  const processedModels = useMemo<ProcessedModel[]>(() => {
    return models.map(processModel).sort((a, b) => {
      // Sort: free first, then by name
      if (a.isFree && !b.isFree) return -1
      if (!a.isFree && b.isFree) return 1
      return a.name.localeCompare(b.name)
    })
  }, [models])

  // Group models by provider
  const groupedModels = useMemo<GroupedModels>(() => {
    const grouped: GroupedModels = {}

    for (const model of processedModels) {
      if (!grouped[model.provider]) {
        grouped[model.provider] = {
          label: model.providerLabel,
          models: [],
        }
      }
      grouped[model.provider].models.push(model)
    }

    return grouped
  }, [processedModels])

  // Get sorted provider list (prioritize popular providers)
  const providers = useMemo<ProviderInfo[]>(() => {
    const priority = ['anthropic', 'openai', 'google', 'meta-llama', 'mistralai', 'deepseek', 'x-ai', 'cohere']
    const allProviders = Object.keys(groupedModels)

    return [
      ...priority.filter(p => allProviders.includes(p)),
      ...allProviders.filter(p => !priority.includes(p)).sort(),
    ].map(id => {
      const providerModels = groupedModels[id]?.models || []
      const meta = PROVIDER_META[id]
      return {
        id,
        label: groupedModels[id]?.label || id,
        modelCount: providerModels.length,
        isOpenSource: meta?.isOpenSource ?? false,
        hasFreeModels: providerModels.some(m => m.isFree),
        hasPaidModels: providerModels.some(m => !m.isFree),
      }
    })
  }, [groupedModels])

  return {
    models,
    groupedModels,
    providers,
    processedModels,
    loading,
    error,
  }
}
