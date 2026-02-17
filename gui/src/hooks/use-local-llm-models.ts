import { useState, useEffect, useMemo } from 'react'
import { useQuery } from 'convex/react'
import { api } from 'convex/_generated/api'
import type { ProcessedModel, GroupedModels, ProviderInfo } from './use-openrouter-models'

export function useLocalLlmModels() {
  const settings = useQuery(api.settings.get)
  const [models, setModels] = useState<ProcessedModel[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const enabled = settings?.localLlmEnabled ?? false
  const endpoint = settings?.localLlmEndpoint ?? ''
  const apiKey = settings?.localLlmApiKey ?? ''

  useEffect(() => {
    if (!enabled || !endpoint) {
      setModels([])
      return
    }

    let cancelled = false

    async function fetchModels() {
      setLoading(true)
      setError(null)
      try {
        const url = endpoint.replace(/\/+$/, '') + '/v1/models'
        const headers: Record<string, string> = {}
        if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`
        const res = await fetch(url, { headers, signal: AbortSignal.timeout(10000) })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()

        if (cancelled) return

        const processed: ProcessedModel[] = (data.data || []).map((m: { id: string }) => ({
          id: `local/${m.id}`,
          name: m.id.split('/').pop() || m.id,
          contextLength: 0,
          pricing: 'Local',
          isFree: true,
          provider: 'local',
          providerLabel: 'Local LLMs',
          isOpenSource: true,
          inputModalities: ['text'],
          outputModalities: ['text'],
          hasVision: false,
          hasAudio: false,
          hasVideo: false,
          hasThinking: false,
        }))
        setModels(processed)
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to fetch local models')
        setModels([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchModels()
    const interval = setInterval(fetchModels, 60000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [enabled, endpoint, apiKey])

  const groupedModels = useMemo<GroupedModels>(() => {
    if (models.length === 0) return {}
    return {
      local: { label: 'Local LLMs', models },
    }
  }, [models])

  const providers = useMemo<ProviderInfo[]>(() => {
    if (models.length === 0) return []
    return [{
      id: 'local',
      label: 'Local LLMs',
      modelCount: models.length,
      isOpenSource: true,
      hasFreeModels: true,
      hasPaidModels: false,
    }]
  }, [models])

  return {
    enabled,
    configured: !!endpoint,
    models,
    groupedModels,
    providers,
    loading,
    error,
  }
}
