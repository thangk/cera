import { useState, useEffect } from 'react'

export interface OpenRouterKeyInfo {
  label: string | null
  usage: number
  limit: number | null
  is_free_tier: boolean
  rate_limit: {
    requests: number
    interval: string
  }
}

interface OpenRouterKeyResponse {
  data: OpenRouterKeyInfo
}

export function useOpenRouterLimits(apiKey: string | undefined) {
  const [keyInfo, setKeyInfo] = useState<OpenRouterKeyInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!apiKey) {
      setKeyInfo(null)
      setError(null)
      return
    }

    async function fetchKeyInfo() {
      setLoading(true)
      setError(null)

      try {
        const response = await fetch('https://openrouter.ai/api/v1/auth/key', {
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
        })

        if (!response.ok) {
          if (response.status === 401) {
            throw new Error('Invalid API key')
          }
          throw new Error('Failed to fetch key info')
        }

        const data: OpenRouterKeyResponse = await response.json()
        setKeyInfo(data.data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
        setKeyInfo(null)
      } finally {
        setLoading(false)
      }
    }

    fetchKeyInfo()
  }, [apiKey])

  const refetch = () => {
    if (apiKey) {
      setLoading(true)
      fetch('https://openrouter.ai/api/v1/auth/key', {
        headers: { Authorization: `Bearer ${apiKey}` },
      })
        .then((res) => res.json())
        .then((data: OpenRouterKeyResponse) => setKeyInfo(data.data))
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false))
    }
  }

  return { keyInfo, loading, error, refetch }
}
