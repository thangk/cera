import { useEffect, useRef } from 'react'
import { useMutation } from 'convex/react'
import { api } from 'convex/_generated/api'

/**
 * Seeds Convex settings from environment variables on first load.
 * Only fills in keys that are missing in Convex - does not overwrite existing values.
 */
export function useSeedSettings() {
  const seedFromEnv = useMutation(api.settings.seedFromEnv)
  const hasSeeded = useRef(false)

  useEffect(() => {
    // Only run once per session
    if (hasSeeded.current) return
    hasSeeded.current = true

    // Get env vars (baked in at build time via Vite)
    const openrouterApiKey = import.meta.env.VITE_OPENROUTER_API_KEY
    const tavilyApiKey = import.meta.env.VITE_TAVILY_API_KEY

    // Only seed if we have at least one key from env
    if (openrouterApiKey || tavilyApiKey) {
      seedFromEnv({
        openrouterApiKey: openrouterApiKey || undefined,
        tavilyApiKey: tavilyApiKey || undefined,
      }).catch((err) => {
        console.warn('Failed to seed settings from env:', err)
      })
    }
  }, [seedFromEnv])
}
