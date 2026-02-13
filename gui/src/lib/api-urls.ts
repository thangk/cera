/**
 * Dynamic API URL resolution for local vs remote access
 *
 * When accessing from localhost, use localhost URLs.
 * When accessing from Tailscale/remote, use the configured URLs.
 */

const isLocalhost = typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

function resolveUrl(envUrl: string): string {
  if (!isLocalhost) return envUrl

  try {
    const url = new URL(envUrl)
    url.hostname = 'localhost'
    return url.toString().replace(/\/$/, '')
  } catch {
    return envUrl
  }
}

export const CONVEX_URL = resolveUrl(import.meta.env.VITE_CONVEX_URL as string)
export const PYTHON_API_URL = resolveUrl(import.meta.env.VITE_PYTHON_API_URL as string)
export const POCKETBASE_URL = resolveUrl((import.meta.env.VITE_POCKETBASE_URL as string) || 'http://localhost:8090')
