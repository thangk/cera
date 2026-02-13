import PocketBase, { RecordModel } from 'pocketbase'
import { useState, useEffect, useRef, useMemo } from 'react'
import { POCKETBASE_URL } from '../lib/api-urls'

// Singleton PocketBase client
const pb = new PocketBase(POCKETBASE_URL)

// Disable auto-cancellation (we manage subscriptions manually)
pb.autoCancellation(false)

// --- Progress Hook ---

export interface JobProgress {
  progress: number
  current_phase: string
  generated_count: number
  generated_sentences: number
  failed_count: number
  current_run: number
  total_runs: number
  model_progress: any[] | null
  target_progress: any[] | null
  run_progress: any[] | null
  heuristic_progress: any | null
}

function recordToProgress(record: RecordModel): JobProgress {
  return {
    progress: record.progress ?? 0,
    current_phase: record.current_phase ?? '',
    generated_count: record.generated_count ?? 0,
    generated_sentences: record.generated_sentences ?? 0,
    failed_count: record.failed_count ?? 0,
    current_run: record.current_run ?? 0,
    total_runs: record.total_runs ?? 0,
    model_progress: record.model_progress ?? null,
    target_progress: record.target_progress ?? null,
    run_progress: record.run_progress ?? null,
    heuristic_progress: record.heuristic_progress ?? null,
  }
}

/**
 * Subscribe to real-time progress updates from PocketBase.
 * Returns null when no PB record exists (e.g. completed/historical jobs).
 */
export function usePocketBaseProgress(jobId: string | undefined): JobProgress | null {
  const [progress, setProgress] = useState<JobProgress | null>(null)
  const recordIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (!jobId) return

    let cancelled = false

    // Initial fetch
    pb.collection('job_progress')
      .getFirstListItem(`job_id="${jobId}"`)
      .then(record => {
        if (cancelled) return
        recordIdRef.current = record.id
        setProgress(recordToProgress(record))
      })
      .catch(() => {
        // No record yet — job hasn't started or PocketBase unavailable
      })

    // Subscribe to real-time updates via SSE
    const unsubPromise = pb.collection('job_progress').subscribe('*', (e) => {
      if (cancelled) return
      if (e.record.job_id === jobId) {
        if (e.action === 'create' || e.action === 'update') {
          recordIdRef.current = e.record.id
          setProgress(recordToProgress(e.record))
        } else if (e.action === 'delete') {
          recordIdRef.current = null
          setProgress(null)
        }
      }
    })

    return () => {
      cancelled = true
      unsubPromise.then(unsub => unsub()).catch(() => {})
    }
  }, [jobId])

  return progress
}

// --- Logs Hook ---

export interface PBLogEntry {
  _id: string
  timestamp: number
  level: 'INFO' | 'WARN' | 'ERROR'
  phase: string
  message: string
}

function recordToLog(record: RecordModel): PBLogEntry {
  return {
    _id: record.id,
    timestamp: new Date(record.created).getTime(),
    level: record.level as 'INFO' | 'WARN' | 'ERROR',
    phase: record.phase,
    message: record.message,
  }
}

/**
 * Subscribe to real-time log entries from PocketBase.
 * Returns logs in the same shape as Convex logs (compatible with LogStream).
 */
export function usePocketBaseLogs(
  jobId: string | undefined,
  phases?: string[],
): PBLogEntry[] | undefined {
  const [logs, setLogs] = useState<PBLogEntry[]>([])
  const [loaded, setLoaded] = useState(false)

  // Memoize phases to avoid re-subscribing on every render
  const phasesKey = phases?.join(',') ?? ''

  useEffect(() => {
    if (!jobId) return

    let cancelled = false
    setLogs([])
    setLoaded(false)

    // Build PocketBase filter
    const parts: string[] = [`job_id="${jobId}"`]
    if (phases && phases.length > 0) {
      const phaseFilters = phases.map(p => `phase="${p}"`).join(' || ')
      parts.push(`(${phaseFilters})`)
    }
    const filter = parts.join(' && ')

    // Initial fetch (last 500, ascending)
    pb.collection('job_logs')
      .getList(1, 500, { filter, sort: '+created' })
      .then(result => {
        if (cancelled) return
        setLogs(result.items.map(recordToLog))
        setLoaded(true)
      })
      .catch(() => {
        if (!cancelled) setLoaded(true)
      })

    // Subscribe to new log entries via SSE
    const phaseSet = phases && phases.length > 0 ? new Set(phases) : null

    const unsubPromise = pb.collection('job_logs').subscribe('*', (e) => {
      if (cancelled) return
      if (e.action === 'create' && e.record.job_id === jobId) {
        if (!phaseSet || phaseSet.has(e.record.phase)) {
          const entry = recordToLog(e.record)
          setLogs(prev => [...prev, entry].slice(-500))
        }
      }
    })

    return () => {
      cancelled = true
      unsubPromise.then(unsub => unsub()).catch(() => {})
    }
  }, [jobId, phasesKey])

  return loaded ? logs : undefined
}
