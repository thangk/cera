import { useCallback, useRef } from 'react'
import { ChevronDown, ChevronUp, Copy, Check } from 'lucide-react'
import { useQuery } from 'convex/react'
import { api } from '../../../convex/_generated/api'
import type { Id } from '../../../convex/_generated/dataModel'
import { LogStream } from './log-stream'
import { useLogPanel, PHASE_LOG_FILTERS, type LogTab } from '@/hooks/use-log-panel'
import { useState } from 'react'

// Phase color definitions (matching job detail page)
const PHASE_COLORS: Record<LogTab, { strong: string; light: string }> = {
  all: { strong: '#6b7280', light: '#e5e7eb' },
  composition: { strong: '#4e95d9', light: '#dceaf7' },
  generation: { strong: '#f2aa84', light: '#fbe3d6' },
  evaluation: { strong: '#8ed973', light: '#d9f2d0' },
}

interface PhaseLogPanelProps {
  jobId: Id<'jobs'>
  /** Sync panel's active tab with the main phase tab */
  activePhaseTab?: LogTab
  /** Which phases are enabled for this job */
  enabledPhases?: LogTab[]
}

export function PhaseLogPanel({
  jobId,
  activePhaseTab,
  enabledPhases = ['all', 'composition', 'generation', 'evaluation'],
}: PhaseLogPanelProps) {
  const {
    height,
    isCollapsed,
    activeLogTab,
    setHeight,
    toggleCollapsed,
    setActiveLogTab,
  } = useLogPanel()

  // Track if user has manually selected a tab (to avoid overriding their choice)
  const userSelectedTab = useRef(false)
  const prevActivePhaseTab = useRef(activePhaseTab)

  // Sync with main phase tab when it changes (only if user hasn't manually selected)
  // Don't sync if activeLogTab is 'all' since that's a special aggregate view
  if (activePhaseTab && activePhaseTab !== prevActivePhaseTab.current && activePhaseTab !== 'all') {
    prevActivePhaseTab.current = activePhaseTab
    if (!userSelectedTab.current && enabledPhases.includes(activePhaseTab) && activeLogTab !== 'all') {
      setActiveLogTab(activePhaseTab)
    }
    // Reset user selection flag when main tab changes
    userSelectedTab.current = false
  }

  // Handle user manually clicking a log tab
  const handleLogTabClick = (tab: LogTab) => {
    userSelectedTab.current = true
    setActiveLogTab(tab)
  }

  // Get phases to filter by based on active tab
  const phasesToFilter = PHASE_LOG_FILTERS[activeLogTab]

  // Query logs with optional phase filtering
  const logs = useQuery(api.logs.getByJobFiltered, {
    jobId,
    phases: phasesToFilter,
    limit: 500,
  })

  // Handle resize via pointer events
  const handleResizeStart = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault()
      const startY = e.clientY
      const startHeight = height

      const handleMove = (moveEvent: PointerEvent) => {
        // Dragging up increases height (startY - currentY is positive when moving up)
        const deltaY = startY - moveEvent.clientY
        setHeight(startHeight + deltaY)
      }

      const handleUp = () => {
        document.removeEventListener('pointermove', handleMove)
        document.removeEventListener('pointerup', handleUp)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }

      document.body.style.cursor = 'row-resize'
      document.body.style.userSelect = 'none'
      document.addEventListener('pointermove', handleMove)
      document.addEventListener('pointerup', handleUp)
    },
    [height, setHeight]
  )

  const collapsedHeight = 44 // Height of just the header bar
  const [copied, setCopied] = useState(false)

  // Format logs for clipboard with nice formatting
  const copyLogsToClipboard = useCallback(() => {
    if (!logs || logs.length === 0) return

    const formattedLogs = logs
      .map((log) => {
        const time = new Date(log.timestamp).toLocaleTimeString('en-US', {
          hour: 'numeric',
          minute: '2-digit',
          second: '2-digit',
          hour12: true,
        })
        return `${time} [${log.level}] [${log.phase}] ${log.message}`
      })
      .join('\n')

    navigator.clipboard.writeText(formattedLogs).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [logs])

  return (
    <div
      className="fixed bottom-0 right-0 border-t bg-background flex flex-col z-10"
      style={{
        height: isCollapsed ? collapsedHeight : height,
        left: '256px' // 16rem = 256px, matching sidebar width exactly
      }}
    >
      {/* Resize handle */}
      {!isCollapsed && (
        <div
          className="h-1 cursor-row-resize bg-border hover:bg-primary/30 transition-colors shrink-0"
          onPointerDown={handleResizeStart}
        />
      )}

      {/* Header with tabs and controls - entire header is clickable to collapse */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b bg-muted/30 shrink-0 cursor-pointer hover:bg-muted/50 transition-colors h-[43px]"
        onClick={toggleCollapsed}
      >
        {/* Phase tabs */}
        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          {enabledPhases.map((phase) => {
            const isActive = activeLogTab === phase
            const colors = PHASE_COLORS[phase]

            return (
              <button
                key={phase}
                onClick={() => handleLogTabClick(phase)}
                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                  isActive
                    ? 'text-white'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
                style={
                  isActive
                    ? { backgroundColor: colors.strong }
                    : undefined
                }
              >
                {phase.toUpperCase()}
              </button>
            )
          })}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          {/* Copy button */}
          <button
            onClick={copyLogsToClipboard}
            className="p-1 rounded hover:bg-muted transition-colors"
            title="Copy logs to clipboard"
          >
            {copied ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4 text-muted-foreground" />
            )}
          </button>

          <span className="text-xs text-muted-foreground font-medium">Logs</span>

          {/* Collapse toggle */}
          <div className="flex items-center">
            {isCollapsed ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </div>
      </div>

      {/* Log content */}
      {!isCollapsed && (
        <LogStream
          logs={logs}
          isLoading={logs === undefined}
          className="flex-1 min-h-0"
        />
      )}
    </div>
  )
}
