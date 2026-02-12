import { useRef } from 'react'
import { ArrowDown } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'
import { Button } from '@/components/ui/button'
import { useAutoScroll } from '@/hooks/use-auto-scroll'

interface LogEntry {
  _id: string
  timestamp: number
  level: 'INFO' | 'WARN' | 'ERROR'
  phase: string
  message: string
}

interface LogStreamProps {
  logs: LogEntry[] | undefined
  isLoading?: boolean
  className?: string
}

export function LogStream({ logs, isLoading, className }: LogStreamProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const { showJumpButton, jumpToLatest } = useAutoScroll(scrollRef, [logs?.length])

  if (isLoading) {
    return (
      <div className={`p-4 space-y-2 ${className}`}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} className="h-4 w-full" />
        ))}
      </div>
    )
  }

  if (!logs || logs.length === 0) {
    return (
      <div className={`flex items-center justify-center py-8 text-sm text-muted-foreground ${className}`}>
        No logs yet
      </div>
    )
  }

  return (
    <div className={`relative ${className}`}>
      <div
        ref={scrollRef}
        className="h-full overflow-y-auto p-3 font-mono text-xs"
      >
        <div className="space-y-0.5">
          {logs.map((log) => (
            <div key={log._id} className="flex gap-2 leading-relaxed">
              <span className="text-muted-foreground shrink-0">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span
                className={`shrink-0 font-medium ${
                  log.level === 'ERROR'
                    ? 'text-red-500'
                    : log.level === 'WARN'
                      ? 'text-yellow-500'
                      : 'text-blue-500'
                }`}
              >
                [{log.level}]
              </span>
              <span className="text-muted-foreground shrink-0">[{log.phase}]</span>
              <span className="text-foreground break-all">{log.message}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Jump to latest button */}
      {showJumpButton && (
        <Button
          size="sm"
          variant="secondary"
          className="absolute bottom-3 right-3 gap-1 shadow-md"
          onClick={jumpToLatest}
        >
          <ArrowDown className="h-3 w-3" />
          Jump to latest
        </Button>
      )}
    </div>
  )
}
