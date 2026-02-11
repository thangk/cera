import { useEffect, useState, useRef, type RefObject } from 'react'

interface UseAutoScrollOptions {
  /** Threshold in pixels from bottom to consider "at bottom" */
  threshold?: number
}

interface UseAutoScrollReturn {
  /** Whether auto-scroll is currently active */
  isAutoScrollActive: boolean
  /** Whether to show the "Jump to latest" button */
  showJumpButton: boolean
  /** Function to jump to the latest content and resume auto-scroll */
  jumpToLatest: () => void
}

/**
 * Hook for managing auto-scroll behavior in a scrollable container.
 * Auto-scrolls to bottom when new content arrives, but pauses when user scrolls up.
 */
export function useAutoScroll(
  scrollRef: RefObject<HTMLDivElement | null>,
  deps: unknown[],
  options: UseAutoScrollOptions = {}
): UseAutoScrollReturn {
  const { threshold = 40 } = options

  const [isAutoScrollActive, setIsAutoScrollActive] = useState(true)
  const [showJumpButton, setShowJumpButton] = useState(false)
  const lastScrollTop = useRef(0)
  const isInitialMount = useRef(true)

  // Detect user manual scroll
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el
      const isAtBottom = scrollHeight - scrollTop - clientHeight < threshold

      // User scrolled UP (away from bottom) - pause auto-scroll
      if (scrollTop < lastScrollTop.current && !isAtBottom) {
        setIsAutoScrollActive(false)
        setShowJumpButton(true)
      }

      // User scrolled back to bottom - resume auto-scroll
      if (isAtBottom) {
        setIsAutoScrollActive(true)
        setShowJumpButton(false)
      }

      lastScrollTop.current = scrollTop
    }

    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el.removeEventListener('scroll', handleScroll)
  }, [scrollRef, threshold])

  // Auto-scroll when deps change (new content arrives)
  useEffect(() => {
    // Skip initial mount to avoid scrolling before content is rendered
    if (isInitialMount.current) {
      isInitialMount.current = false
      // Still scroll to bottom on initial mount
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      }
      return
    }

    if (isAutoScrollActive && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [deps, isAutoScrollActive, scrollRef])

  const jumpToLatest = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      setIsAutoScrollActive(true)
      setShowJumpButton(false)
    }
  }

  return { isAutoScrollActive, showJumpButton, jumpToLatest }
}
