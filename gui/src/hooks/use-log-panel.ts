import { create } from 'zustand'

export type LogTab = 'all' | 'composition' | 'generation' | 'evaluation'

interface LogPanelState {
  // Panel geometry
  height: number
  isCollapsed: boolean

  // Active tab (also acts as filter)
  activeLogTab: LogTab

  // Actions
  setHeight: (height: number) => void
  toggleCollapsed: () => void
  setCollapsed: (collapsed: boolean) => void
  setActiveLogTab: (tab: LogTab) => void
}

const MIN_HEIGHT = 80
const MAX_HEIGHT_RATIO = 0.5 // 50vh
const DEFAULT_HEIGHT = 200

export const useLogPanel = create<LogPanelState>((set) => ({
  height: DEFAULT_HEIGHT,
  isCollapsed: false,
  activeLogTab: 'all',

  setHeight: (height) =>
    set(() => {
      // Clamp height within bounds
      const maxHeight = typeof window !== 'undefined' ? window.innerHeight * MAX_HEIGHT_RATIO : 400
      const clampedHeight = Math.max(MIN_HEIGHT, Math.min(height, maxHeight))
      return { height: clampedHeight }
    }),

  toggleCollapsed: () => set((state) => ({ isCollapsed: !state.isCollapsed })),

  setCollapsed: (collapsed) => set({ isCollapsed: collapsed }),

  setActiveLogTab: (tab) => set({ activeLogTab: tab }),
}))

// Phase to log phase mapping for filtering
// 'all' = undefined means no filtering (show all logs)
export const PHASE_LOG_FILTERS: Record<LogTab, string[] | undefined> = {
  all: undefined,
  composition: ['SIL', 'MAV', 'RGM', 'ACM', 'Pipeline', 'composition'],
  generation: ['AML', 'Pipeline', 'generation', 'Config'],
  evaluation: ['MDQA', 'Pipeline', 'evaluation'],
}
