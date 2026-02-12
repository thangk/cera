import { useMemo } from 'react'
import { ChevronDown, Star, Layers } from 'lucide-react'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu'
import type { Id } from 'convex/_generated/dataModel'
import type { ProcessedModel } from '../hooks/use-openrouter-models'

export interface LLMPreset {
  _id: Id<"llm_presets">
  name: string
  isDefault: boolean
  rdeModel?: string
  mavModels?: string[]
  savModel?: string
  genModel?: string
}

interface PresetSelectorProps {
  presets: LLMPreset[] | undefined
  selectedPresetId: Id<"llm_presets"> | null
  onSelect: (preset: LLMPreset) => void
  onClear: () => void
  processedModels: ProcessedModel[]
  loading?: boolean
}

export function PresetSelector({
  presets,
  selectedPresetId,
  onSelect,
  onClear,
  processedModels,
  loading,
}: PresetSelectorProps) {
  const selectedPreset = useMemo(() => {
    if (!selectedPresetId || !presets) return null
    return presets.find(p => p._id === selectedPresetId)
  }, [selectedPresetId, presets])

  // Check if any models in preset are unavailable
  const hasUnavailableModels = useMemo(() => {
    if (!selectedPreset) return false
    const modelIds = new Set(processedModels.map(m => m.id))
    const presetModels = [
      selectedPreset.rdeModel,
      ...(selectedPreset.mavModels || []),
      selectedPreset.savModel,
      selectedPreset.genModel,
    ].filter(Boolean)
    return presetModels.some(m => !modelIds.has(m!))
  }, [selectedPreset, processedModels])

  if (!presets || presets.length === 0) {
    return null // Don't show if no presets exist
  }

  return (
    <div className="flex items-center gap-2">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className="justify-between min-w-[200px]"
            disabled={loading}
          >
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-muted-foreground" />
              {selectedPreset ? (
                <span className="flex items-center gap-1.5">
                  {selectedPreset.name}
                  {selectedPreset.isDefault && (
                    <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                  )}
                  {hasUnavailableModels && (
                    <Badge variant="destructive" className="text-[9px] px-1">!</Badge>
                  )}
                </span>
              ) : (
                <span className="text-muted-foreground">Select preset...</span>
              )}
            </div>
            <ChevronDown className="h-4 w-4 ml-2" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[250px]">
          {presets.map((preset) => (
            <DropdownMenuItem
              key={preset._id}
              onClick={() => onSelect(preset)}
              className="flex items-center justify-between"
            >
              <span>{preset.name}</span>
              {preset.isDefault && (
                <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
              )}
            </DropdownMenuItem>
          ))}
          {selectedPreset && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onClear}>
                Clear selection
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      {hasUnavailableModels && (
        <span className="text-xs text-destructive">
          Some models unavailable
        </span>
      )}
    </div>
  )
}
