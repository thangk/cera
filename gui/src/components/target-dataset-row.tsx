import { Input } from './ui/input'
import { Label } from './ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip'
import { X, HelpCircle } from 'lucide-react'

// ==========================================
// Types
// ==========================================

export interface CeraTarget {
  count_mode: 'sentences' | 'reviews'
  target_value: number
  batch_size: number
  request_size: number
  total_runs: number
  runs_mode: 'parallel' | 'sequential'
  neb_depth: number // 0 = disabled
}

export interface HeuristicTarget {
  targetMode: 'sentences' | 'reviews'
  targetValue: number
  reviewsPerBatch: number // = batch_size equivalent
  requestSize: number
  totalRuns: number
  runsMode: 'parallel' | 'sequential'
}

export const DEFAULT_CERA_TARGET: CeraTarget = {
  count_mode: 'sentences',
  target_value: 100,
  batch_size: 1,
  request_size: 25,
  total_runs: 1,
  runs_mode: 'parallel',
  neb_depth: 0,
}

export const DEFAULT_HEURISTIC_TARGET: HeuristicTarget = {
  targetMode: 'sentences',
  targetValue: 100,
  reviewsPerBatch: 1,
  requestSize: 25,
  totalRuns: 1,
  runsMode: 'parallel',
}

// ==========================================
// CERA Target Row
// ==========================================

interface CeraTargetRowProps {
  index: number
  target: CeraTarget
  onChange: (index: number, updated: CeraTarget) => void
  onRemove: (index: number) => void
  canRemove: boolean
  /** Sentence length range from Attributes Profile [min, max] - used to estimate review count */
  lengthRange?: [number, number]
}

export function CeraTargetRow({ index, target, onChange, onRemove, canRemove, lengthRange }: CeraTargetRowProps) {
  const modeLabel = target.count_mode === 'sentences' ? 'Sentences' : 'Reviews'

  const update = (field: keyof CeraTarget, value: any) => {
    onChange(index, { ...target, [field]: value })
  }

  const nebBuffer = target.neb_depth * target.request_size

  // Estimate review count when in sentences mode
  const avgSentencesPerReview = lengthRange ? (lengthRange[0] + lengthRange[1]) / 2 : 0
  const estimatedReviews = target.count_mode === 'sentences' && avgSentencesPerReview > 0
    ? Math.ceil(target.target_value / avgSentencesPerReview)
    : null

  return (
    <div className="flex items-end gap-2 p-3 rounded-lg border bg-muted/20">
      <TooltipProvider>
        {/* Target Mode */}
        <div className="space-y-1">
          {index === 0 && <Label className="text-xs text-muted-foreground">Mode</Label>}
          <Select
            value={target.count_mode}
            onValueChange={(v: 'sentences' | 'reviews') => update('count_mode', v)}
          >
            <SelectTrigger className="w-[120px] h-9 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sentences">Sentences</SelectItem>
              <SelectItem value="reviews">Reviews</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Target Value */}
        <div className="space-y-1">
          {index === 0 && <Label className="text-xs text-muted-foreground">Target {modeLabel}</Label>}
          <div className="relative">
            <Input
              type="number"
              min={1}
              step={50}
              value={target.target_value}
              onChange={(e) => update('target_value', parseInt(e.target.value) || 100)}
              className="w-[100px] h-9 text-xs"
            />
            {estimatedReviews !== null && (
              <span className="absolute -bottom-4 left-0 text-[10px] text-muted-foreground whitespace-nowrap">
                ~{estimatedReviews} reviews
              </span>
            )}
          </div>
        </div>

        {/* Batch Size */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Batch</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">AML prompts per API request</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <Input
            type="number"
            min={1}
            max={100}
            value={target.batch_size}
            onChange={(e) => update('batch_size', Math.max(1, parseInt(e.target.value) || 1))}
            className="w-[70px] h-9 text-xs"
          />
        </div>

        {/* Request Size */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Requests</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Parallel API calls</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <Input
            type="number"
            min={1}
            max={100}
            value={target.request_size}
            onChange={(e) => update('request_size', Math.max(1, parseInt(e.target.value) || 25))}
            className="w-[70px] h-9 text-xs"
          />
        </div>

        {/* Total Runs + Runs Mode (grouped) */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Runs</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Multiple runs for research variability. Parallel runs all N concurrently.</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Input
              type="number"
              min={1}
              max={10}
              value={target.total_runs}
              onChange={(e) => update('total_runs', Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
              className="w-[55px] h-9 text-xs"
            />
            <Select
              value={target.runs_mode}
              onValueChange={(v: 'parallel' | 'sequential') => update('runs_mode', v)}
            >
              <SelectTrigger className="w-[100px] h-9 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="parallel">Parallel</SelectItem>
                <SelectItem value="sequential">Sequential</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* NEB Depth */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">NEB</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">
                    Negative Example Buffer depth. 0 = disabled.
                    {target.neb_depth > 0 && ` Buffer = ${nebBuffer} reviews.`}
                    {nebBuffer >= 50 && ' Large buffer — ensure model supports 1M+ context.'}
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <Input
            type="number"
            min={0}
            value={target.neb_depth}
            onChange={(e) => update('neb_depth', Math.max(0, parseInt(e.target.value) || 0))}
            className="w-[60px] h-9 text-xs"
          />
        </div>
      </TooltipProvider>

      {/* Remove button */}
      {canRemove && (
        <button
          type="button"
          className="text-muted-foreground hover:text-destructive shrink-0 pb-0.5"
          onClick={() => onRemove(index)}
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}

// ==========================================
// Heuristic Target Row
// ==========================================

interface HeuristicTargetRowProps {
  index: number
  target: HeuristicTarget
  onChange: (index: number, updated: HeuristicTarget) => void
  onRemove: (index: number) => void
  canRemove: boolean
  /** Avg sentences per review string from heuristic config (e.g. "4-7" or "5") */
  avgSentencesPerReview?: string
}

export function HeuristicTargetRow({ index, target, onChange, onRemove, canRemove, avgSentencesPerReview }: HeuristicTargetRowProps) {
  const modeLabel = target.targetMode === 'sentences' ? 'Sentences' : 'Reviews'

  const update = (field: keyof HeuristicTarget, value: any) => {
    onChange(index, { ...target, [field]: value })
  }

  // Parse avg sentences for estimate
  let avgSent = 0
  if (avgSentencesPerReview) {
    const rangeMatch = avgSentencesPerReview.match(/^(\d+)\s*-\s*(\d+)$/)
    if (rangeMatch) {
      avgSent = (parseInt(rangeMatch[1]) + parseInt(rangeMatch[2])) / 2
    } else {
      avgSent = parseFloat(avgSentencesPerReview) || 0
    }
  }
  const estimatedReviews = target.targetMode === 'sentences' && avgSent > 0
    ? Math.ceil(target.targetValue / avgSent)
    : null

  return (
    <div className="flex items-end gap-2 p-3 rounded-lg border bg-muted/20">
      <TooltipProvider>
        {/* Target Mode */}
        <div className="space-y-1">
          {index === 0 && <Label className="text-xs text-muted-foreground">Mode</Label>}
          <Select
            value={target.targetMode}
            onValueChange={(v: 'sentences' | 'reviews') => update('targetMode', v)}
          >
            <SelectTrigger className="w-[120px] h-9 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sentences">Sentences</SelectItem>
              <SelectItem value="reviews">Reviews</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Target Value */}
        <div className="space-y-1">
          {index === 0 && <Label className="text-xs text-muted-foreground">Target {modeLabel}</Label>}
          <div className="relative">
            <Input
              type="number"
              min={1}
              step={50}
              value={target.targetValue}
              onChange={(e) => update('targetValue', parseInt(e.target.value) || 100)}
              className="w-[100px] h-9 text-xs"
            />
            {estimatedReviews !== null && (
              <span className="absolute -bottom-4 left-0 text-[10px] text-muted-foreground whitespace-nowrap">
                ~{estimatedReviews} reviews
              </span>
            )}
          </div>
        </div>

        {/* Batch Size (reviewsPerBatch) */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Batch</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Reviews per batch (per API request)</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <Input
            type="number"
            min={1}
            max={100}
            value={target.reviewsPerBatch}
            onChange={(e) => update('reviewsPerBatch', Math.max(1, parseInt(e.target.value) || 25))}
            className="w-[70px] h-9 text-xs"
          />
        </div>

        {/* Request Size */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Requests</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Parallel batch requests</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <Input
            type="number"
            min={1}
            value={target.requestSize}
            onChange={(e) => update('requestSize', Math.max(1, parseInt(e.target.value) || 3))}
            className="w-[70px] h-9 text-xs"
          />
        </div>

        {/* Total Runs + Runs Mode (grouped) */}
        <div className="space-y-1">
          {index === 0 && (
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Runs</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-xs">Multiple runs for research variability. Parallel runs all N concurrently.</p>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Input
              type="number"
              min={1}
              max={10}
              value={target.totalRuns}
              onChange={(e) => update('totalRuns', Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))}
              className="w-[55px] h-9 text-xs"
            />
            <Select
              value={target.runsMode}
              onValueChange={(v: 'parallel' | 'sequential') => update('runsMode', v)}
            >
              <SelectTrigger className="w-[100px] h-9 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="parallel">Parallel</SelectItem>
                <SelectItem value="sequential">Sequential</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </TooltipProvider>

      {/* Remove button */}
      {canRemove && (
        <button
          type="button"
          className="text-muted-foreground hover:text-destructive shrink-0 pb-0.5"
          onClick={() => onRemove(index)}
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  )
}
