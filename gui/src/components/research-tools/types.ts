/** Research Tools type definitions */

// ==========================================
// Core Metric Keys
// ==========================================

export const MDQA_METRIC_KEYS = ['bleu', 'rouge_l', 'bertscore', 'moverscore', 'distinct_1', 'distinct_2', 'self_bleu'] as const
export type MdqaMetricKey = typeof MDQA_METRIC_KEYS[number]

export const MDQA_METRIC_LABELS: Record<MdqaMetricKey, string> = {
  bleu: 'BLEU',
  rouge_l: 'ROUGE-L',
  bertscore: 'BERTScore',
  moverscore: 'MoverScore',
  distinct_1: 'Distinct-1',
  distinct_2: 'Distinct-2',
  self_bleu: 'Self-BLEU',
}

/** Direction indicator for metrics (higher = better, except Self-BLEU) */
export const MDQA_METRIC_DIRECTION: Record<MdqaMetricKey, 'up' | 'down'> = {
  bleu: 'up',
  rouge_l: 'up',
  bertscore: 'up',
  moverscore: 'up',
  distinct_1: 'up',
  distinct_2: 'up',
  self_bleu: 'down',
}

export const LADY_METRIC_KEYS = ['precision_at_5', 'map_at_5', 'ndcg_at_5', 'recall_at_5', 'specificity_at_5'] as const
export type LadyMetricKey = typeof LADY_METRIC_KEYS[number]

export const LADY_METRIC_LABELS: Record<LadyMetricKey, string> = {
  precision_at_5: 'P@5',
  map_at_5: 'MAP@5',
  ndcg_at_5: 'NDCG@5',
  recall_at_5: 'R@5',
  specificity_at_5: 'S@5',
}

/** Direction indicator for LADy metrics (all higher = better) */
export const LADY_METRIC_DIRECTION: Record<LadyMetricKey, 'up' | 'down'> = {
  precision_at_5: 'up',
  map_at_5: 'up',
  ndcg_at_5: 'up',
  recall_at_5: 'up',
  specificity_at_5: 'up',
}

// ==========================================
// Method & RQ Types
// ==========================================

export type Method = 'real' | 'cera' | 'heuristic'

export const METHOD_LABELS: Record<Method, string> = {
  real: 'Real',
  cera: 'CERA',
  heuristic: 'Heuristic',
}

export type RqId = 'rq1' | 'rq2' | 'rq3' | 'rq4'
export type Rq1TableId = '1a' | '1b' | '1c'

// ==========================================
// Metric Value Types
// ==========================================

export interface MetricStat {
  mean: number
  std?: number
}

export interface PerRunMetrics {
  run: number
  metrics: Partial<Record<MdqaMetricKey, number>>
}

export interface PerRunLadyMetrics {
  run: number
  metrics: Partial<Record<LadyMetricKey, number>>
}

export interface PerModelMetrics {
  model: string
  modelSlug: string
  metrics: Partial<Record<MdqaMetricKey, number>>
  runs?: Array<{
    run: number
    metrics: Partial<Record<MdqaMetricKey, number>>
  }>
}

// ==========================================
// Data Source Types
// ==========================================

export interface JobSource {
  type: 'job'
  jobId: string
  jobName: string
  mdqaMetrics: Partial<Record<MdqaMetricKey, MetricStat>> | null
  perRunMetrics: PerRunMetrics[] | null
  perModelMetrics: PerModelMetrics[] | null
  ladyMetrics: Partial<Record<LadyMetricKey, MetricStat>> | null
  perRunLadyMetrics: PerRunLadyMetrics[] | null
  nRuns?: number
}

export interface FileSource {
  type: 'file'
  fileName: string
  mdqaMetrics: Partial<Record<MdqaMetricKey, MetricStat>> | null
  perRunMetrics: PerRunMetrics[] | null
  perModelMetrics: PerModelMetrics[] | null
  ladyMetrics: Partial<Record<LadyMetricKey, MetricStat>> | null
  perRunLadyMetrics: PerRunLadyMetrics[] | null
  nRuns?: number
}

export interface DataEntry {
  id: string
  method: Method
  size: number
  modelSlug?: string
  source: JobSource | FileSource
}

// ==========================================
// Scan Targets API Types
// ==========================================

export interface ScannedTarget {
  targetValue: number
  countMode: string
  hasMetrics: boolean
  metricsFiles: string[]
  modelSlugs: string[]
}

export interface ScanTargetsResponse {
  targets: ScannedTarget[]
  isMultiModel: boolean
  totalTargets: number
}

// ==========================================
// LADy Scan/Read API Types
// ==========================================

export interface LadyOutputDir {
  name: string
  path: string
  type: Method
  targets: number[]
}

export interface LadyScanResponse {
  outputs: LadyOutputDir[]
}

export interface LadyReadMetricsResponse {
  metrics: Partial<Record<LadyMetricKey, MetricStat>>
  perRun: PerRunLadyMetrics[] | null
  nRuns: number
}

// ==========================================
// Table Rendering Types
// ==========================================

export type AxisOrientation = 'default' | 'swapped'

export interface TableData {
  rowHeaders: string[]
  columnHeaders: string[]
  cells: string[][]
  columnGroups?: { label: string; span: number }[]
  columnSubLabels?: (string | null)[]
  /** Raw numeric values for each cell (for color coding). Same shape as cells. */
  cellValues?: (number | null)[][]
  /** Direction for each row metric: 'up' = higher is better, 'down' = lower is better. */
  metricDirections?: ('up' | 'down')[]
}

// ==========================================
// Statistical Types
// ==========================================

export interface StatisticalResult {
  metric: string
  size: number
  ceraMean: number
  heuristicMean: number
  delta: number
  tStat: number
  df: number
  pValue: number
  cohensD: number
  significance: '***' | '**' | '*' | 'ns'
}

// ==========================================
// RQ Definitions
// ==========================================

export interface RqDefinition {
  id: RqId
  title: string
  shortTitle: string
  description: string
  tables: { id: string; label: string; description: string }[]
}

export const RQ_DEFINITIONS: RqDefinition[] = [
  {
    id: 'rq1',
    title: 'RQ1: Framework Effectiveness',
    shortTitle: 'RQ1',
    description: 'Compare CERA vs Heuristic baseline across dataset sizes (Laptop domain)',
    tables: [
      { id: '1a', label: 'MDQA Metrics', description: 'Intrinsic quality: BLEU, ROUGE-L, BERTScore, MoverScore, Distinct-1/2, Self-BLEU' },
      { id: '1b', label: 'LADy-kap Metrics', description: 'Extrinsic utility: P@5, MAP@5, NDCG@5, R@5, S@5' },
      { id: '1c', label: 'Statistical Significance', description: 'Paired t-test, Cohen\'s d between CERA and Heuristic' },
    ],
  },
  {
    id: 'rq2',
    title: 'RQ2: Component Effectiveness',
    shortTitle: 'RQ2',
    description: 'Ablation study of CERA components (Restaurant domain)',
    tables: [
      { id: '2a', label: 'MDQA Metrics', description: 'Intrinsic quality across ablation conditions' },
      { id: '2b', label: 'LADy-kap Metrics', description: 'Extrinsic utility across ablation conditions' },
      { id: '2c', label: 'MDQA Significance', description: 'Statistical significance of ablations (MDQA)' },
      { id: '2d', label: 'LADy-kap Significance', description: 'Statistical significance of ablations (LADy-kap)' },
      { id: '2e', label: 'Impact Ranking', description: 'Component impact ranking by NDCG@5 degradation' },
    ],
  },
  {
    id: 'rq3',
    title: 'RQ3: Scaling',
    shortTitle: 'RQ3',
    description: 'Size vs performance relationship (Hotel domain)',
    tables: [
      { id: '3a', label: 'Reference Metrics', description: 'MDQA reference-based metrics across sizes' },
      { id: '3b', label: 'Diversity Metrics', description: 'MDQA diversity metrics: CERA vs Real' },
      { id: '3c', label: 'LADy-kap CERA', description: 'LADy-kap metrics for CERA across sizes' },
      { id: '3d', label: 'LADy-kap Real', description: 'LADy-kap metrics for Real data across sizes' },
      { id: '3e', label: 'Gap Analysis', description: 'CERA vs Real gap for NDCG@5' },
      { id: '3f', label: 'Correlations', description: 'Pearson correlation: log(size) vs metric scores' },
    ],
  },
  {
    id: 'rq4',
    title: 'RQ4: Generalizability',
    shortTitle: 'RQ4',
    description: 'Cross-domain comparison (Restaurant, Laptop, Hotel)',
    tables: [
      { id: '4a', label: 'MDQA Metrics', description: 'Intrinsic quality across domains' },
      { id: '4b', label: 'LADy-kap Metrics', description: 'Extrinsic utility across domains' },
      { id: '4c', label: 'MDQA Significance', description: 'Per-domain statistical significance (MDQA)' },
      { id: '4d', label: 'LADy-kap Significance', description: 'Per-domain statistical significance (LADy-kap)' },
      { id: '4e', label: 'Consistency', description: 'Cross-domain consistency (CV of deltas)' },
    ],
  },
]

// ==========================================
// URL Search State
// ==========================================

export interface ResearchToolsSearch {
  rq?: RqId
  table?: string
}
