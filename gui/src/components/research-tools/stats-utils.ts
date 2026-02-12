/** Statistical calculation utilities for Research Tools */

// @ts-expect-error jstat has no type declarations
import jstatModule from 'jstat'

const jStat: {
  mean(arr: number[]): number
  stdev(arr: number[], isSample: boolean): number
  studentt: { cdf(x: number, df: number): number }
} = jstatModule.jStat

/**
 * Paired two-tailed t-test.
 * Requires equal-length arrays of paired observations.
 */
export function pairedTTest(a: number[], b: number[]): { t: number; p: number; df: number } {
  if (a.length !== b.length || a.length < 2) {
    return { t: 0, p: 1, df: 0 }
  }

  const n = a.length
  const diffs = a.map((v, i) => v - b[i])
  const meanDiff = jStat.mean(diffs)
  const sdDiff = jStat.stdev(diffs, true) // sample SD (Bessel's correction)

  if (sdDiff === 0) {
    return { t: 0, p: 1, df: n - 1 }
  }

  const t = meanDiff / (sdDiff / Math.sqrt(n))
  const df = n - 1
  const p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df))

  return { t, p, df }
}

/**
 * Cohen's d for paired samples (using SD of differences).
 */
export function cohensD(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length < 2) return 0

  const diffs = a.map((v, i) => v - b[i])
  const meanDiff = jStat.mean(diffs)
  const sdDiff = jStat.stdev(diffs, true)

  return sdDiff === 0 ? 0 : meanDiff / sdDiff
}

/**
 * Significance level stars from p-value.
 */
export function significanceStars(p: number): '***' | '**' | '*' | 'ns' {
  if (p < 0.001) return '***'
  if (p < 0.01) return '**'
  if (p < 0.05) return '*'
  return 'ns'
}

/**
 * Pearson correlation coefficient and p-value.
 */
export function pearsonCorrelation(x: number[], y: number[]): { r: number; p: number } {
  if (x.length !== y.length || x.length < 3) {
    return { r: 0, p: 1 }
  }

  const n = x.length
  const meanX = jStat.mean(x)
  const meanY = jStat.mean(y)

  let sumXY = 0, sumX2 = 0, sumY2 = 0
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX
    const dy = y[i] - meanY
    sumXY += dx * dy
    sumX2 += dx * dx
    sumY2 += dy * dy
  }

  if (sumX2 === 0 || sumY2 === 0) return { r: 0, p: 1 }

  const r = sumXY / Math.sqrt(sumX2 * sumY2)
  const t = r * Math.sqrt((n - 2) / (1 - r * r))
  const df = n - 2
  const p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df))

  return { r, p }
}

/**
 * Compute mean of an array.
 */
export function mean(arr: number[]): number {
  if (arr.length === 0) return 0
  return jStat.mean(arr)
}

/**
 * Compute sample standard deviation.
 */
export function stdev(arr: number[]): number {
  if (arr.length < 2) return 0
  return jStat.stdev(arr, true)
}
