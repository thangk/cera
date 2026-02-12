import { useState, useEffect } from 'react'
import { PYTHON_API_URL } from '../lib/api-urls'

export interface ConstraintValue {
  min: number
  max: number
  step: number
  default: number | number[] | { positive: number; neutral: number; negative: number }
}

export interface UIConstraints {
  version: string
  constraints: {
    age_range: ConstraintValue
    sex_distribution: ConstraintValue
    polarity: ConstraintValue
    sentence_range: ConstraintValue
    review_count: ConstraintValue
    batch_size: ConstraintValue
    request_size: ConstraintValue
    temperature_range: ConstraintValue
    typo_rate: ConstraintValue
    mav_similarity_threshold: ConstraintValue
    mav_max_queries: ConstraintValue
  }
}

// Fallback defaults when API is unavailable
const FALLBACK_CONSTRAINTS: UIConstraints = {
  version: '1.0',
  constraints: {
    age_range: { min: 13, max: 80, step: 1, default: [18, 65] },
    sex_distribution: { min: 0, max: 100, step: 5, default: 50 },
    polarity: {
      min: 0,
      max: 100,
      step: 5,
      default: { positive: 65, neutral: 15, negative: 20 },
    },
    sentence_range: { min: 1, max: 100, step: 1, default: [2, 5] },
    review_count: { min: 10, max: 10000, step: 10, default: 1000 },
    batch_size: { min: 1, max: 100, step: 1, default: 1 },
    request_size: { min: 1, max: 20, step: 1, default: 5 },
    temperature_range: { min: 0.0, max: 2.0, step: 0.1, default: [0.7, 0.9] },
    typo_rate: { min: 0, max: 0.1, step: 0.005, default: 0.01 },
    mav_similarity_threshold: { min: 0.5, max: 1.0, step: 0.05, default: 0.75 },
    mav_max_queries: { min: 10, max: 100, step: 5, default: 30 },
  },
}

export function useUIConstraints() {
  const [constraints, setConstraints] = useState<UIConstraints>(FALLBACK_CONSTRAINTS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchConstraints = async () => {
      const pythonApiUrl = PYTHON_API_URL

      try {
        const response = await fetch(`${pythonApiUrl}/api/config/ui-constraints`)

        if (!response.ok) {
          throw new Error('Failed to fetch UI constraints')
        }

        const data: UIConstraints = await response.json()
        setConstraints(data)
        setError(null)
      } catch (err) {
        // Use fallback defaults silently - this is expected when API is not running
        console.debug(
          'Using fallback UI constraints (Python API unavailable):',
          err instanceof Error ? err.message : 'Unknown error'
        )
        setConstraints(FALLBACK_CONSTRAINTS)
        // Don't set error state - fallbacks are valid
      } finally {
        setLoading(false)
      }
    }

    fetchConstraints()
  }, [])

  return { constraints, loading, error }
}
