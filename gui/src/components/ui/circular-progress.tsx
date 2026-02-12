import * as React from 'react'
import { motion, useSpring, useTransform } from 'framer-motion'

interface CircularProgressProps {
  /** Progress value from 0 to 100 */
  value: number
  /** Size of the circular progress in pixels */
  size?: number
  /** Width of the progress stroke */
  strokeWidth?: number
  /** Color of the progress arc */
  color: string
  /** Background track color */
  trackColor: string
  /** Content to display inside the circle */
  children?: React.ReactNode
  /** Additional className for the container */
  className?: string
}

export function CircularProgress({
  value,
  size = 64,
  strokeWidth = 4,
  color,
  trackColor,
  children,
  className,
}: CircularProgressProps) {
  // Clamp value between 0 and 100
  const clampedValue = Math.max(0, Math.min(100, value))

  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI

  // Spring animation for smooth progress
  const springValue = useSpring(clampedValue, {
    stiffness: 60,
    damping: 15,
    mass: 1,
  })

  // Transform spring value to stroke offset
  const strokeDashoffset = useTransform(
    springValue,
    (latest) => circumference - (latest / 100) * circumference
  )

  // Update spring when value changes
  React.useEffect(() => {
    springValue.set(clampedValue)
  }, [clampedValue, springValue])

  return (
    <div className={`relative ${className ?? ''}`} style={{ width: size, height: size }}>
      <svg
        className="transform -rotate-90"
        width={size}
        height={size}
        aria-valuenow={clampedValue}
        aria-valuemin={0}
        aria-valuemax={100}
        role="progressbar"
      >
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={trackColor}
          strokeWidth={strokeWidth}
        />
        {/* Animated progress arc */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeLinecap="round"
          style={{ strokeDashoffset }}
        />
      </svg>
      {/* Center content */}
      {children && (
        <div className="absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      )}
    </div>
  )
}
