'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { Box, CircularProgress, Alert } from '@mui/material'

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import('react-apexcharts'), { ssr: false })

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

interface PRChartData {
  precision: number[]
  recall: number[]
  thresholds: number[]
}

export default function PRChart() {
  const [data, setData] = useState<PRChartData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/curves/pr`)
        if (!response.ok) {
          throw new Error('Failed to fetch PR curve data')
        }
        const prData = await response.json()
        setData(prData)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <Alert severity="error">{error}</Alert>
      </Box>
    )
  }

  if (!data || !data.precision || !data.recall) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <Alert severity="warning">No PR curve data available</Alert>
      </Box>
    )
  }

  const chartOptions = {
    chart: {
      type: 'line' as const,
      toolbar: {
        show: false
      },
      zoom: {
        enabled: false
      }
    },
    stroke: {
      curve: 'smooth' as const,
      width: 3
    },
    colors: ['#d32f2f'],
    xaxis: {
      title: {
        text: 'Recall',
        style: {
          fontSize: '14px',
          fontWeight: 600
        }
      },
      min: 0,
      max: 1,
      tickAmount: 5
    },
    yaxis: {
      title: {
        text: 'Precision',
        style: {
          fontSize: '14px',
          fontWeight: 600
        }
      },
      min: 0,
      max: 1,
      tickAmount: 5
    },
    grid: {
      borderColor: '#e0e0e0',
      strokeDashArray: 5
    },
    tooltip: {
      x: {
        formatter: (value: number) => `Recall: ${value.toFixed(3)}`
      },
      y: {
        formatter: (value: number) => `Precision: ${value.toFixed(3)}`
      }
    },
    annotations: {
      yaxis: [
        {
          y: 0.8,
          y2: 0.8,
          borderColor: '#ff9800',
          fillColor: '#ff9800',
          opacity: 0.1,
          label: {
            borderColor: '#ff9800',
            style: {
              color: '#fff',
              background: '#ff9800'
            },
            text: 'High Precision Target'
          }
        }
      ]
    }
  }

  const chartSeries = [
    {
      name: 'Precision-Recall',
      data: data.precision.map((p, i) => ({
        x: data.recall[i],
        y: p
      }))
    }
  ]

  return (
    <Box>
      <Chart
        options={chartOptions}
        series={chartSeries}
        type="line"
        height={300}
      />
    </Box>
  )
}
