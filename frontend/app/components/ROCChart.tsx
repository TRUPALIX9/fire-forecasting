'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { Box, CircularProgress, Alert } from '@mui/material'

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import('react-apexcharts'), { ssr: false })

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

interface ROCChartData {
  fpr: number[]
  tpr: number[]
  thresholds: number[]
}

export default function ROCChart() {
  const [data, setData] = useState<ROCChartData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/curves/roc`)
        if (!response.ok) {
          throw new Error('Failed to fetch ROC curve data')
        }
        const rocData = await response.json()
        setData(rocData)
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

  if (!data || !data.fpr || !data.tpr) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
        <Alert severity="warning">No ROC curve data available</Alert>
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
    colors: ['#1976d2'],
    xaxis: {
      title: {
        text: 'False Positive Rate',
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
        text: 'True Positive Rate',
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
        formatter: (value: number) => `FPR: ${value.toFixed(3)}`
      },
      y: {
        formatter: (value: number) => `TPR: ${value.toFixed(3)}`
      }
    },
    annotations: {
      xaxis: [
        {
          x: 0.5,
          x2: 0.5,
          borderColor: '#ff9800',
          fillColor: '#ff9800',
          opacity: 0.1,
          label: {
            borderColor: '#ff9800',
            style: {
              color: '#fff',
              background: '#ff9800'
            },
            text: 'Random Classifier'
          }
        }
      ],
      yaxis: [
        {
          y: 0.5,
          y2: 0.5,
          borderColor: '#ff9800',
          fillColor: '#ff9800',
          opacity: 0.1,
          label: {
            borderColor: '#ff9800',
            style: {
              color: '#fff',
              background: '#ff9800'
            },
            text: 'Random Classifier'
          }
        }
      ]
    }
  }

  const chartSeries = [
    {
      name: 'ROC Curve',
      data: data.tpr.map((tpr, i) => ({
        x: data.fpr[i],
        y: tpr
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
