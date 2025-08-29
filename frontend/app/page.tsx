'use client'

import { useState, useEffect } from 'react'
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Button, 
  Box,
  Alert,
  CircularProgress,
  Slider,
  FormControlLabel,
  Switch
} from '@mui/material'
import { Refresh, Download } from '@mui/icons-material'
import dynamic from 'next/dynamic'
import useSWR from 'swr'

// Dynamically import charts to avoid SSR issues
const PRChart = dynamic(() => import('./components/PRChart'), { ssr: false })
const ROCChart = dynamic(() => import('./components/ROCChart'), { ssr: false })
const ConfusionMatrix = dynamic(() => import('./components/ConfusionMatrix'), { ssr: false })

// API client
const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

const fetcher = (url: string) => fetch(url).then(res => res.json())

export default function Dashboard() {
  const [threshold, setThreshold] = useState(0.5)
  const [showConfusionMatrix, setShowConfusionMatrix] = useState(true)
  const [isTraining, setIsTraining] = useState(false)

  // Fetch data
  const { data: status, error: statusError } = useSWR(`${API_BASE}/api/status`, fetcher)
  const { data: metrics, error: metricsError } = useSWR(`${API_BASE}/api/metrics/global`, fetcher)
  const { data: confusionMatrix, error: cmError } = useSWR(
    `${API_BASE}/api/confusion?threshold=${threshold}`, 
    fetcher
  )

  // Handle training
  const handleTrain = async () => {
    setIsTraining(true)
    try {
      const response = await fetch(`${API_BASE}/api/train`, { method: 'POST' })
      if (response.ok) {
        // Wait a bit then refresh
        setTimeout(() => {
          window.location.reload()
        }, 2000)
      }
    } catch (error) {
      console.error('Training failed:', error)
    } finally {
      setIsTraining(false)
    }
  }

  // Handle threshold change
  const handleThresholdChange = (event: Event, newValue: number | number[]) => {
    setThreshold(newValue as number)
  }

  if (statusError) return <Alert severity="error">Failed to load API status</Alert>
  if (!status) return <CircularProgress />

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Fire Forecasting Dashboard
      </Typography>
      
      {/* Status and Actions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Typography color="text.secondary">
                Model: {status.model_available ? '✅ Available' : '❌ Not Available'}
              </Typography>
              <Typography color="text.secondary">
                Last Run: {status.last_run || 'Never'}
              </Typography>
              <Typography color="text.secondary">
                Total Rows: {status.rows_total || 'Unknown'}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  variant="contained"
                  startIcon={<Refresh />}
                  onClick={handleTrain}
                  disabled={isTraining || status.pipeline_running}
                >
                  {isTraining ? 'Training...' : 'Retrain Model'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  href="/artifacts"
                  target="_blank"
                >
                  Download Artifacts
                </Button>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* KPI Cards */}
      {metrics && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  PR-AUC
                </Typography>
                <Typography variant="h4" component="div">
                  {metrics.ann?.pr_auc?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metrics.improvement?.pr_auc && (
                    <span style={{ color: metrics.improvement.pr_auc > 0 ? 'green' : 'red' }}>
                      {metrics.improvement.pr_auc > 0 ? '+' : ''}{metrics.improvement.pr_auc.toFixed(3)} vs baseline
                    </span>
                  )}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  ROC-AUC
                </Typography>
                <Typography variant="h4" component="div">
                  {metrics.ann?.roc_auc?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {metrics.improvement?.roc_auc && (
                    <span style={{ color: metrics.improvement.roc_auc > 0 ? 'green' : 'red' }}>
                      {metrics.improvement.roc_auc > 0 ? '+' : ''}{metrics.improvement.roc_auc.toFixed(3)} vs baseline
                    </span>
                  )}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Precision
                </Typography>
                <Typography variant="h4" component="div">
                  {metrics.ann?.precision?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  at threshold {metrics.ann?.threshold?.toFixed(2) || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Recall
                </Typography>
                <Typography variant="h4" component="div">
                  {metrics.ann?.recall?.toFixed(3) || 'N/A'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  at threshold {metrics.ann?.threshold?.toFixed(2) || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Threshold Control */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Threshold Control
          </Typography>
          <Box sx={{ px: 2 }}>
            <Slider
              value={threshold}
              onChange={handleThresholdChange}
              min={0}
              max={1}
              step={0.01}
              marks={[
                { value: 0, label: '0' },
                { value: 0.5, label: '0.5' },
                { value: 1, label: '1' }
              ]}
              valueLabelDisplay="auto"
            />
            <Typography variant="body2" color="text.secondary" align="center">
              Current Threshold: {threshold.toFixed(2)}
            </Typography>
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={showConfusionMatrix}
                onChange={(e) => setShowConfusionMatrix(e.target.checked)}
              />
            }
            label="Show Confusion Matrix"
          />
        </CardContent>
      </Card>

      {/* Charts and Matrix */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Precision-Recall Curve
              </Typography>
              <PRChart />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ROC Curve
              </Typography>
              <ROCChart />
            </CardContent>
          </Card>
        </Grid>
        
        {showConfusionMatrix && confusionMatrix && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Confusion Matrix (Threshold: {threshold.toFixed(2)})
                </Typography>
                <ConfusionMatrix data={confusionMatrix} />
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  )
}
