'use client'

import { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  CircularProgress, 
  Alert,
  Chip
} from '@mui/material'
import { DataGrid, GridColDef, GridValueGetterParams } from '@mui/x-data-grid'
import useSWR from 'swr'

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

const fetcher = (url: string) => fetch(url).then(res => res.json())

interface SiteMetrics {
  site: string
  pr_auc: number
  f1_at_tau: number
  positives: number
  n: number
}

export default function SitesPage() {
  const { data: sites, error, isLoading } = useSWR<SiteMetrics[]>(`${API_BASE}/api/metrics/sites`, fetcher)

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <Alert severity="error">Failed to load site metrics</Alert>
      </Box>
    )
  }

  if (!sites || sites.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <Alert severity="warning">No site metrics available. Run training first.</Alert>
      </Box>
    )
  }

  // Define columns
  const columns: GridColDef[] = [
    {
      field: 'site',
      headerName: 'Site Name',
      width: 300,
      renderCell: (params) => (
        <Typography variant="body2" fontWeight="medium">
          {params.value}
        </Typography>
      )
    },
    {
      field: 'pr_auc',
      headerName: 'PR-AUC',
      width: 120,
      type: 'number',
      renderCell: (params) => (
        <Chip
          label={params.value.toFixed(3)}
          color={params.value >= 0.7 ? 'success' : params.value >= 0.5 ? 'warning' : 'error'}
          size="small"
        />
      )
    },
    {
      field: 'f1_at_tau',
      headerName: 'F1 Score',
      width: 120,
      type: 'number',
      renderCell: (params) => (
        <Chip
          label={params.value.toFixed(3)}
          color={params.value >= 0.7 ? 'success' : params.value >= 0.5 ? 'warning' : 'error'}
          size="small"
        />
      )
    },
    {
      field: 'positives',
      headerName: 'Fire Events',
      width: 120,
      type: 'number',
      renderCell: (params) => (
        <Typography variant="body2" color="error.main" fontWeight="medium">
          {params.value}
        </Typography>
      )
    },
    {
      field: 'n',
      headerName: 'Total Samples',
      width: 140,
      type: 'number',
      renderCell: (params) => (
        <Typography variant="body2" color="text.secondary">
          {params.value.toLocaleString()}
        </Typography>
      )
    },
    {
      field: 'positive_rate',
      headerName: 'Fire Rate',
      width: 120,
      type: 'number',
      valueGetter: (params: GridValueGetterParams) => {
        const positives = params.row.positives
        const total = params.row.n
        return total > 0 ? (positives / total) * 100 : 0
      },
      renderCell: (params) => (
        <Typography variant="body2" color="text.secondary">
          {params.value.toFixed(1)}%
        </Typography>
      )
    }
  ]

  // Add row IDs
  const rowsWithIds = sites.map((site, index) => ({
    id: index,
    ...site
  }))

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Site Performance Metrics
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Detailed performance metrics for each monitoring site in the Tri-County area.
      </Typography>
      
      {/* Summary Stats */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <Box>
              <Typography variant="h6" color="primary.main">
                {sites.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Sites
              </Typography>
            </Box>
            <Box>
              <Typography variant="h6" color="success.main">
                {sites.filter(s => s.pr_auc >= 0.7).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                High Performance (PR-AUC ≥ 0.7)
              </Typography>
            </Box>
            <Box>
              <Typography variant="h6" color="warning.main">
                {sites.filter(s => s.pr_auc >= 0.5 && s.pr_auc < 0.7).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Medium Performance (PR-AUC 0.5-0.7)
              </Typography>
            </Box>
            <Box>
              <Typography variant="h6" color="error.main">
                {sites.filter(s => s.pr_auc < 0.5).length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Low Performance (PR-AUC < 0.5)
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
      
      {/* Sites Table */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          <Box sx={{ height: 600, width: '100%' }}>
            <DataGrid
              rows={rowsWithIds}
              columns={columns}
              pageSize={10}
              rowsPerPageOptions={[10, 25, 50]}
              disableSelectionOnClick
              sx={{
                '& .MuiDataGrid-cell': {
                  borderBottom: '1px solid #e0e0e0'
                },
                '& .MuiDataGrid-columnHeaders': {
                  backgroundColor: '#f5f5f5',
                  borderBottom: '2px solid #e0e0e0'
                }
              }}
            />
          </Box>
        </CardContent>
      </Card>
      
      {/* Legend */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Legend
          </Typography>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label="0.7+" color="success" size="small" />
              <Typography variant="body2">Excellent</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label="0.5-0.7" color="warning" size="small" />
              <Typography variant="body2">Good</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label="<0.5" color="error" size="small" />
              <Typography variant="body2">Needs Improvement</Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}
