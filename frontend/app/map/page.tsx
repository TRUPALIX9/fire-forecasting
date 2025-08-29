'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, Typography, Box, FormControlLabel, Switch, CircularProgress, Alert } from '@mui/material'
import dynamic from 'next/dynamic'

// Dynamically import map components to avoid SSR issues
const MapLeaflet = dynamic(() => import('../components/MapLeaflet'), { ssr: false })

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function MapPage() {
  const [showSites, setShowSites] = useState(true)
  const [showFRAP, setShowFRAP] = useState(true)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Simulate loading time for map initialization
    const timer = setTimeout(() => {
      setLoading(false)
    }, 1000)
    
    return () => clearTimeout(timer)
  }, [])

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <Alert severity="error">{error}</Alert>
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Fire Risk Map
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Interactive map showing monitoring sites and historical fire perimeters in the Tri-County area.
      </Typography>
      
      {/* Map Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <FormControlLabel
              control={
                <Switch
                  checked={showSites}
                  onChange={(e) => setShowSites(e.target.checked)}
                />
              }
              label="Show Monitoring Sites"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showFRAP}
                  onChange={(e) => setShowFRAP(e.target.checked)}
                />
              }
              label="Show Historical Fires (FRAP)"
            />
          </Box>
        </CardContent>
      </Card>
      
      {/* Map */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          <Box sx={{ height: 600, width: '100%' }}>
            <MapLeaflet
              showSites={showSites}
              showFRAP={showFRAP}
            />
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}
