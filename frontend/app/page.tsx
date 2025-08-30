"use client";

import { useState, useEffect } from "react";
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import {
  LocalFireDepartment,
  Map,
  LocationOn,
  Checklist,
  Info,
  TrendingUp,
  Security,
  Speed,
  Science,
} from "@mui/icons-material";
import Link from "next/link";
import useSWR from "swr";

// API client
const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function HomePage() {
  const { data: status, error: statusError } = useSWR(
    `${API_BASE}/api/status`,
    fetcher
  );
  const { data: modelStatus, error: modelError } = useSWR(
    `${API_BASE}/api/models/status`,
    fetcher
  );

  if (statusError)
    return <Alert severity="error">Failed to load system status</Alert>;
  if (!status) return <CircularProgress />;

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        elevation={0}
        sx={{
          background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
          color: "white",
          p: 6,
          mb: 4,
          borderRadius: 3,
          textAlign: "center",
        }}
      >
        <LocalFireDepartment sx={{ fontSize: 80, mb: 2, color: "#ff9800" }} />
        <Typography variant="h2" gutterBottom sx={{ fontWeight: 700 }}>
          Welcome to Fire Forecasting
        </Typography>
        <Typography variant="h5" sx={{ mb: 3, opacity: 0.9 }}>
          ML-Powered Wildfire Prediction System for the Tri-County Area
        </Typography>
        <Typography
          variant="body1"
          sx={{ mb: 4, opacity: 0.8, maxWidth: 800, mx: "auto" }}
        >
          Advanced machine learning system that combines satellite data, weather
          information, and historical fire patterns to predict wildfire risks in
          Santa Barbara, Ventura, and Los Angeles counties.
        </Typography>

        <Box
          sx={{
            display: "flex",
            gap: 2,
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <Link href="/map" style={{ textDecoration: "none" }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<Map />}
              sx={{
                bgcolor: "#ff9800",
                color: "white",
                "&:hover": { bgcolor: "#f57c00" },
              }}
            >
              Explore Map
            </Button>
          </Link>
          <Link href="/checklist" style={{ textDecoration: "none" }}>
            <Button
              variant="outlined"
              size="large"
              startIcon={<Checklist />}
              sx={{
                borderColor: "white",
                color: "white",
                "&:hover": {
                  borderColor: "white",
                  bgcolor: "rgba(255,255,255,0.1)",
                },
              }}
            >
              System Status
            </Button>
          </Link>
        </Box>
      </Paper>

      {/* System Status Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%", borderLeft: "4px solid #4caf50" }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Security color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">System Status</Typography>
              </Box>
              <Typography variant="h4" color="primary" gutterBottom>
                {status.api_status === "running"
                  ? "🟢 Operational"
                  : "🔴 Offline"}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Backend API is running and responding to requests
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%", borderLeft: "4px solid #ff9800" }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Science color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">ML Model</Typography>
              </Box>
              <Typography variant="h4" color="primary" gutterBottom>
                {modelStatus?.trained ? "🟢 Trained" : "🟡 Not Ready"}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {modelStatus?.trained
                  ? "Model is trained and ready for predictions"
                  : "Model needs training or artifacts are missing"}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%", borderLeft: "4px solid #2196f3" }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <TrendingUp color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Data Sources</Typography>
              </Box>
              <Typography variant="h4" color="primary" gutterBottom>
                🟢 Active
              </Typography>
              <Typography variant="body2" color="text.secondary">
                FIRMS, FRAP, and RAWS data sources are operational
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Key Features */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Key Features
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Map color="primary" sx={{ mr: 1, fontSize: 32 }} />
                <Typography variant="h6">Interactive Mapping</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Explore fire risk data with our advanced mapping interface
                featuring:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <LocationOn color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Real-time monitoring sites" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <LocalFireDepartment color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Historical fire perimeters (FRAP)" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <TrendingUp color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Risk prediction overlays" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <Science color="primary" sx={{ mr: 1, fontSize: 32 }} />
                <Typography variant="h6">Machine Learning</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Advanced AI-powered wildfire prediction featuring:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <TrendingUp color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Neural network models" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Speed color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Real-time predictions" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <Security color="action" />
                  </ListItemIcon>
                  <ListItemText primary="Risk assessment algorithms" />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Link href="/map" style={{ textDecoration: "none" }}>
            <Card
              sx={{
                height: "100%",
                cursor: "pointer",
                transition: "transform 0.2s ease-in-out",
                "&:hover": { transform: "translateY(-4px)" },
              }}
            >
              <CardContent sx={{ textAlign: "center" }}>
                <Map color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  View Map
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Interactive fire risk visualization
                </Typography>
              </CardContent>
            </Card>
          </Link>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Link href="/sites" style={{ textDecoration: "none" }}>
            <Card
              sx={{
                height: "100%",
                cursor: "pointer",
                transition: "transform 0.2s ease-in-out",
                "&:hover": { transform: "translateY(-4px)" },
              }}
            >
              <CardContent sx={{ textAlign: "center" }}>
                <LocationOn color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Monitoring Sites
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Weather station data and locations
                </Typography>
              </CardContent>
            </Card>
          </Link>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Link href="/checklist" style={{ textDecoration: "none" }}>
            <Card
              sx={{
                height: "100%",
                cursor: "pointer",
                transition: "transform 0.2s ease-in-out",
                "&:hover": { transform: "translateY(-4px)" },
              }}
            >
              <CardContent sx={{ textAlign: "center" }}>
                <Checklist color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  System Health
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Check system status and data integrity
                </Typography>
              </CardContent>
            </Card>
          </Link>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Link href="/about" style={{ textDecoration: "none" }}>
            <Card
              sx={{
                height: "100%",
                cursor: "pointer",
                transition: "transform 0.2s ease-in-out",
                "&:hover": { transform: "translateY(-4px)" },
              }}
            >
              <CardContent sx={{ textAlign: "center" }}>
                <Info color="primary" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Learn More
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Project details and documentation
                </Typography>
              </CardContent>
            </Card>
          </Link>
        </Grid>
      </Grid>

      {/* Technology Stack */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Technology Stack
      </Typography>
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Backend
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                <Chip label="FastAPI" color="primary" />
                <Chip label="Python" color="primary" />
                <Chip label="TensorFlow" color="primary" />
                <Chip label="PostgreSQL" color="primary" />
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Frontend
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                <Chip label="Next.js" color="secondary" />
                <Chip label="React" color="secondary" />
                <Chip label="Material-UI" color="secondary" />
                <Chip label="Leaflet" color="secondary" />
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                ML & Data
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                <Chip label="Scikit-learn" color="success" />
                <Chip label="Pandas" color="success" />
                <Chip label="GeoPandas" color="success" />
                <Chip label="NASA FIRMS" color="success" />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}
