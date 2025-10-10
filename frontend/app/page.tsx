"use client";

import React from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from "@mui/material";
import {
  LocalFireDepartment,
  Dataset,
  Psychology,
  Settings,
  History,
  Info,
  Warning,
  CheckCircle,
} from "@mui/icons-material";
import Link from "next/link";

export default function HomePage() {
  return (
    <Box>
      {/* Hero Section */}
      <Box sx={{ mb: 6, textAlign: "center" }}>
        <LocalFireDepartment
          sx={{
            fontSize: 80,
            color: "#ff9800",
            mb: 2,
          }}
        />
        <Typography variant="h3" component="h1" gutterBottom>
          Fire Forecasting Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
          ML-powered wildfire prediction system using temporal weather data
        </Typography>
        <Alert severity="info" sx={{ maxWidth: 600, mx: "auto" }}>
          <Typography variant="body2">
            <strong>Note:</strong> This is a frontend-only application. Backend
            API has been removed. Navigate to Settings and ML History to explore
            the interface.
          </Typography>
        </Alert>
      </Box>

      {/* Quick Actions */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Settings sx={{ fontSize: 48, color: "primary.main", mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                Settings
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Configure your dashboard preferences and data sources
              </Typography>
              <Button
                variant="contained"
                component={Link}
                href="/settings"
                fullWidth
              >
                Open Settings
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ textAlign: "center" }}>
              <History sx={{ fontSize: 48, color: "secondary.main", mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                ML History
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                View machine learning model training history and results
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                component={Link}
                href="/ml-history"
                fullWidth
              >
                View History
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Info sx={{ fontSize: 48, color: "info.main", mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                About
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Learn more about the fire forecasting system
              </Typography>
              <Button variant="outlined" color="info" fullWidth disabled>
                Coming Soon
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Dataset Information */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Dataset sx={{ mr: 1, verticalAlign: "middle" }} />
                Available Datasets
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Hourly Weather Dataset"
                    secondary="40,711 records - Every hour weather data (2020-2023)"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Bihourly Weather Dataset"
                    secondary="23,954 records - Every 2 hours weather data (2020-2023)"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Trihourly Weather Dataset"
                    secondary="18,110 records - Every 3 hours weather data (2020-2023)"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Psychology sx={{ mr: 1, verticalAlign: "middle" }} />
                Features
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Temporal Weather Data"
                    secondary="Temperature, humidity, precipitation, wind speed, pressure"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Soil Conditions"
                    secondary="Soil temperature and moisture at different depths"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Fire Labels"
                    secondary="Binary fire indicators (N/Y) and severity scores"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Multi-Resolution"
                    secondary="Data available at 1-hour, 2-hour, and 3-hour intervals"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Status Information */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <Warning
              sx={{ mr: 1, verticalAlign: "middle", color: "warning.main" }}
            />
            System Status
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Paper variant="outlined" sx={{ p: 2, bgcolor: "grey.50" }}>
            <Typography variant="body2" color="text.secondary">
              <strong>Current Status:</strong> Frontend-only mode
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Backend API:</strong> Removed (as requested)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Data Sources:</strong> 3 temporal weather datasets
              available
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>ML Models:</strong> Simulated data in ML History page
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>Last Updated:</strong> {new Date().toLocaleDateString()}
            </Typography>
          </Paper>
        </CardContent>
      </Card>
    </Box>
  );
}
