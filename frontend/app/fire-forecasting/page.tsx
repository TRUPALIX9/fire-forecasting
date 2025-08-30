"use client";

import React from "react";
import {
  Box,
  Container,
  Typography,
  Paper,
  Chip,
  Stack,
  Alert,
  AlertTitle,
} from "@mui/material";
import { Construction as ConstructionIcon } from "@mui/icons-material";

export default function FireForecastingPage() {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: "center", mb: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Fire Forecasting
        </Typography>
        <Typography variant="h5" color="text.secondary" gutterBottom>
          Advanced AI-powered wildfire prediction and analysis
        </Typography>
      </Box>

      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
        <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
          <ConstructionIcon sx={{ fontSize: 40, color: "warning.main" }} />
          <Typography variant="h4" component="h2">
            Under Construction
          </Typography>
        </Stack>

        <Alert severity="info" sx={{ mb: 3 }}>
          <AlertTitle>Development in Progress</AlertTitle>
          We're working hard to bring you the most advanced fire forecasting
          capabilities. This page will feature real-time wildfire predictions,
          historical analysis, and machine learning insights.
        </Alert>

        <Typography variant="body1" paragraph>
          Our fire forecasting system is being developed using cutting-edge
          machine learning algorithms and comprehensive environmental data
          analysis. When complete, it will provide:
        </Typography>

        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: 2,
            mb: 3,
          }}
        >
          <Paper elevation={1} sx={{ p: 2, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              Real-time Predictions
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Live wildfire risk assessments and probability forecasts
            </Typography>
          </Paper>

          <Paper elevation={1} sx={{ p: 2, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              Historical Analysis
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Comprehensive data on past fire events and patterns
            </Typography>
          </Paper>

          <Paper elevation={1} sx={{ p: 2, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              AI Insights
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Machine learning-powered risk factor analysis
            </Typography>
          </Paper>
        </Box>

        <Box sx={{ textAlign: "center" }}>
          <Typography variant="h6" gutterBottom>
            Expected Features
          </Typography>
          <Stack
            direction="row"
            spacing={1}
            justifyContent="center"
            flexWrap="wrap"
          >
            <Chip label="Weather Integration" color="primary" />
            <Chip label="Satellite Data" color="primary" />
            <Chip label="Risk Mapping" color="primary" />
            <Chip label="Alert System" color="primary" />
            <Chip label="Data Visualization" color="primary" />
            <Chip label="API Access" color="primary" />
          </Stack>
        </Box>
      </Paper>

      <Paper elevation={2} sx={{ p: 3, textAlign: "center" }}>
        <Typography variant="body2" color="text.secondary">
          Check back soon for updates and early access to our fire forecasting
          platform!
        </Typography>
      </Paper>
    </Container>
  );
}
