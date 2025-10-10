"use client";

import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Alert,
  Divider,
  LinearProgress,
  IconButton,
  Tooltip,
} from "@mui/material";
import {
  History as HistoryIcon,
  Refresh,
  Visibility,
  Download,
  Info,
  Warning,
  CheckCircle,
  Error,
} from "@mui/icons-material";

interface ModelRun {
  id: string;
  name: string;
  type: string;
  dataset: string;
  status: "completed" | "failed" | "running" | "pending";
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  timestamp: string;
  duration: string;
}

export default function MLHistoryPage() {
  const [models, setModels] = useState<ModelRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading ML model history
    setTimeout(() => {
      setModels([
        {
          id: "1",
          name: "Random Forest - Hourly",
          type: "Random Forest",
          dataset: "Hourly_Weather_Dataset.csv",
          status: "completed",
          accuracy: 0.854,
          precision: 0.823,
          recall: 0.789,
          f1Score: 0.806,
          auc: 0.891,
          timestamp: "2024-01-15 14:30:00",
          duration: "2m 45s",
        },
        {
          id: "2",
          name: "XGBoost - Bihourly",
          type: "XGBoost",
          dataset: "Bihourly_Weather_Dataset.csv",
          status: "completed",
          accuracy: 0.867,
          precision: 0.841,
          recall: 0.812,
          f1Score: 0.826,
          auc: 0.903,
          timestamp: "2024-01-15 12:15:00",
          duration: "3m 12s",
        },
        {
          id: "3",
          name: "Neural Network - Trihourly",
          type: "Neural Network",
          dataset: "Trihourly_Weather_Dataset.csv",
          status: "completed",
          accuracy: 0.839,
          precision: 0.815,
          recall: 0.798,
          f1Score: 0.806,
          auc: 0.877,
          timestamp: "2024-01-15 10:45:00",
          duration: "5m 23s",
        },
        {
          id: "4",
          name: "SVM - Hourly",
          type: "Support Vector Machine",
          dataset: "Hourly_Weather_Dataset.csv",
          status: "failed",
          accuracy: 0.0,
          precision: 0.0,
          recall: 0.0,
          f1Score: 0.0,
          auc: 0.0,
          timestamp: "2024-01-15 09:20:00",
          duration: "1m 34s",
        },
        {
          id: "5",
          name: "Logistic Regression - Bihourly",
          type: "Logistic Regression",
          dataset: "Bihourly_Weather_Dataset.csv",
          status: "completed",
          accuracy: 0.782,
          precision: 0.756,
          recall: 0.734,
          f1Score: 0.745,
          auc: 0.834,
          timestamp: "2024-01-15 08:10:00",
          duration: "1m 56s",
        },
      ]);
      setLoading(false);
    }, 1500);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "success";
      case "failed":
        return "error";
      case "running":
        return "warning";
      case "pending":
        return "default";
      default:
        return "default";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle />;
      case "failed":
        return <Error />;
      case "running":
        return <LinearProgress sx={{ width: 20, height: 20 }} />;
      case "pending":
        return <Warning />;
      default:
        return <Info />;
    }
  };

  const formatMetric = (value: number) => {
    return value > 0 ? (value * 100).toFixed(1) + "%" : "N/A";
  };

  const handleViewDetails = (modelId: string) => {
    console.log("View details for model:", modelId);
    // In a real app, this would navigate to model details
  };

  const handleDownloadModel = (modelId: string) => {
    console.log("Download model:", modelId);
    // In a real app, this would download the model file
  };

  const handleRefresh = () => {
    setLoading(true);
    // Simulate refresh
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              <HistoryIcon sx={{ mr: 2, verticalAlign: "middle" }} />
              ML Model History
            </Typography>
            <Typography variant="subtitle1" color="text.secondary">
              Track and manage your machine learning model training runs
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Alert */}
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>Note:</strong> This is a frontend-only application. Model
          history is simulated data. In a production environment, this would
          connect to your ML training backend.
        </Typography>
      </Alert>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Models
              </Typography>
              <Typography variant="h4">{models.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h4" color="success.main">
                {models.filter((m) => m.status === "completed").length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Failed
              </Typography>
              <Typography variant="h4" color="error.main">
                {models.filter((m) => m.status === "failed").length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Best Accuracy
              </Typography>
              <Typography variant="h4" color="primary.main">
                {formatMetric(
                  Math.max(
                    ...models
                      .filter((m) => m.status === "completed")
                      .map((m) => m.accuracy)
                  )
                )}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Models Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Model Training Runs
          </Typography>
          <Divider sx={{ mb: 2 }} />

          {loading ? (
            <Box sx={{ p: 3 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1, textAlign: "center" }}>
                Loading model history...
              </Typography>
            </Box>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Model Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Dataset</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Accuracy</TableCell>
                    <TableCell>Precision</TableCell>
                    <TableCell>Recall</TableCell>
                    <TableCell>F1 Score</TableCell>
                    <TableCell>AUC</TableCell>
                    <TableCell>Duration</TableCell>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {models.map((model) => (
                    <TableRow key={model.id} hover>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {model.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={model.type}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {model.dataset}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={model.status}
                          color={getStatusColor(model.status) as any}
                          size="small"
                          icon={getStatusIcon(model.status)}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatMetric(model.accuracy)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatMetric(model.precision)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatMetric(model.recall)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatMetric(model.f1Score)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatMetric(model.auc)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {model.duration}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {model.timestamp}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleViewDetails(model.id)}
                              disabled={model.status !== "completed"}
                            >
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Download Model">
                            <IconButton
                              size="small"
                              onClick={() => handleDownloadModel(model.id)}
                              disabled={model.status !== "completed"}
                            >
                              <Download />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
