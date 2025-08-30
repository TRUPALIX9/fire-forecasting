"use client";

import { useEffect, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Divider,
} from "@mui/material";
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  ExpandMore,
  Refresh,
  Download,
  PlayArrow,
  Settings,
  DataUsage,
  Map,
  Science,
  Storage,
} from "@mui/icons-material";

interface ChecklistItem {
  id: string;
  category: string;
  name: string;
  description: string;
  status: "success" | "error" | "warning" | "info";
  message: string;
  action?: string;
  actionUrl?: string;
  priority: "critical" | "high" | "medium" | "low";
}

interface CategoryStatus {
  name: string;
  icon: React.ReactNode;
  total: number;
  success: number;
  error: number;
  warning: number;
  info: number;
}

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function ChecklistPage() {
  const [checklistItems, setChecklistItems] = useState<ChecklistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [overallStatus, setOverallStatus] = useState<
    "success" | "error" | "warning" | "info"
  >("info");

  const runSystemCheck = async () => {
    setLoading(true);
    const items: ChecklistItem[] = [];

    try {
      // Check backend connectivity
      try {
        const statusResponse = await fetch(`${API_BASE}/api/status`);
        if (statusResponse.ok) {
          items.push({
            id: "backend-connectivity",
            category: "Backend",
            name: "Backend Service",
            description: "FastAPI backend is running and accessible",
            status: "success",
            message: "Backend is running on port 8000",
            priority: "critical",
          });
        } else {
          items.push({
            id: "backend-connectivity",
            category: "Backend",
            name: "Backend Service",
            description: "FastAPI backend is running and accessible",
            status: "error",
            message: `Backend responded with status ${statusResponse.status}`,
            action: "Start backend",
            actionUrl: "make run-backend",
            priority: "critical",
          });
        }
      } catch (error) {
        items.push({
          id: "backend-connectivity",
          category: "Backend",
          name: "Backend Service",
          description: "FastAPI backend is running and accessible",
          status: "error",
          message: "Cannot connect to backend service",
          action: "Start backend",
          actionUrl: "make run-backend",
          priority: "critical",
        });
      }

      // Check FIRMS data
      try {
        const firmsResponse = await fetch(`${API_BASE}/api/data/firms`);
        if (firmsResponse.ok) {
          const firmsData = await firmsResponse.json();
          if (firmsData.count > 0) {
            items.push({
              id: "firms-data",
              category: "Data Sources",
              name: "FIRMS Fire Detection Data",
              description: "NASA FIRMS fire detection data is available",
              status: "success",
              message: `${firmsData.count.toLocaleString()} fire detections available`,
              priority: "critical",
            });
          } else {
            items.push({
              id: "firms-data",
              category: "Data Sources",
              name: "FIRMS Fire Detection Data",
              description: "NASA FIRMS fire detection data is available",
              status: "warning",
              message: "FIRMS data exists but contains no records",
              action: "Fetch FIRMS data",
              actionUrl: "make fetch-firms",
              priority: "critical",
            });
          }
        } else {
          items.push({
            id: "firms-data",
            category: "Data Sources",
            name: "FIRMS Fire Detection Data",
            description: "NASA FIRMS fire detection data is available",
            status: "error",
            message: "FIRMS data endpoint not found",
            action: "Fetch FIRMS data",
            actionUrl: "make fetch-firms",
            priority: "critical",
          });
        }
      } catch (error) {
        items.push({
          id: "firms-data",
          category: "Data Sources",
          name: "FIRMS Fire Detection Data",
          description: "NASA FIRMS fire detection data is available",
          status: "error",
          message: "Cannot access FIRMS data endpoint",
          action: "Fetch FIRMS data",
          actionUrl: "make fetch-firms",
          priority: "critical",
        });
      }

      // Check FRAP data
      try {
        const frapResponse = await fetch(`${API_BASE}/api/geo/frap`);
        if (frapResponse.ok) {
          const frapData = await frapResponse.json();
          if (frapData.features && frapData.features.length > 0) {
            items.push({
              id: "frap-data",
              category: "Data Sources",
              name: "FRAP Fire Perimeters",
              description: "CAL FIRE FRAP historical fire perimeter data",
              status: "success",
              message: `${frapData.features.length} fire perimeters available`,
              priority: "high",
            });
          } else {
            items.push({
              id: "frap-data",
              category: "Data Sources",
              name: "FRAP Fire Perimeters",
              description: "CAL FIRE FRAP historical fire perimeter data",
              status: "warning",
              message: "FRAP data exists but contains no features",
              action: "Fetch FRAP data",
              actionUrl: "make fetch-frap",
              priority: "high",
            });
          }
        } else {
          items.push({
            id: "frap-data",
            category: "Data Sources",
            name: "FRAP Fire Perimeters",
            description: "CAL FIRE FRAP historical fire perimeter data",
            status: "error",
            message: "FRAP data not found",
            action: "Fetch FRAP data",
            actionUrl: "make fetch-frap",
            priority: "high",
          });
        }
      } catch (error) {
        items.push({
          id: "frap-data",
          category: "Data Sources",
          name: "FRAP Fire Perimeters",
          description: "CAL FIRE FRAP historical fire perimeter data",
          status: "error",
          message: "Cannot access FRAP data",
          action: "Fetch FRAP data",
          actionUrl: "make fetch-frap",
          priority: "high",
        });
      }

      // Check RAWS data
      try {
        const rawsResponse = await fetch(`${API_BASE}/api/data/raws`);
        if (rawsResponse.ok) {
          const rawsData = await rawsResponse.json();
          if (rawsData.count > 0) {
            items.push({
              id: "raws-data",
              category: "Data Sources",
              name: "RAWS Weather Data",
              description: "Remote Automated Weather Station data",
              status: "success",
              message: `${rawsData.count.toLocaleString()} weather records available`,
              priority: "high",
            });
          } else {
            items.push({
              id: "raws-data",
              category: "Data Sources",
              name: "RAWS Weather Data",
              description: "Remote Automated Weather Station data",
              status: "warning",
              message: "RAWS data exists but contains no records",
              action: "Fetch RAWS data",
              actionUrl: "make fetch-raws",
              priority: "high",
            });
          }
        } else {
          items.push({
            id: "raws-data",
            category: "Data Sources",
            name: "RAWS Weather Data",
            description: "Remote Automated Weather Station data",
            status: "error",
            message: "RAWS data endpoint not found",
            action: "Fetch RAWS data",
            actionUrl: "make fetch-raws",
            priority: "high",
          });
        }
      } catch (error) {
        items.push({
          id: "raws-data",
          category: "Data Sources",
          name: "RAWS Weather Data",
          description: "Remote Automated Weather Station data",
          status: "error",
          message: "Cannot access RAWS data endpoint",
          action: "Fetch RAWS data",
          actionUrl: "make fetch-raws",
          priority: "high",
        });
      }

      // Check model artifacts
      try {
        const modelResponse = await fetch(`${API_BASE}/api/models/status`);
        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          if (modelData.trained) {
            items.push({
              id: "model-artifacts",
              category: "ML Models",
              name: "Trained Model",
              description: "Machine learning model artifacts are available",
              status: "success",
              message: `Model trained on ${modelData.training_date}`,
              priority: "critical",
            });
          } else {
            items.push({
              id: "model-artifacts",
              category: "ML Models",
              name: "Trained Model",
              description: "Machine learning model artifacts are available",
              status: "warning",
              message:
                "Model artifacts exist but model may not be fully trained",
              action: "Train model",
              actionUrl: "make run-train",
              priority: "critical",
            });
          }
        } else {
          items.push({
            id: "model-artifacts",
            category: "ML Models",
            name: "Trained Model",
            description: "Machine learning model artifacts are available",
            status: "error",
            message: "Model artifacts not found",
            action: "Train model",
            actionUrl: "make run-train",
            priority: "critical",
          });
        }
      } catch (error) {
        items.push({
          id: "model-artifacts",
          category: "ML Models",
          name: "Trained Model",
          description: "Machine learning model artifacts are available",
          status: "error",
          message: "Cannot access model status",
          action: "Train model",
          actionUrl: "make run-train",
          priority: "critical",
        });
      }

      // Check metrics and figures
      try {
        const metricsResponse = await fetch(`${API_BASE}/api/metrics/global`);
        if (metricsResponse.ok) {
          items.push({
            id: "metrics-data",
            category: "ML Models",
            name: "Model Metrics",
            description: "Model performance metrics and evaluation results",
            status: "success",
            message: "Model metrics are available",
            priority: "medium",
          });
        } else {
          items.push({
            id: "metrics-data",
            category: "ML Models",
            name: "Model Metrics",
            description: "Model performance metrics and evaluation results",
            status: "warning",
            message: "Model metrics not found",
            action: "Train model",
            actionUrl: "make run-train",
            priority: "medium",
          });
        }
      } catch (error) {
        items.push({
          id: "metrics-data",
          category: "ML Models",
          name: "Model Metrics",
          description: "Model performance metrics and evaluation results",
          status: "warning",
          message: "Cannot access model metrics",
          action: "Train model",
          actionUrl: "make run-train",
          priority: "medium",
        });
      }

      // Check geographic data
      try {
        const sitesResponse = await fetch(`${API_BASE}/api/geo/sites`);
        if (sitesResponse.ok) {
          const sitesData = await sitesResponse.json();
          if (sitesData.features && sitesData.features.length > 0) {
            items.push({
              id: "sites-data",
              category: "Geographic Data",
              name: "Monitoring Sites",
              description: "WUI monitoring site locations and metadata",
              status: "success",
              message: `${sitesData.features.length} monitoring sites available`,
              priority: "medium",
            });
          } else {
            items.push({
              id: "sites-data",
              category: "Geographic Data",
              name: "Monitoring Sites",
              description: "WUI monitoring site locations and metadata",
              status: "warning",
              message: "Sites data exists but contains no features",
              priority: "medium",
            });
          }
        } else {
          items.push({
            id: "sites-data",
            category: "Geographic Data",
            name: "Monitoring Sites",
            description: "WUI monitoring site locations and metadata",
            status: "warning",
            message: "Sites data not found",
            priority: "medium",
          });
        }
      } catch (error) {
        items.push({
          id: "sites-data",
          category: "Geographic Data",
          name: "Monitoring Sites",
          description: "WUI monitoring site locations and metadata",
          status: "warning",
          message: "Cannot access sites data",
          priority: "medium",
        });
      }

      // Check configuration
      items.push({
        id: "config-file",
        category: "Configuration",
        name: "Configuration File",
        description: "config.yaml with pipeline parameters",
        status: "info",
        message: "Configuration file should be present",
        priority: "low",
      });

      // Check environment setup
      items.push({
        id: "env-setup",
        category: "Configuration",
        name: "Environment Variables",
        description: "Required environment variables are set",
        status: "info",
        message: "Check .env file for required variables",
        priority: "low",
      });

      // Check dependencies
      items.push({
        id: "dependencies",
        category: "Configuration",
        name: "Python Dependencies",
        description: "All required Python packages are installed",
        status: "info",
        message: "Run 'pip install -r requirements.txt'",
        priority: "low",
      });
    } catch (error) {
      console.error("Error running system check:", error);
    }

    setChecklistItems(items);
    setLastChecked(new Date());
    setLoading(false);

    // Determine overall status
    const hasErrors = items.some((item) => item.status === "error");
    const hasWarnings = items.some((item) => item.status === "warning");

    if (hasErrors) {
      setOverallStatus("error");
    } else if (hasWarnings) {
      setOverallStatus("warning");
    } else {
      setOverallStatus("success");
    }
  };

  useEffect(() => {
    runSystemCheck();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "success":
        return <CheckCircle color="success" />;
      case "error":
        return <Error color="error" />;
      case "warning":
        return <Warning color="warning" />;
      case "info":
        return <Info color="info" />;
      default:
        return <Info color="info" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "error";
      case "high":
        return "warning";
      case "medium":
        return "info";
      case "low":
        return "default";
      default:
        return "default";
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "Backend":
        return <Settings />;
      case "Data Sources":
        return <DataUsage />;
      case "ML Models":
        return <Science />;
      case "Geographic Data":
        return <Map />;
      case "Configuration":
        return <Storage />;
      default:
        return <Info />;
    }
  };

  const getCategoryStatus = (category: string): CategoryStatus => {
    const categoryItems = checklistItems.filter(
      (item) => item.category === category
    );
    return {
      name: category,
      icon: getCategoryIcon(category),
      total: categoryItems.length,
      success: categoryItems.filter((item) => item.status === "success").length,
      error: categoryItems.filter((item) => item.status === "error").length,
      warning: categoryItems.filter((item) => item.status === "warning").length,
      info: categoryItems.filter((item) => item.status === "info").length,
    };
  };

  const categories = Array.from(
    new Set(checklistItems.map((item) => item.category))
  );

  const getActionSteps = () => {
    const steps: string[] = [];

    if (
      checklistItems.some(
        (item) => item.status === "error" && item.priority === "critical"
      )
    ) {
      steps.push("1. Start the backend service: make run-backend");
    }

    if (
      checklistItems.some(
        (item) => item.id === "firms-data" && item.status !== "success"
      )
    ) {
      steps.push("2. Fetch FIRMS data: make fetch-firms");
    }

    if (
      checklistItems.some(
        (item) => item.id === "frap-data" && item.status !== "success"
      )
    ) {
      steps.push("3. Fetch FRAP data: make fetch-frap");
    }

    if (
      checklistItems.some(
        (item) => item.id === "raws-data" && item.status !== "success"
      )
    ) {
      steps.push("4. Fetch RAWS data: make fetch-raws");
    }

    if (
      checklistItems.some(
        (item) => item.id === "model-artifacts" && item.status !== "success"
      )
    ) {
      steps.push("5. Train the model: make run-train");
    }

    if (checklistItems.some((item) => item.status === "success")) {
      steps.push("6. Start the frontend: cd frontend && npm run dev");
      steps.push("7. Open http://localhost:3000 to view the dashboard");
    }

    return steps;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        🔍 System Health Check
      </Typography>

      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Comprehensive checklist of all required components, data sources, and
        system health
      </Typography>

      {/* Overall Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
            {getStatusIcon(overallStatus)}
            <Typography variant="h6">
              Overall System Status: {overallStatus.toUpperCase()}
            </Typography>
          </Box>

          {overallStatus === "success" && (
            <Alert severity="success">
              All critical systems are operational! Your fire forecasting system
              is ready to use.
            </Alert>
          )}

          {overallStatus === "warning" && (
            <Alert severity="warning">
              System is mostly operational but has some warnings. Review the
              checklist below.
            </Alert>
          )}

          {overallStatus === "error" && (
            <Alert severity="error">
              Critical systems are not operational. Follow the action steps
              below to resolve issues.
            </Alert>
          )}

          <Box sx={{ mt: 2, display: "flex", alignItems: "center", gap: 2 }}>
            <Button
              variant="contained"
              startIcon={<Refresh />}
              onClick={runSystemCheck}
              disabled={loading}
            >
              {loading ? "Checking..." : "Refresh Status"}
            </Button>

            {lastChecked && (
              <Typography variant="body2" color="text.secondary">
                Last checked: {lastChecked.toLocaleString()}
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Category Summary */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Components Overview
          </Typography>

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            {categories.map((category) => {
              const status = getCategoryStatus(category);
              const successRate =
                status.total > 0 ? (status.success / status.total) * 100 : 0;

              return (
                <Card key={category} variant="outlined" sx={{ minWidth: 200 }}>
                  <CardContent>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                        mb: 1,
                      }}
                    >
                      {status.icon}
                      <Typography variant="subtitle2">{category}</Typography>
                    </Box>

                    <LinearProgress
                      variant="determinate"
                      value={successRate}
                      color={
                        successRate === 100
                          ? "success"
                          : successRate > 50
                          ? "warning"
                          : "error"
                      }
                      sx={{ mb: 1 }}
                    />

                    <Typography variant="body2" color="text.secondary">
                      {status.success}/{status.total} components ready
                    </Typography>

                    <Box sx={{ display: "flex", gap: 0.5, mt: 1 }}>
                      {status.success > 0 && (
                        <Chip
                          size="small"
                          label={status.success}
                          color="success"
                        />
                      )}
                      {status.warning > 0 && (
                        <Chip
                          size="small"
                          label={status.warning}
                          color="warning"
                        />
                      )}
                      {status.error > 0 && (
                        <Chip size="small" label={status.error} color="error" />
                      )}
                      {status.info > 0 && (
                        <Chip size="small" label={status.info} color="info" />
                      )}
                    </Box>
                  </CardContent>
                </Card>
              );
            })}
          </Box>
        </CardContent>
      </Card>

      {/* Detailed Checklist */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Detailed Component Checklist
          </Typography>

          {categories.map((category) => (
            <Accordion key={category} defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {getCategoryIcon(category)}
                  <Typography variant="subtitle1">{category}</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  {checklistItems
                    .filter((item) => item.category === category)
                    .map((item) => (
                      <ListItem key={item.id} sx={{ pl: 0 }}>
                        <ListItemIcon sx={{ minWidth: 40 }}>
                          {getStatusIcon(item.status)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 1,
                              }}
                            >
                              {item.name}
                              <Chip
                                label={item.priority}
                                size="small"
                                color={getPriorityColor(item.priority) as any}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography
                                variant="body2"
                                color="text.secondary"
                              >
                                {item.description}
                              </Typography>
                              <Typography variant="body2" sx={{ mt: 0.5 }}>
                                {item.message}
                              </Typography>
                              {item.action && (
                                <Typography
                                  variant="body2"
                                  color="primary"
                                  sx={{ mt: 0.5 }}
                                >
                                  Action: {item.action} - {item.actionUrl}
                                </Typography>
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                    ))}
                </List>
              </AccordionDetails>
            </Accordion>
          ))}
        </CardContent>
      </Card>

      {/* Action Steps */}
      {getActionSteps().length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🚀 Next Steps to Complete Setup
            </Typography>

            <Alert severity="info" sx={{ mb: 2 }}>
              Follow these steps in order to get your fire forecasting system
              fully operational.
            </Alert>

            <List>
              {getActionSteps().map((step, index) => (
                <ListItem key={index} sx={{ pl: 0 }}>
                  <ListItemIcon>
                    <PlayArrow color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography
                        variant="body1"
                        sx={{ fontFamily: "monospace" }}
                      >
                        {step}
                      </Typography>
                    }
                  />
                </ListItem>
              ))}
            </List>

            <Divider sx={{ my: 2 }} />

            <Typography variant="body2" color="text.secondary">
              <strong>Note:</strong> Run these commands from your project root
              directory. Make sure your virtual environment is activated and all
              dependencies are installed.
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
