"use client";

import {
  Card,
  CardContent,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
} from "@mui/material";
import {
  ExpandMore,
  DataUsage,
  Science,
  Map,
  Warning,
  CheckCircle,
  Info,
} from "@mui/icons-material";

export default function AboutPage() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        About Fire Forecasting
      </Typography>

      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        A machine learning system for predicting wildfire risk in the Tri-County
        area of California (Ventura, Santa Barbara, and Los Angeles counties).
      </Typography>

      {/* Project Overview */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Project Overview
          </Typography>
          <Typography variant="body1" paragraph>
            This system uses advanced machine learning techniques to predict the
            likelihood of wildfires occurring within the next 24 hours at
            specific monitoring sites. By analyzing weather patterns, historical
            fire data, and environmental conditions, we can provide early
            warning systems for fire-prone areas.
          </Typography>
          <Typography variant="body1">
            The system is designed to be fast (training completes in under 10
            minutes on CPU) and accurate, with a focus on precision-recall
            metrics due to the imbalanced nature of wildfire events.
          </Typography>
        </CardContent>
      </Card>

      {/* Data Sources */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Data Sources & Availability
          </Typography>

          <List>
            <ListItem>
              <ListItemIcon>
                <DataUsage color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="NASA FIRMS (Fire Information for Resource Management System)"
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ✅ <strong>Available:</strong> Local CSV files (2017-2024)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Satellite-based active fire detection used for creating
                      training labels
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Coverage:</strong> Tri-County area (Santa Barbara,
                      Ventura, Los Angeles)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Format:</strong> CSV with coordinates, dates,
                      confidence scores
                    </Typography>
                  </Box>
                }
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Map color="primary" />
              </ListItemIcon>
              <ListItemText
                primary="CAL FIRE FRAP (Fire and Resource Assessment Program)"
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ✅ <strong>Available:</strong> Real-time ArcGIS REST API →
                      GeoJSON
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Historical fire perimeter data for contextual mapping and
                      analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Coverage:</strong> California Historical Fire
                      Perimeters (2019-2024)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Processing:</strong> Geometry simplification for
                      web performance
                    </Typography>
                  </Box>
                }
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="RAWS (Remote Automated Weather Stations)"
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ⚠️ <strong>Status:</strong> Mock/Synthetic Data Only
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Currently using simulated weather data for demonstration
                      purposes
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Note:</strong> Real RAWS data integration requires
                      NOAA API access
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Coverage:</strong> 20 monitoring sites in
                      Tri-County area (simulated)
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>

      {/* Methodology */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Methodology
          </Typography>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Feature Engineering</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                Our system creates comprehensive features from weather data:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Lag features (1-7 days)" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Rolling statistics (7-day windows)" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Seasonal features (day-of-year, month)" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Neighbor site signals" />
                </ListItem>
              </List>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Labeling Methods</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                We use two approaches for creating fire labels:
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="FIRMS Proximity"
                    secondary="Fire detected within 15km buffer on next day"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText
                    primary="FWI Threshold"
                    secondary="Fire Weather Index above 85th percentile"
                  />
                </ListItem>
              </List>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Model Architecture</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                Our primary model is an Artificial Neural Network (ANN):
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Input layer: Variable features" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Hidden layers: 256 → 128 → 64 neurons" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Output: Sigmoid activation for probability" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircle color="success" fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Regularization: Dropout (0.2) and early stopping" />
                </ListItem>
              </List>
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Performance Metrics
          </Typography>

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2 }}>
            <Chip
              label="PR-AUC (Primary)"
              color="primary"
              variant="outlined"
              icon={<CheckCircle />}
            />
            <Chip label="ROC-AUC" color="secondary" variant="outlined" />
            <Chip label="Precision" color="success" variant="outlined" />
            <Chip label="Recall" color="warning" variant="outlined" />
            <Chip label="F1 Score" color="info" variant="outlined" />
          </Box>

          <Typography variant="body2" color="text.secondary">
            We focus on Precision-Recall AUC due to the imbalanced nature of
            wildfire events. High precision ensures we don't overwhelm emergency
            services with false alarms, while good recall ensures we don't miss
            actual fire events.
          </Typography>
        </CardContent>
      </Card>

      {/* Limitations */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Limitations & Considerations
          </Typography>

          <List>
            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Satellite Detection Latency"
                secondary="FIRMS data may have 1-2 day delays, affecting real-time predictions"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Weather Station Coverage"
                secondary="Some sites may be far from RAWS stations, affecting data quality"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Climate Change Impact"
                secondary="Historical patterns may not fully represent future fire conditions"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Warning color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Human Factors"
                secondary="Model doesn't account for arson, accidents, or other human-caused fires"
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>

      {/* Future Work */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Future Improvements
          </Typography>

          <List>
            <ListItem>
              <ListItemIcon>
                <Info color="info" />
              </ListItemIcon>
              <ListItemText
                primary="Hourly Data Integration"
                secondary="Incorporate hourly weather updates for more granular predictions"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Info color="info" />
              </ListItemIcon>
              <ListItemText
                primary="Fuel Moisture & NDVI"
                secondary="Add vegetation health and fuel moisture indicators"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Info color="info" />
              </ListItemIcon>
              <ListItemText
                primary="Topographic Features"
                secondary="Include elevation, slope, and aspect data"
              />
            </ListItem>

            <ListItem>
              <ListItemIcon>
                <Info color="info" />
              </ListItemIcon>
              <ListItemText
                primary="Ensemble Methods"
                secondary="Combine multiple models for improved robustness"
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>

      {/* Technical Details */}
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Technical Details
          </Typography>

          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2 }}>
            <Chip label="Python 3.10+" color="primary" />
            <Chip label="TensorFlow 2.16" color="primary" />
            <Chip label="FastAPI Backend" color="secondary" />
            <Chip label="Next.js Frontend" color="secondary" />
            <Chip label="Material UI" color="secondary" />
            <Chip label="ApexCharts" color="secondary" />
            <Chip label="Leaflet Maps" color="secondary" />
          </Box>

          <Typography variant="body2" color="text.secondary">
            The system is built with modern, scalable technologies. The backend
            uses FastAPI for high-performance API endpoints, while the frontend
            provides an intuitive dashboard for monitoring and analysis. All
            components are designed to handle the computational requirements of
            real-time wildfire prediction.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
