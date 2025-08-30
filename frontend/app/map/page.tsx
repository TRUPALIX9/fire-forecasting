"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Slider,
  Button,
  Grid,
  ToggleButton,
  ToggleButtonGroup,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import {
  Refresh,
  Settings,
  LocationOn,
  Fireplace,
  Terrain,
  LocalFireDepartment,
  Download,
} from "@mui/icons-material";
import dynamic from "next/dynamic";
import { useFRAPData } from "../../hooks/useFRAPData";

// Dynamically import map components to avoid SSR issues
const MapLeaflet = dynamic(() => import("../components/MapLeaflet"), {
  ssr: false,
});

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function MapPage() {
  // State for map layers
  const [showSites, setShowSites] = useState(true);
  const [showFRAP, setShowFRAP] = useState(true);
  const [showBoundingBox, setShowBoundingBox] = useState(false);

  // State for map settings modal
  const [mapSettingsOpen, setMapSettingsOpen] = useState(false);

  // Advanced controls
  const [mapOpacity, setMapOpacity] = useState(0.8);
  const [frapOpacity, setFrapOpacity] = useState(0.6);
  const [sitesOpacity, setSitesOpacity] = useState(0.8);
  const [boundingBoxOpacity, setBoundingBoxOpacity] = useState(0.7);

  // Map view controls
  const [mapView, setMapView] = useState("standard");
  const [autoFitBounds, setAutoFitBounds] = useState(true);

  // Data filtering
  const [confidenceThreshold, setConfidenceThreshold] = useState(20);
  const [selectedAgencies, setSelectedAgencies] = useState<string[]>([]);

  // UI states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch FRAP data for the fire list
  const {
    filteredData: fireListData,
    loading: fireListLoading,
    error: fireListError,
  } = useFRAPData(null, null, false, [0, 100000]);

  useEffect(() => {
    // Simulate loading time for map initialization
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  // Log filter changes for debugging
  useEffect(() => {
    console.log("🔍 Filter state updated:", {
      showSites,
      showFRAP,
      showBoundingBox,
      frapOpacity: Math.round(frapOpacity * 100),
      sitesOpacity: Math.round(sitesOpacity * 100),
    });
  }, [showSites, showFRAP, showBoundingBox, frapOpacity, sitesOpacity]);

  const handleResetFilters = () => {
    setMapOpacity(0.8);
    setFrapOpacity(0.6);
    setSitesOpacity(0.9);
    setBoundingBoxOpacity(0.7);
    console.log("🎯 Filters reset to defaults");
  };

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight={400}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight={400}
      >
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box>
      {/* Horizontal Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent sx={{ py: 2 }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 3,
              flexWrap: "wrap",
            }}
          >
            {/* Layer Toggles */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography
                variant="subtitle2"
                sx={{ mr: 1, whiteSpace: "nowrap" }}
              >
                Layers:
              </Typography>
              <ToggleButtonGroup
                value={[
                  showSites && "sites",
                  showFRAP && "frap",
                  showBoundingBox && "bbox",
                ].filter(Boolean)}
                onChange={(_, newValues) => {
                  setShowSites(newValues.includes("sites"));
                  setShowFRAP(newValues.includes("frap"));
                  setShowBoundingBox(newValues.includes("bbox"));
                }}
                size="small"
              >
                <ToggleButton value="sites">
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <LocationOn fontSize="small" />
                    Sites
                  </Box>
                </ToggleButton>
                <ToggleButton value="frap">
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Fireplace fontSize="small" />
                    FRAP
                  </Box>
                </ToggleButton>
                <ToggleButton value="bbox">
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Terrain fontSize="small" />
                    BBox
                  </Box>
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            {/* Map Settings Button */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<Settings />}
                size="small"
                onClick={() => setMapSettingsOpen(true)}
                sx={{ whiteSpace: "nowrap" }}
              >
                Map Settings
              </Button>
            </Box>

            {/* Reset Button */}
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleResetFilters}
              color="secondary"
              size="small"
              sx={{ whiteSpace: "nowrap" }}
            >
              Reset
            </Button>
          </Box>
        </CardContent>
      </Card>
      {/* Map */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          <Box sx={{ height: 700, width: "100%" }}>
            <MapLeaflet
              showSites={showSites}
              showFRAP={showFRAP}
              showBoundingBox={showBoundingBox}
              mapOpacity={mapOpacity}
              frapOpacity={frapOpacity}
              sitesOpacity={sitesOpacity}
              boundingBoxOpacity={boundingBoxOpacity}
              mapView={mapView}
              autoFitBounds={autoFitBounds}
            />
          </Box>
        </CardContent>
      </Card>

      {/* Map Settings Modal */}
      <Dialog
        open={mapSettingsOpen}
        onClose={() => setMapSettingsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Settings color="primary" />
            Map Settings
          </Box>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            {/* Layer Opacity Controls */}
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Layer Opacity
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    FRAP Fire Perimeters: {Math.round(frapOpacity * 100)}%
                  </Typography>
                  <Slider
                    value={frapOpacity}
                    onChange={(_, value) => setFrapOpacity(value as number)}
                    min={0.1}
                    max={1}
                    step={0.1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Box>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Monitoring Sites: {Math.round(sitesOpacity * 100)}%
                  </Typography>
                  <Slider
                    value={sitesOpacity}
                    onChange={(_, value) => setSitesOpacity(value as number)}
                    min={0.1}
                    max={1}
                    step={0.1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Box>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Bounding Box: {Math.round(boundingBoxOpacity * 100)}%
                  </Typography>
                  <Slider
                    value={boundingBoxOpacity}
                    onChange={(_, value) =>
                      setBoundingBoxOpacity(value as number)
                    }
                    min={0.1}
                    max={1}
                    step={0.1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Box>
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMapSettingsOpen(false)} color="secondary">
            Cancel
          </Button>
          <Button
            onClick={() => {
              setMapSettingsOpen(false);
              console.log("🎛️ Map settings applied");
            }}
            color="primary"
            variant="contained"
          >
            Apply Settings
          </Button>
        </DialogActions>
      </Dialog>

      {/* Fire List Section */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography
            variant="h5"
            gutterBottom
            sx={{ display: "flex", alignItems: "center", gap: 1 }}
          >
            <LocalFireDepartment color="primary" />
            Fire Events (Sorted by Size)
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Historical fire events in the Tri-County area, filtered by current
            map settings
          </Typography>

          <Box
            sx={{
              maxHeight: 400,
              overflow: "auto",
              border: "1px solid #e0e0e0",
              borderRadius: 1,
            }}
          >
            <TableContainer>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600 }}>Fire Name</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Year</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Size (Acres)</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>County</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {fireListLoading ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <CircularProgress size={20} />
                      </TableCell>
                    </TableRow>
                  ) : fireListError ? (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Alert severity="error">{fireListError}</Alert>
                      </TableCell>
                    </TableRow>
                  ) : fireListData && fireListData.length > 0 ? (
                    fireListData
                      .sort((a: any, b: any) => b.size - a.size) // Sort by size descending
                      .map((fire: any, index: any) => (
                        <TableRow key={index} hover>
                          <TableCell>
                            <Typography
                              variant="body2"
                              sx={{ fontWeight: 500 }}
                            >
                              {fire.name}
                            </Typography>
                          </TableCell>
                          <TableCell>{fire.year}</TableCell>
                          <TableCell>
                            <Typography
                              variant="body2"
                              sx={{
                                fontWeight: 500,
                                color:
                                  fire.size > 100000
                                    ? "#d32f2f"
                                    : fire.size > 50000
                                    ? "#f57c00"
                                    : "#1976d2",
                              }}
                            >
                              {fire.size.toLocaleString()}
                            </Typography>
                          </TableCell>
                          <TableCell>{fire.county}</TableCell>
                          <TableCell>
                            <Chip
                              label={fire.status}
                              size="small"
                              color={
                                fire.status === "Contained"
                                  ? "success"
                                  : "warning"
                              }
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary">
                              {fire.details}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        No fire events found for the selected date range and
                        filters.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>

          <Box
            sx={{
              mt: 2,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="body2" color="text.secondary">
              Showing {fireListData?.length || 0} fires • Total affected area:{" "}
              {fireListData
                ?.reduce((sum, fire) => sum + fire.size, 0)
                .toLocaleString()}{" "}
              acres
            </Typography>
            <Button variant="outlined" size="small" startIcon={<Download />}>
              Export Data
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
