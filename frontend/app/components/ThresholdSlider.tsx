"use client";

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Slider,
  Paper,
  Grid,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from "@mui/material";
import { Tune, Speed, Precision, Visibility } from "@mui/icons-material";

interface ThresholdSliderProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  showPresets?: boolean;
  showMetrics?: boolean;
  onMetricsChange?: (metrics: {
    precision: number;
    recall: number;
    f1: number;
  }) => void;
}

interface PresetThreshold {
  name: string;
  value: number;
  description: string;
  color: "primary" | "secondary" | "success" | "warning" | "error" | "info";
}

const ThresholdSlider: React.FC<ThresholdSliderProps> = ({
  value,
  onChange,
  disabled = false,
  showPresets = true,
  showMetrics = true,
  onMetricsChange,
}) => {
  const [localValue, setLocalValue] = useState(value);
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  // Preset thresholds for common use cases
  const presets: PresetThreshold[] = [
    {
      name: "Balanced",
      value: 0.5,
      description: "Equal precision/recall",
      color: "primary",
    },
    {
      name: "High Precision",
      value: 0.8,
      description: "Minimize false positives",
      color: "success",
    },
    {
      name: "High Recall",
      value: 0.2,
      description: "Minimize false negatives",
      color: "warning",
    },
    {
      name: "Conservative",
      value: 0.9,
      description: "Very high precision",
      color: "error",
    },
    {
      name: "Sensitive",
      value: 0.1,
      description: "Very high recall",
      color: "info",
    },
  ];

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleSliderChange = (_event: Event, newValue: number | number[]) => {
    const threshold = newValue as number;
    setLocalValue(threshold);
    onChange(threshold);

    // Clear preset selection when manually adjusting
    setSelectedPreset("");
  };

  const handlePresetChange = (event: SelectChangeEvent) => {
    const presetName = event.target.value;
    const preset = presets.find((p) => p.name === presetName);

    if (preset) {
      setSelectedPreset(presetName);
      setLocalValue(preset.value);
      onChange(preset.value);
    }
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const threshold = parseFloat(event.target.value);
    if (!isNaN(threshold) && threshold >= 0 && threshold <= 1) {
      setLocalValue(threshold);
      onChange(threshold);
      setSelectedPreset("");
    }
  };

  const getThresholdLabel = (threshold: number): string => {
    if (threshold <= 0.1) return "Very Sensitive";
    if (threshold <= 0.3) return "Sensitive";
    if (threshold <= 0.5) return "Balanced";
    if (threshold <= 0.7) return "Precise";
    if (threshold <= 0.9) return "Very Precise";
    return "Ultra Precise";
  };

  const getThresholdColor = (
    threshold: number
  ): "success" | "warning" | "error" | "info" => {
    if (threshold <= 0.3) return "info";
    if (threshold <= 0.5) return "warning";
    if (threshold <= 0.7) return "success";
    return "error";
  };

  return (
    <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
      <Box display="flex" alignItems="center" mb={2}>
        <Tune sx={{ mr: 1, color: "primary.main" }} />
        <Typography variant="h6" component="h3" color="primary">
          Threshold Tuning
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Main Slider */}
        <Grid item xs={12} md={8}>
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Operating Threshold: {localValue.toFixed(3)}
            </Typography>
            <Slider
              value={localValue}
              onChange={handleSliderChange}
              min={0.01}
              max={0.99}
              step={0.01}
              disabled={disabled}
              marks={[
                { value: 0.1, label: "0.1" },
                { value: 0.3, label: "0.3" },
                { value: 0.5, label: "0.5" },
                { value: 0.7, label: "0.7" },
                { value: 0.9, label: "0.9" },
              ]}
              sx={{
                "& .MuiSlider-mark": {
                  backgroundColor: "primary.main",
                },
                "& .MuiSlider-markLabel": {
                  color: "text.secondary",
                  fontSize: "0.75rem",
                },
              }}
            />
          </Box>

          {/* Threshold Characteristics */}
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={getThresholdLabel(localValue)}
              color={getThresholdColor(localValue)}
              variant="outlined"
              size="small"
            />
            <Typography variant="body2" color="text.secondary">
              •{" "}
              {localValue <= 0.5
                ? "Higher recall, lower precision"
                : "Higher precision, lower recall"}
            </Typography>
          </Box>
        </Grid>

        {/* Presets */}
        {showPresets && (
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Quick Presets</InputLabel>
              <Select
                value={selectedPreset}
                label="Quick Presets"
                onChange={handlePresetChange}
                disabled={disabled}
              >
                {presets.map((preset) => (
                  <MenuItem key={preset.name} value={preset.name}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip
                        label={preset.name}
                        color={preset.color}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="caption" color="text.secondary">
                        {preset.value}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box mt={1}>
              <Typography variant="caption" color="text.secondary">
                {selectedPreset &&
                  presets.find((p) => p.name === selectedPreset)?.description}
              </Typography>
            </Box>
          </Grid>
        )}
      </Grid>

      {/* Manual Input */}
      <Box mt={2}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Manual Input:
        </Typography>
        <input
          type="number"
          min="0.01"
          max="0.99"
          step="0.01"
          value={localValue}
          onChange={handleInputChange}
          disabled={disabled}
          style={{
            width: "100px",
            padding: "8px",
            border: "1px solid #ccc",
            borderRadius: "4px",
            fontSize: "14px",
          }}
        />
      </Box>

      {/* Threshold Guidelines */}
      <Box mt={3} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Threshold Guidelines:
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" gap={1}>
              <Speed color="info" fontSize="small" />
              <Typography variant="caption">
                <strong>Low (0.1-0.3):</strong> High recall, many false
                positives
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center" gap={1}>
              <Precision color="success" fontSize="small" />
              <Typography variant="caption">
                <strong>High (0.7-0.9):</strong> High precision, many false
                negatives
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12}>
            <Box display="flex" alignItems="center" gap={1}>
              <Visibility color="warning" fontSize="small" />
              <Typography variant="caption">
                <strong>Balanced (0.4-0.6):</strong> Trade-off between precision
                and recall
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default ThresholdSlider;
