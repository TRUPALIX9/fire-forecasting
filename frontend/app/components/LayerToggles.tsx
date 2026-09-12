"use client";

import { Box, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { LocationOn, Fireplace, Terrain } from "@mui/icons-material";

export type Layer = "sites" | "frap" | "bbox";

interface LayerTogglesProps {
  value: Layer[];
  onChange: (layers: Layer[]) => void;
}

export default function LayerToggles({ value, onChange }: LayerTogglesProps) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
      <Typography variant="body2" sx={{ fontWeight: 500 }}>
        Layers:
      </Typography>
      <ToggleButtonGroup
        size="small"
        value={value}
        onChange={(_, layers: Layer[]) => onChange(layers)}
        aria-label="Map layers"
      >
        <ToggleButton value="sites" aria-label="Sites">
          <LocationOn sx={{ mr: 0.5 }} />
          Sites
        </ToggleButton>
        <ToggleButton value="frap" aria-label="FRAP fire perimeters">
          <Fireplace sx={{ mr: 0.5 }} fontSize="small" />
          FRAP
        </ToggleButton>
        <ToggleButton value="bbox" aria-label="Area of interest">
          <Terrain sx={{ mr: 0.5 }} />
          BBox
        </ToggleButton>
      </ToggleButtonGroup>
    </Box>
  );
}
