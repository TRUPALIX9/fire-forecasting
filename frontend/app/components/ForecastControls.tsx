"use client";

import {
  Box,
  Chip,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { InfoOutlined } from "@mui/icons-material";
import { FIRST_DATE, LAST_DATE, HORIZONS, Horizon } from "../../lib/forecast";

interface ForecastControlsProps {
  start: string;
  horizon: Horizon;
  onStartChange: (date: string) => void;
  onHorizonChange: (horizon: Horizon) => void;
}

export default function ForecastControls({
  start,
  horizon,
  onStartChange,
  onHorizonChange,
}: ForecastControlsProps) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap" }}>
      <Chip
        icon={<InfoOutlined />}
        label="Sample data"
        color="warning"
        variant="outlined"
        title="Weather rows come from the bundled trihourly CSV; sites and risk values are illustrative samples, not model output."
        sx={{ color: "#b34f00", fontWeight: 500 }}
      />
      <TextField
        size="small"
        type="date"
        label="Forecast start"
        value={start}
        onChange={(e) => {
          const value = e.target.value;
          if (value >= FIRST_DATE && value <= LAST_DATE) onStartChange(value);
        }}
        InputLabelProps={{ shrink: true }}
        inputProps={{ min: FIRST_DATE, max: LAST_DATE }}
        sx={{ width: 170, bgcolor: "background.paper" }}
      />
      <ToggleButtonGroup
        size="small"
        color="primary"
        exclusive
        value={horizon}
        onChange={(_, value: Horizon | null) => value && onHorizonChange(value)}
        aria-label="Forecast horizon"
        sx={{ bgcolor: "background.paper" }}
      >
        {HORIZONS.map((h) => (
          <ToggleButton key={h} value={h} sx={{ px: 2 }}>
            {h}h
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
}
