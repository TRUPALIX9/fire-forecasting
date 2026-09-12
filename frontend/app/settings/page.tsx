"use client";

import React, { useEffect, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from "@mui/material";
import {
  Settings as SettingsIcon,
  Save,
  Refresh,
  Info,
} from "@mui/icons-material";
import { forecast, formatRange } from "../../lib/forecast";

const STORAGE_KEY = "fire-forecasting:settings";

const DEFAULT_SETTINGS = {
  notifications: true,
  autoRefresh: false,
  refreshInterval: 30,
  temperatureUnit: "celsius",
  windSpeedUnit: "m/s",
  pressureUnit: "hPa",
  theme: "light",
  dataRetention: 365,
};

type SettingsState = typeof DEFAULT_SETTINGS;

const sampleRange = formatRange(
  forecast.weather[0].time,
  forecast.weather[forecast.weather.length - 1].time
);

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
  const [alert, setAlert] = useState<"saved" | "error" | null>(null);

  // Load saved settings from this browser
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(saved) });
    } catch {
      // Storage unavailable or malformed: keep defaults
    }
  }, []);

  const handleSettingChange = <K extends keyof SettingsState>(
    setting: K,
    value: SettingsState[K]
  ) => {
    setSettings((prev) => ({
      ...prev,
      [setting]: value,
    }));
  };

  // Number fields: ignore empty/invalid input and keep values >= 1
  const handleNumberChange = (
    setting: "refreshInterval" | "dataRetention",
    raw: string
  ) => {
    const value = parseInt(raw, 10);
    if (!Number.isNaN(value)) handleSettingChange(setting, Math.max(1, value));
  };

  const handleSaveSettings = () => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      setAlert("saved");
    } catch {
      setAlert("error");
    }
    setTimeout(() => setAlert(null), 3000);
  };

  const handleResetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          <SettingsIcon sx={{ mr: 2, verticalAlign: "middle" }} />
          Settings
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Configure your fire forecasting dashboard preferences
        </Typography>
      </Box>

      {alert && (
        <Alert
          severity={alert === "saved" ? "success" : "error"}
          sx={{ mb: 3 }}
          onClose={() => setAlert(null)}
        >
          {alert === "saved"
            ? "Settings saved in this browser."
            : "Settings could not be saved: browser storage is unavailable."}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                General Settings
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <FormControlLabel
                control={
                  <Switch
                    checked={settings.notifications}
                    onChange={(e) =>
                      handleSettingChange("notifications", e.target.checked)
                    }
                  />
                }
                label="Enable Notifications"
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={settings.autoRefresh}
                    onChange={(e) =>
                      handleSettingChange("autoRefresh", e.target.checked)
                    }
                  />
                }
                label="Auto Refresh Data"
                sx={{ display: "block", mt: 1 }}
              />

              <TextField
                label="Refresh Interval (seconds)"
                type="number"
                value={settings.refreshInterval}
                onChange={(e) =>
                  handleNumberChange("refreshInterval", e.target.value)
                }
                inputProps={{ min: 1 }}
                disabled={!settings.autoRefresh}
                fullWidth
                sx={{ mt: 2 }}
                helperText={
                  !settings.autoRefresh
                    ? "Enable auto refresh to set interval"
                    : ""
                }
              />

              <TextField
                label="Data Retention (days)"
                type="number"
                value={settings.dataRetention}
                onChange={(e) =>
                  handleNumberChange("dataRetention", e.target.value)
                }
                inputProps={{ min: 1 }}
                fullWidth
                sx={{ mt: 2 }}
                helperText="How long to keep historical data"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Display Settings */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Display Settings
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <TextField
                select
                label="Temperature Unit"
                value={settings.temperatureUnit}
                onChange={(e) =>
                  handleSettingChange("temperatureUnit", e.target.value)
                }
                fullWidth
                sx={{ mb: 2 }}
                SelectProps={{
                  native: true,
                }}
              >
                <option value="celsius">Celsius (°C)</option>
                <option value="fahrenheit">Fahrenheit (°F)</option>
              </TextField>

              <TextField
                select
                label="Wind Speed Unit"
                value={settings.windSpeedUnit}
                onChange={(e) =>
                  handleSettingChange("windSpeedUnit", e.target.value)
                }
                fullWidth
                sx={{ mb: 2 }}
                SelectProps={{
                  native: true,
                }}
              >
                <option value="m/s">Meters per second (m/s)</option>
                <option value="mph">Miles per hour (mph)</option>
                <option value="km/h">Kilometers per hour (km/h)</option>
              </TextField>

              <TextField
                select
                label="Pressure Unit"
                value={settings.pressureUnit}
                onChange={(e) =>
                  handleSettingChange("pressureUnit", e.target.value)
                }
                fullWidth
                sx={{ mb: 2 }}
                SelectProps={{
                  native: true,
                }}
              >
                <option value="hPa">Hectopascal (hPa)</option>
                <option value="mbar">Millibar (mbar)</option>
                <option value="inHg">Inches of Mercury (inHg)</option>
              </TextField>

              <TextField
                select
                label="Theme"
                value={settings.theme}
                onChange={(e) => handleSettingChange("theme", e.target.value)}
                fullWidth
                SelectProps={{
                  native: true,
                }}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="auto">Auto</option>
              </TextField>
            </CardContent>
          </Card>
        </Grid>

        {/* Data Sources */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Sources
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Paper variant="outlined" sx={{ p: 2 }}>
                <List>
                  <ListItem>
                    <ListItemText
                      primary="Trihourly Weather Dataset"
                      secondary="18,069 records - Every 3 hours weather data (2020-01-01 to 2024-01-01), data/trihourly_weather.csv"
                    />
                    <ListItemSecondaryAction>
                      <Info color="action" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Sample Forecast (bundled)"
                      secondary={`${forecast.sites.length} fictional sites - ${forecast.weather.length} steps of ${forecast.stepHours} hours (${sampleRange}), illustrative risk values`}
                    />
                    <ListItemSecondaryAction>
                      <Info color="action" />
                    </ListItemSecondaryAction>
                  </ListItem>
                </List>
              </Paper>
            </CardContent>
          </Card>
        </Grid>

        {/* Actions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={handleResetSettings}
                >
                  Reset to Defaults
                </Button>
                <Button
                  variant="contained"
                  startIcon={<Save />}
                  onClick={handleSaveSettings}
                >
                  Save Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Info Alert */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Note:</strong> Settings are saved in this browser
          (localStorage). This prototype has no backend; the dashboard shows
          metric units and the light theme.
        </Typography>
      </Alert>
    </Box>
  );
}
