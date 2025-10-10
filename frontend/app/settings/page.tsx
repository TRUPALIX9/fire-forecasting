"use client";

import React, { useState } from "react";
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
  Warning,
} from "@mui/icons-material";

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    notifications: true,
    autoRefresh: false,
    refreshInterval: 30,
    temperatureUnit: "celsius",
    windSpeedUnit: "m/s",
    pressureUnit: "hPa",
    theme: "light",
    dataRetention: 365,
  });

  const [showAlert, setShowAlert] = useState(false);

  const handleSettingChange = (setting: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      [setting]: value,
    }));
  };

  const handleSaveSettings = () => {
    // Simulate saving settings
    setShowAlert(true);
    setTimeout(() => setShowAlert(false), 3000);
  };

  const handleResetSettings = () => {
    setSettings({
      notifications: true,
      autoRefresh: false,
      refreshInterval: 30,
      temperatureUnit: "celsius",
      windSpeedUnit: "m/s",
      pressureUnit: "hPa",
      theme: "light",
      dataRetention: 365,
    });
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

      {showAlert && (
        <Alert
          severity="success"
          sx={{ mb: 3 }}
          onClose={() => setShowAlert(false)}
        >
          Settings saved successfully!
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12} md={6}>
          <Card>
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
                  handleSettingChange(
                    "refreshInterval",
                    parseInt(e.target.value)
                  )
                }
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
                  handleSettingChange("dataRetention", parseInt(e.target.value))
                }
                fullWidth
                sx={{ mt: 2 }}
                helperText="How long to keep historical data"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Display Settings */}
        <Grid item xs={12} md={6}>
          <Card>
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
                      primary="Hourly Weather Dataset"
                      secondary="40,711 records - Every hour weather data"
                    />
                    <ListItemSecondaryAction>
                      <Info color="action" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Bihourly Weather Dataset"
                      secondary="23,954 records - Every 2 hours weather data"
                    />
                    <ListItemSecondaryAction>
                      <Info color="action" />
                    </ListItemSecondaryAction>
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Trihourly Weather Dataset"
                      secondary="18,110 records - Every 3 hours weather data"
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
          <strong>Note:</strong> This is a frontend-only application. Settings
          are stored locally in your browser. No backend API is currently
          available for persistent settings storage.
        </Typography>
      </Alert>
    </Box>
  );
}
