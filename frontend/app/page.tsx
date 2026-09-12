"use client";

import React, { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Box, Card, Grid, Typography } from "@mui/material";
import KPICard from "./components/KPICard";
import ForecastControls from "./components/ForecastControls";
import ForecastChart from "./components/ForecastChart";
import SitesTable from "./components/SitesTable";
import LayerToggles, { Layer } from "./components/LayerToggles";
import MapLoading from "./components/MapLoading";
import {
  computeView,
  FIRST_DATE,
  formatDay,
  formatDayTime,
  Horizon,
} from "../lib/forecast";

// Leaflet needs the browser: load the map on the client only
const MapLeaflet = dynamic(() => import("./components/MapLeaflet"), {
  ssr: false,
  loading: () => <MapLoading />,
});

export default function DashboardPage() {
  const [start, setStart] = useState(FIRST_DATE);
  const [horizon, setHorizon] = useState<Horizon>(72);
  const [layers, setLayers] = useState<Layer[]>(["sites", "bbox"]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const view = useMemo(() => computeView(start, horizon), [start, horizon]);
  const selected =
    view.sites.find((s) => s.id === selectedId) ?? view.peakSite;

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 2,
          mb: 2.5,
        }}
      >
        <Box>
          <Typography variant="h5" component="h1" sx={{ fontWeight: 600 }}>
            Fire Forecasting Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Sample forecasts for the Tri-County area from the bundled trihourly
            weather dataset
          </Typography>
        </Box>
        <ForecastControls
          start={start}
          horizon={horizon}
          onStartChange={setStart}
          onHorizonChange={setHorizon}
        />
      </Box>

      {/* KPI cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Peak fire risk"
            value={view.peakSite.peak}
            precision={2}
            color="error"
            subtitle={`${view.peakSite.name} · ${formatDayTime(view.peakSite.peakTime)}`}
            showTrend={false}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Sites above threshold"
            value={view.aboveThreshold}
            precision={0}
            unit={`of ${view.sites.length} sites`}
            color="warning"
            subtitle={`Decision threshold τ = ${view.threshold.toFixed(2)}`}
            showTrend={false}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Max temperature"
            value={view.maxTemp.temperature_2m}
            precision={1}
            unit="°C"
            color="primary"
            subtitle={`temperature_2m · ${formatDayTime(view.maxTemp.time)}`}
            showTrend={false}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Min humidity"
            value={view.minHumidity.relative_humidity_2m}
            precision={1}
            unit="%"
            color="info"
            subtitle={`relative_humidity_2m · ${formatDay(view.minHumidity.time)}`}
            showTrend={false}
          />
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        {/* Forecast map */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
            <Box
              sx={{
                px: 2,
                py: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 1,
                borderBottom: 1,
                borderColor: "divider",
              }}
            >
              <Box>
                <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600 }}>
                  Forecast Map
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Peak risk per site · {view.rangeLabel}
                </Typography>
              </Box>
              <LayerToggles value={layers} onChange={setLayers} />
            </Box>
            <Box sx={{ flexGrow: 1, minHeight: { xs: 360, md: 420 } }}>
              <MapLeaflet
                sites={view.sites}
                threshold={view.threshold}
                showSites={layers.includes("sites")}
                showFRAP={layers.includes("frap")}
                showBoundingBox={layers.includes("bbox")}
                onSelectSite={setSelectedId}
                expandHref="/map"
              />
            </Box>
          </Card>
        </Grid>

        {/* Risk chart + site table */}
        <Grid item xs={12} md={5}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            <ForecastChart
              site={selected}
              times={view.times}
              threshold={view.threshold}
            />
            <SitesTable
              sites={view.sites}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
