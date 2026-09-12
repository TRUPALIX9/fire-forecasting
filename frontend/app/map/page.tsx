"use client";

import React, { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Box, Card, Typography } from "@mui/material";
import ForecastControls from "../components/ForecastControls";
import LayerToggles, { Layer } from "../components/LayerToggles";
import MapLoading from "../components/MapLoading";
import { computeView, FIRST_DATE, Horizon } from "../../lib/forecast";

// Leaflet needs the browser: load the map on the client only
const MapLeaflet = dynamic(() => import("../components/MapLeaflet"), {
  ssr: false,
  loading: () => <MapLoading />,
});

export default function MapPage() {
  const [start, setStart] = useState(FIRST_DATE);
  const [horizon, setHorizon] = useState<Horizon>(72);
  const [layers, setLayers] = useState<Layer[]>(["sites", "bbox"]);

  const view = useMemo(() => computeView(start, horizon), [start, horizon]);

  return (
    <Box>
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
            Forecast Map
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Peak risk per site · {view.rangeLabel}
          </Typography>
        </Box>
        <ForecastControls
          start={start}
          horizon={horizon}
          onStartChange={setStart}
          onHorizonChange={setHorizon}
        />
      </Box>

      <Card>
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
          <Typography variant="body2" color="text.secondary">
            {view.sites.length} sites · {view.aboveThreshold} above τ ={" "}
            {view.threshold.toFixed(2)}
          </Typography>
          <LayerToggles value={layers} onChange={setLayers} />
        </Box>
        <Box sx={{ height: { xs: 420, md: "calc(100vh - 270px)" }, minHeight: 420 }}>
          <MapLeaflet
            sites={view.sites}
            threshold={view.threshold}
            showSites={layers.includes("sites")}
            showFRAP={layers.includes("frap")}
            showBoundingBox={layers.includes("bbox")}
          />
        </Box>
      </Card>
    </Box>
  );
}
