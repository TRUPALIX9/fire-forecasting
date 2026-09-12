"use client";

import React from "react";
import dynamic from "next/dynamic";
import type { ApexOptions } from "apexcharts";
import { Box, Card, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { SiteView, parseTime } from "../../lib/forecast";

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface ForecastChartProps {
  site: SiteView;
  times: string[];
  threshold: number;
  height?: number;
}

const ForecastChart: React.FC<ForecastChartProps> = ({
  site,
  times,
  threshold,
  height = 150,
}) => {
  const theme = useTheme();
  const color = "#ff9800";
  const danger = theme.palette.error.main;
  const data = site.series.map((y, i) => ({ x: parseTime(times[i]).getTime(), y }));
  const peakX = parseTime(site.peakTime).getTime();

  const options: ApexOptions = {
    chart: {
      type: "area",
      fontFamily: theme.typography.fontFamily,
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: false,
          zoom: false,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true,
        },
      },
      animations: { enabled: false },
    },
    colors: [color],
    stroke: { curve: "smooth", width: 2.5 },
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.45,
        opacityTo: 0.04,
        stops: [0, 100],
      },
    },
    dataLabels: { enabled: false },
    markers: { size: 0, hover: { size: 4 } },
    grid: { borderColor: "#e0e0e0", strokeDashArray: 4 },
    xaxis: {
      type: "datetime",
      labels: {
        datetimeUTC: false,
        datetimeFormatter: { day: "dd MMM", hour: "HH:mm" },
      },
      tooltip: { enabled: false },
    },
    yaxis: {
      min: 0,
      max: 1,
      tickAmount: 4,
      labels: { formatter: (v: number) => v.toFixed(2) },
    },
    tooltip: {
      x: { format: "ddd dd MMM, HH:mm" },
      y: { formatter: (v: number) => v.toFixed(2), title: { formatter: () => "Risk" } },
    },
    annotations: {
      yaxis: [
        {
          y: threshold,
          borderColor: danger,
          strokeDashArray: 5,
          label: {
            text: `τ = ${threshold.toFixed(2)}`,
            position: "left",
            textAnchor: "start",
            offsetX: 4,
            borderColor: danger,
            style: { background: danger, color: "#fff", fontSize: "10px", fontWeight: 600 },
          },
        },
      ],
      points: [
        {
          x: peakX,
          y: site.peak,
          marker: { size: 4.5, fillColor: "#fff", strokeColor: danger, strokeWidth: 2 },
          label: {
            text: `Peak ${site.peak.toFixed(2)}`,
            borderColor: danger,
            style: { background: "#fff", color: danger, fontSize: "10px", fontWeight: 600 },
          },
        },
      ],
    },
    legend: { show: false },
  };

  return (
    <Card>
      <Box sx={{ px: 2, pt: 2 }}>
        <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600 }}>
          Fire Risk Forecast
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {site.name} · P(forest_fire = Y)
        </Typography>
      </Box>
      <Box sx={{ px: 1, pb: 0.5 }}>
        <Chart
          type="area"
          height={height}
          options={options}
          series={[{ name: site.name, data }]}
        />
      </Box>
    </Card>
  );
};

export default ForecastChart;
