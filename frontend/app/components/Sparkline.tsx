"use client";

import React from "react";
import { Box, Typography, Paper, Tooltip } from "@mui/material";
import dynamic from "next/dynamic";

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface SparklineProps {
  data: number[];
  dates: string[];
  title: string;
  subtitle?: string;
  height?: number;
  width?: number;
  showPoints?: boolean;
  color?: string;
  showTooltip?: boolean;
  formatValue?: (value: number) => string;
  highlightPoints?: number[]; // Indices of points to highlight
  highlightColor?: string;
  showTrend?: boolean;
  trendDirection?: "up" | "down" | "flat";
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  dates,
  title,
  subtitle,
  height = 100,
  width = "100%",
  showPoints = false,
  color = "#2196f3",
  showTooltip = true,
  formatValue = (value) => value.toFixed(3),
  highlightPoints = [],
  highlightColor = "#f44336",
  showTrend = true,
  trendDirection = "flat",
}) => {
  // Calculate trend
  const calculateTrend = () => {
    if (data.length < 2) return "flat";

    const firstHalf = data.slice(0, Math.floor(data.length / 2));
    const secondHalf = data.slice(Math.floor(data.length / 2));

    const firstAvg =
      firstHalf.reduce((sum, val) => sum + val, 0) / firstHalf.length;
    const secondAvg =
      secondHalf.reduce((sum, val) => sum + val, 0) / secondHalf.length;

    const change = secondAvg - firstAvg;
    const threshold = Math.max(...data) * 0.1; // 10% of max value

    if (Math.abs(change) < threshold) return "flat";
    return change > 0 ? "up" : "down";
  };

  const actualTrend =
    trendDirection === "flat" ? calculateTrend() : trendDirection;

  // Chart options
  const chartOptions = {
    chart: {
      type: "line" as const,
      sparkline: {
        enabled: true,
      },
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
    },
    stroke: {
      curve: "smooth" as const,
      width: 2,
      colors: [color],
    },
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.7,
        opacityTo: 0.1,
        stops: [0, 100],
        colorStops: [
          {
            offset: 0,
            color: color,
            opacity: 0.7,
          },
          {
            offset: 100,
            color: color,
            opacity: 0.1,
          },
        ],
      },
    },
    markers: {
      size: showPoints ? 3 : 0,
      colors: [color],
      strokeColors: "#fff",
      strokeWidth: 1,
      hover: {
        size: 5,
      },
    },
    tooltip: {
      enabled: showTooltip,
      theme: "light" as const,
      x: {
        show: true,
        format: "MMM dd, yyyy",
      },
      y: {
        title: {
          formatter: () => title,
        },
        formatter: (value: number) => formatValue(value),
      },
      marker: {
        show: false,
      },
    },
    grid: {
      show: false,
    },
    xaxis: {
      type: "datetime" as const,
      labels: {
        show: false,
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    yaxis: {
      labels: {
        show: false,
      },
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    dataLabels: {
      enabled: false,
    },
    legend: {
      show: false,
    },
    responsive: [
      {
        breakpoint: 480,
        options: {
          chart: {
            height: height * 0.8,
          },
        },
      },
    ],
  };

  // Prepare data for chart
  const chartData = [
    {
      name: title,
      data: dates.map((date, index) => ({
        x: new Date(date).getTime(),
        y: data[index],
        // Add custom properties for highlighting
        custom: highlightPoints.includes(index),
      })),
    },
  ];

  // Custom tooltip for highlighted points
  const customTooltip = ({ series, seriesIndex, dataPointIndex, w }: any) => {
    if (!showTooltip) return "";

    const point = series[seriesIndex][dataPointIndex];
    const date = new Date(dates[dataPointIndex]).toLocaleDateString();
    const value = formatValue(point);
    const isHighlighted = highlightPoints.includes(dataPointIndex);

    return `
      <div style="padding: 8px; background: white; border: 1px solid #ccc; border-radius: 4px;">
        <div style="font-weight: bold; color: ${
          isHighlighted ? highlightColor : color
        };">
          ${title}
        </div>
        <div style="margin: 4px 0;">
          <strong>Date:</strong> ${date}
        </div>
        <div style="margin: 4px 0;">
          <strong>Value:</strong> ${value}
        </div>
        ${
          isHighlighted
            ? '<div style="color: #f44336; font-weight: bold;">⚠️ Fire Event</div>'
            : ""
        }
      </div>
    `;
  };

  // Enhanced chart options with custom tooltip
  const enhancedOptions = {
    ...chartOptions,
    tooltip: {
      ...chartOptions.tooltip,
      custom: customTooltip,
    },
  };

  // Get trend icon and color
  const getTrendIcon = () => {
    switch (actualTrend) {
      case "up":
        return "↗️";
      case "down":
        return "↘️";
      case "flat":
        return "→";
      default:
        return "→";
    }
  };

  const getTrendColor = () => {
    switch (actualTrend) {
      case "up":
        return "success.main";
      case "down":
        return "error.main";
      case "flat":
        return "text.secondary";
      default:
        return "text.secondary";
    }
  };

  // Calculate statistics
  const stats = {
    current: data[data.length - 1] || 0,
    average: data.reduce((sum, val) => sum + val, 0) / data.length || 0,
    min: Math.min(...data) || 0,
    max: Math.max(...data) || 0,
    change: data.length > 1 ? data[data.length - 1] - data[0] : 0,
    changePercent:
      data.length > 1 ? ((data[data.length - 1] - data[0]) / data[0]) * 100 : 0,
  };

  return (
    <Paper elevation={1} sx={{ p: 2, height: "100%" }}>
      {/* Header */}
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mb={2}
      >
        <Box>
          <Typography
            variant="h6"
            component="h3"
            color="text.primary"
            gutterBottom
          >
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary">
              {subtitle}
            </Typography>
          )}
        </Box>

        {showTrend && (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="h4" component="span">
              {getTrendIcon()}
            </Typography>
            <Typography
              variant="caption"
              color={getTrendColor()}
              sx={{ fontWeight: "bold" }}
            >
              {actualTrend.toUpperCase()}
            </Typography>
          </Box>
        )}
      </Box>

      {/* Chart */}
      <Box sx={{ mb: 2 }}>
        <Chart
          options={enhancedOptions}
          series={chartData}
          type="line"
          height={height}
          width={width}
        />
      </Box>

      {/* Statistics */}
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="h6" color="primary" gutterBottom>
            Current
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: "bold" }}>
            {formatValue(stats.current)}
          </Typography>
        </Box>

        <Box textAlign="right">
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Change
          </Typography>
          <Typography
            variant="h6"
            color={stats.change >= 0 ? "success.main" : "error.main"}
            sx={{ fontWeight: "bold" }}
          >
            {stats.change >= 0 ? "+" : ""}
            {formatValue(stats.change)}
          </Typography>
          <Typography
            variant="caption"
            color={stats.changePercent >= 0 ? "success.main" : "error.main"}
          >
            ({stats.changePercent >= 0 ? "+" : ""}
            {stats.changePercent.toFixed(1)}%)
          </Typography>
        </Box>
      </Box>

      {/* Additional Stats */}
      <Box
        display="flex"
        justifyContent="space-between"
        mt={2}
        pt={2}
        borderTop="1px solid"
        borderColor="divider"
      >
        <Box textAlign="center">
          <Typography variant="caption" color="text.secondary">
            Average
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: "bold" }}>
            {formatValue(stats.average)}
          </Typography>
        </Box>

        <Box textAlign="center">
          <Typography variant="caption" color="text.secondary">
            Range
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: "bold" }}>
            {formatValue(stats.min)} - {formatValue(stats.max)}
          </Typography>
        </Box>

        <Box textAlign="center">
          <Typography variant="caption" color="text.secondary">
            Points
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: "bold" }}>
            {data.length}
          </Typography>
        </Box>
      </Box>

      {/* Highlighted Points Legend */}
      {highlightPoints.length > 0 && (
        <Box mt={2} p={1} bgcolor="grey.50" borderRadius={1}>
          <Typography variant="caption" color="text.secondary">
            🔴 {highlightPoints.length} highlighted point(s) - Fire events or
            anomalies
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default Sparkline;
