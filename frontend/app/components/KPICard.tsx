"use client";

import React from "react";
import { Card, CardContent, Typography, Box, Chip } from "@mui/material";
import { TrendingUp, TrendingDown, TrendingFlat } from "@mui/icons-material";

interface KPICardProps {
  title: string;
  value: number;
  unit?: string;
  trend?: "up" | "down" | "flat";
  color?: "primary" | "secondary" | "success" | "warning" | "error" | "info";
  subtitle?: string;
  precision?: number;
  showTrend?: boolean;
}

const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit = "",
  trend = "flat",
  color = "primary",
  subtitle,
  precision = 3,
  showTrend = true,
}) => {
  const formatValue = (val: number): string => {
    if (val !== 0 && Math.abs(val) < 0.001) {
      return val.toExponential(2);
    }
    return val.toFixed(precision);
  };

  const getTrendIcon = () => {
    switch (trend) {
      case "up":
        return <TrendingUp color="success" />;
      case "down":
        return <TrendingDown color="error" />;
      case "flat":
        return <TrendingFlat color="action" />;
      default:
        return null;
    }
  };

  const getValueColor = () => {
    // Color coding based on metric type and value
    if (title.toLowerCase().includes("auc")) {
      if (value >= 0.8) return "success.main";
      if (value >= 0.6) return "warning.main";
      return "error.main";
    }
    if (
      title.toLowerCase().includes("f1") ||
      title.toLowerCase().includes("precision") ||
      title.toLowerCase().includes("recall")
    ) {
      if (value >= 0.7) return "success.main";
      if (value >= 0.5) return "warning.main";
      return "error.main";
    }
    return "text.primary";
  };

  return (
    <Card
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        transition: "transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
        "&:hover": {
          transform: "translateY(-2px)",
          boxShadow: 4,
        },
        border: `2px solid`,
        borderColor: `${color}.main`,
      }}
    >
      <CardContent sx={{ flexGrow: 1, p: 2 }}>
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="flex-start"
          mb={1}
        >
          <Typography
            variant="h6"
            component="h3"
            color="text.secondary"
            sx={{
              fontSize: "0.875rem",
              fontWeight: 500,
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            {title}
          </Typography>
          {showTrend && getTrendIcon()}
        </Box>

        <Box display="flex" alignItems="baseline" mb={1}>
          <Typography
            variant="h4"
            component="div"
            sx={{
              fontWeight: "bold",
              color: getValueColor(),
              lineHeight: 1,
              mr: 0.5,
            }}
          >
            {formatValue(value)}
          </Typography>
          {unit && (
            <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
              {unit}
            </Typography>
          )}
        </Box>

        {subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {subtitle}
          </Typography>
        )}

        {showTrend && trend !== "flat" && (
          <Chip
            label={trend === "up" ? "Improving" : "Declining"}
            size="small"
            color={trend === "up" ? "success" : "error"}
            variant="outlined"
            sx={{
              fontSize: "0.75rem",
              height: "20px",
            }}
          />
        )}
      </CardContent>
    </Card>
  );
};

export default KPICard;
