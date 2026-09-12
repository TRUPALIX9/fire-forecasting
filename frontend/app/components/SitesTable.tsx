"use client";

import React from "react";
import {
  Box,
  Card,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { RiskLevel, SiteView } from "../../lib/forecast";

const levelChipSx: Record<RiskLevel, object> = {
  Extreme: { bgcolor: "error.main", color: "#fff" },
  High: { bgcolor: "warning.main", color: "#fff" },
  Moderate: { bgcolor: "#fff8e1", borderColor: "#f9a825", color: "#7a5a00" },
  Low: { bgcolor: "success.main", color: "#fff" },
};

export function LevelChip({ level }: { level: RiskLevel }) {
  return (
    <Chip
      label={level}
      size="small"
      variant={level === "Moderate" ? "outlined" : "filled"}
      sx={levelChipSx[level]}
    />
  );
}

interface SitesTableProps {
  sites: SiteView[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}

const SitesTable: React.FC<SitesTableProps> = ({
  sites,
  selectedId,
  onSelect,
}) => (
  <Card>
    <Box
      sx={{
        px: 2,
        pt: 2,
        pb: 1,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
      }}
    >
      <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600 }}>
        Site Forecasts
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {sites.length} sites
      </Typography>
    </Box>
    {/* Tighter inner padding so the table fits the card at desktop widths;
        on narrow screens it scrolls instead of being clipped by the Card. */}
    <TableContainer sx={{ overflowX: "auto" }}>
      <Table
        size="small"
        aria-label="Site forecasts"
        sx={{
          "& td, & th": { whiteSpace: "nowrap", px: 1.25 },
          "& td:first-of-type, & th:first-of-type": { pl: 2 },
          "& td:last-of-type, & th:last-of-type": { pr: 2 },
        }}
      >
        <TableHead>
          <TableRow>
            <TableCell>Site</TableCell>
            <TableCell>County</TableCell>
            <TableCell align="right">Peak risk</TableCell>
            <TableCell>Level</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {sites.map((site) => (
            <TableRow
              key={site.id}
              hover
              selected={site.id === selectedId}
              tabIndex={0}
              onClick={() => onSelect?.(site.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onSelect?.(site.id);
              }}
              sx={{ cursor: "pointer" }}
            >
              <TableCell>{site.name}</TableCell>
              <TableCell sx={{ color: "text.secondary" }}>{site.county}</TableCell>
              <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                {site.peak.toFixed(2)}
              </TableCell>
              <TableCell>
                <LevelChip level={site.level} />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  </Card>
);

export default SitesTable;
