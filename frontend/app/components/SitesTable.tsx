'use client';

import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import {
  DataGrid,
  GridColDef,
  GridValueGetterParams,
  GridRenderCellParams,
  GridToolbar,
  GridPaginationModel,
  GridSortModel,
} from '@mui/x-data-grid';
import {
  LocationOn,
  TrendingUp,
  Warning,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import { SiteMetrics } from '../api-client';

interface SitesTableProps {
  sites: SiteMetrics[];
  loading?: boolean;
  onSiteSelect?: (site: string) => void;
}

interface PerformanceStats {
  totalSites: number;
  avgPrAuc: number;
  avgF1Score: number;
  totalFireEvents: number;
  totalSamples: number;
  highPerformers: number;
  lowPerformers: number;
}

const SitesTable: React.FC<SitesTableProps> = ({
  sites,
  loading = false,
  onSiteSelect,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [performanceFilter, setPerformanceFilter] = useState<string>('all');
  const [paginationModel, setPaginationModel] = useState<GridPaginationModel>({
    page: 0,
    pageSize: 25,
  });
  const [sortModel, setSortModel] = useState<GridSortModel>([
    { field: 'pr_auc', sort: 'desc' },
  ]);

  // Calculate performance statistics
  const stats: PerformanceStats = useMemo(() => {
    if (!sites.length) {
      return {
        totalSites: 0,
        avgPrAuc: 0,
        avgF1Score: 0,
        totalFireEvents: 0,
        totalSamples: 0,
        highPerformers: 0,
        lowPerformers: 0,
      };
    }

    const avgPrAuc = sites.reduce((sum, site) => sum + site.pr_auc, 0) / sites.length;
    const avgF1Score = sites.reduce((sum, site) => sum + site.f1_at_tau, 0) / sites.length;
    const totalFireEvents = sites.reduce((sum, site) => sum + site.positives, 0);
    const totalSamples = sites.reduce((sum, site) => sum + site.n, 0);
    const highPerformers = sites.filter(site => site.pr_auc >= 0.7).length;
    const lowPerformers = sites.filter(site => site.pr_auc < 0.5).length;

    return {
      totalSites: sites.length,
      avgPrAuc,
      avgF1Score,
      totalFireEvents,
      totalSamples,
      highPerformers,
      lowPerformers,
    };
  }, [sites]);

  // Filter and sort sites
  const filteredSites = useMemo(() => {
    let filtered = sites.filter(site =>
      site.site.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Apply performance filter
    switch (performanceFilter) {
      case 'high':
        filtered = filtered.filter(site => site.pr_auc >= 0.7);
        break;
      case 'medium':
        filtered = filtered.filter(site => site.pr_auc >= 0.5 && site.pr_auc < 0.7);
        break;
      case 'low':
        filtered = filtered.filter(site => site.pr_auc < 0.5);
        break;
      case 'fire-prone':
        filtered = filtered.filter(site => site.positives >= 10);
        break;
      case 'data-rich':
        filtered = filtered.filter(site => site.n >= 1000);
        break;
    }

    return filtered;
  }, [sites, searchTerm, performanceFilter]);

  // Performance indicator component
  const PerformanceIndicator: React.FC<{ value: number; type: 'pr_auc' | 'f1' }> = ({
    value,
    type,
  }) => {
    const getColor = () => {
      if (type === 'pr_auc') {
        if (value >= 0.8) return 'success';
        if (value >= 0.6) return 'warning';
        return 'error';
      } else {
        if (value >= 0.7) return 'success';
        if (value >= 0.5) return 'warning';
        return 'error';
      }
    };

    const getIcon = () => {
      if (value >= 0.7) return <CheckCircle fontSize="small" />;
      if (value >= 0.5) return <Warning fontSize="small" />;
      return <Error fontSize="small" />;
    };

    return (
      <Box display="flex" alignItems="center" gap={1}>
        {getIcon()}
        <Typography
          variant="body2"
          color={`${getColor()}.main`}
          sx={{ fontWeight: 'bold' }}
        >
          {value.toFixed(3)}
        </Typography>
      </Box>
    );
  };

  // Data grid columns
  const columns: GridColDef[] = [
    {
      field: 'site',
      headerName: 'Site Name',
      width: 250,
      renderCell: (params: GridRenderCellParams) => (
        <Box display="flex" alignItems="center" gap={1}>
          <LocationOn color="primary" fontSize="small" />
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {params.value}
          </Typography>
        </Box>
      ),
    },
    {
      field: 'pr_auc',
      headerName: 'PR-AUC',
      width: 150,
      renderCell: (params: GridRenderCellParams) => (
        <PerformanceIndicator value={params.value} type="pr_auc" />
      ),
    },
    {
      field: 'f1_at_tau',
      headerName: 'F1 Score',
      width: 150,
      renderCell: (params: GridRenderCellParams) => (
        <PerformanceIndicator value={params.value} type="f1" />
      ),
    },
    {
      field: 'positives',
      headerName: 'Fire Events',
      width: 120,
      renderCell: (params: GridRenderCellParams) => (
        <Box display="flex" alignItems="center" gap={1}>
          <Warning color="error" fontSize="small" />
          <Typography variant="body2">{params.value}</Typography>
        </Box>
      ),
    },
    {
      field: 'n',
      headerName: 'Total Samples',
      width: 140,
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="body2">{params.value.toLocaleString()}</Typography>
      ),
    },
    {
      field: 'fire_rate',
      headerName: 'Fire Rate',
      width: 120,
      valueGetter: (params: GridValueGetterParams) => {
        const site = params.row;
        return site.positives / site.n;
      },
      renderCell: (params: GridRenderCellParams) => {
        const rate = params.value;
        const percentage = (rate * 100).toFixed(1);
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Box sx={{ width: '100%', mr: 1 }}>
              <LinearProgress
                variant="determinate"
                value={Math.min(rate * 100, 100)}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'grey.200',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: rate > 0.05 ? 'error.main' : 'warning.main',
                  },
                }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              {percentage}%
            </Typography>
          </Box>
        );
      },
    },
    {
      field: 'performance_category',
      headerName: 'Performance',
      width: 140,
      valueGetter: (params: GridValueGetterParams) => {
        const prAuc = params.row.pr_auc;
        if (prAuc >= 0.8) return 'Excellent';
        if (prAuc >= 0.7) return 'Good';
        if (prAuc >= 0.6) return 'Fair';
        if (prAuc >= 0.5) return 'Poor';
        return 'Very Poor';
      },
      renderCell: (params: GridRenderCellParams) => {
        const category = params.value;
        const getColor = () => {
          switch (category) {
            case 'Excellent': return 'success';
            case 'Good': return 'success';
            case 'Fair': return 'warning';
            case 'Poor': return 'error';
            case 'Very Poor': return 'error';
            default: return 'default';
          }
        };

        return (
          <Chip
            label={category}
            color={getColor() as any}
            size="small"
            variant="outlined"
          />
        );
      },
    },
  ];

  // Handle row click
  const handleRowClick = (params: any) => {
    if (onSiteSelect) {
      onSiteSelect(params.row.site);
    }
  };

  return (
    <Box>
      {/* Summary Statistics */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Total Sites
              </Typography>
              <Typography variant="h4">{stats.totalSites}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="success.main" gutterBottom>
                Avg PR-AUC
              </Typography>
              <Typography variant="h4">{stats.avgPrAuc.toFixed(3)}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="warning.main" gutterBottom>
                Fire Events
              </Typography>
              <Typography variant="h4">{stats.totalFireEvents}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="info.main" gutterBottom>
                High Performers
              </Typography>
              <Typography variant="h4">{stats.highPerformers}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              label="Search Sites"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              size="small"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Performance Filter</InputLabel>
              <Select
                value={performanceFilter}
                label="Performance Filter"
                onChange={(e) => setPerformanceFilter(e.target.value)}
              >
                <MenuItem value="all">All Sites</MenuItem>
                <MenuItem value="high">High Performance (PR-AUC ≥ 0.7)</MenuItem>
                <MenuItem value="medium">Medium Performance (0.5 ≤ PR-AUC < 0.7)</MenuItem>
                <MenuItem value="low">Low Performance (PR-AUC < 0.5)</MenuItem>
                <MenuItem value="fire-prone">Fire-Prone (≥10 events)</MenuItem>
                <MenuItem value="data-rich">Data-Rich (≥1000 samples)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="body2" color="text.secondary">
              Showing {filteredSites.length} of {sites.length} sites
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Performance Legend */}
      <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Performance Legend:
        </Typography>
        <Box display="flex" gap={2} flexWrap="wrap">
          <Box display="flex" alignItems="center" gap={1}>
            <CheckCircle color="success" fontSize="small" />
            <Typography variant="caption">Excellent (≥0.8)</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <CheckCircle color="success" fontSize="small" />
            <Typography variant="caption">Good (≥0.7)</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Warning color="warning" fontSize="small" />
            <Typography variant="caption">Fair (≥0.6)</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Error color="error" fontSize="small" />
            <Typography variant="caption">Poor (<0.6)</Typography>
          </Box>
        </Box>
      </Box>

      {/* Data Grid */}
      <Paper elevation={2} sx={{ height: 600, width: '100%' }}>
        <DataGrid
          rows={filteredSites}
          columns={columns}
          paginationModel={paginationModel}
          onPaginationModelChange={setPaginationModel}
          sortModel={sortModel}
          onSortModelChange={setSortModel}
          onRowClick={handleRowClick}
          loading={loading}
          pageSizeOptions={[10, 25, 50, 100]}
          disableRowSelectionOnClick
          slots={{
            toolbar: GridToolbar,
          }}
          slotProps={{
            toolbar: {
              showQuickFilter: true,
              quickFilterProps: { debounceMs: 500 },
            },
          }}
          sx={{
            '& .MuiDataGrid-row:hover': {
              backgroundColor: 'action.hover',
              cursor: 'pointer',
            },
            '& .MuiDataGrid-cell:focus': {
              outline: 'none',
            },
          }}
        />
      </Paper>
    </Box>
  );
};

export default SitesTable;
