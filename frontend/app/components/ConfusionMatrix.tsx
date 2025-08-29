'use client'

import { Grid, Paper, Typography, Box } from '@mui/material'

interface ConfusionMatrixData {
  tn: number
  fp: number
  fn: number
  tp: number
  threshold: number
}

interface ConfusionMatrixProps {
  data: ConfusionMatrixData
}

export default function ConfusionMatrix({ data }: ConfusionMatrixProps) {
  const { tn, fp, fn, tp, threshold } = data
  
  // Calculate metrics
  const total = tn + fp + fn + tp
  const accuracy = total > 0 ? (tp + tn) / total : 0
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0
  const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0

  return (
    <Box>
      <Grid container spacing={2}>
        {/* Matrix */}
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom align="center">
            Confusion Matrix
          </Typography>
          <Grid container spacing={1} justifyContent="center">
            {/* Header row */}
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Predicted</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">No Fire (0)</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Fire (1)</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Total</Typography>
              </Paper>
            </Grid>
            
            {/* Actual No Fire row */}
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Actual</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                <Typography variant="h6" color="success.contrastText">
                  {tn}
                </Typography>
                <Typography variant="caption" color="success.contrastText">
                  TN
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light' }}>
                <Typography variant="h6" color="error.contrastText">
                  {fp}
                </Typography>
                <Typography variant="caption" color="error.contrastText">
                  FP
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.200' }}>
                <Typography variant="h6">{tn + fp}</Typography>
              </Paper>
            </Grid>
            
            {/* Actual Fire row */}
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Fire</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light' }}>
                <Typography variant="h6" color="error.contrastText">
                  {fn}
                </Typography>
                <Typography variant="caption" color="error.contrastText">
                  FN
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                <Typography variant="h6" color="success.contrastText">
                  {tp}
                </Typography>
                <Typography variant="caption" color="success.contrastText">
                  TP
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.200' }}>
                <Typography variant="h6">{fn + tp}</Typography>
              </Paper>
            </Grid>
            
            {/* Total row */}
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.100' }}>
                <Typography variant="caption">Total</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.200' }}>
                <Typography variant="h6">{tn + fn}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'grey.200' }}>
                <Typography variant="h6">{fp + tp}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={3}>
              <Paper sx={{ p: 1, textAlign: 'center', bgcolor: 'primary.light' }}>
                <Typography variant="h6" color="primary.contrastText">
                  {total}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
        
        {/* Metrics */}
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Metrics Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary.main">
                  {accuracy.toFixed(3)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Accuracy
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary.main">
                  {precision.toFixed(3)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Precision
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary.main">
                  {recall.toFixed(3)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Recall
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={6}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h4" color="primary.main">
                  {f1.toFixed(3)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  F1 Score
                </Typography>
              </Paper>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              <strong>TN (True Negative):</strong> Correctly predicted no fire
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>FP (False Positive):</strong> Incorrectly predicted fire
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>FN (False Negative):</strong> Missed fire prediction
            </Typography>
            <Typography variant="body2" color="text.secondary">
              <strong>TP (True Positive):</strong> Correctly predicted fire
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  )
}
