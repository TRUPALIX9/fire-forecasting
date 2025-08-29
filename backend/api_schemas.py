"""
Pydantic models for FastAPI request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class StatusResponse(BaseModel):
    """Health check and system status response."""
    status: str = Field(..., description="System status")
    timestamp: datetime = Field(..., description="Current timestamp")
    last_run: Optional[datetime] = Field(None, description="Last training run timestamp")
    rows_total: Optional[int] = Field(None, description="Total rows in dataset")
    model: Optional[str] = Field(None, description="Current model type")


class ConfigResponse(BaseModel):
    """Configuration response (sanitized)."""
    project: Dict[str, Any] = Field(..., description="Project configuration")
    data: Dict[str, Any] = Field(..., description="Data configuration")
    labeling: Dict[str, Any] = Field(..., description="Labeling configuration")
    features: Dict[str, Any] = Field(..., description="Feature configuration")
    train: Dict[str, Any] = Field(..., description="Training configuration")


class TrainingRequest(BaseModel):
    """Training request parameters."""
    config_override: Optional[Dict[str, Any]] = Field(None, description="Optional config overrides")


class TrainingResponse(BaseModel):
    """Training response with metrics summary."""
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Training message")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics if successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class GlobalMetrics(BaseModel):
    """Global model performance metrics."""
    pr_auc: float = Field(..., description="Precision-Recall AUC")
    roc_auc: float = Field(..., description="ROC AUC")
    precision: float = Field(..., description="Precision at threshold")
    recall: float = Field(..., description="Recall at threshold")
    f1_score: float = Field(..., description="F1 score at threshold")
    threshold: float = Field(..., description="Operating threshold")
    train_time_s: float = Field(..., description="Training time in seconds")
    infer_time_s: float = Field(..., description="Inference time in seconds")


class SiteMetrics(BaseModel):
    """Per-site performance metrics."""
    site: str = Field(..., description="Site name")
    pr_auc: float = Field(..., description="Site-specific PR-AUC")
    f1_at_tau: float = Field(..., description="F1 score at threshold")
    positives: int = Field(..., description="Number of positive samples")
    n: int = Field(..., description="Total number of samples")


class SitesMetricsResponse(BaseModel):
    """Response containing per-site metrics."""
    sites: List[SiteMetrics] = Field(..., description="List of site metrics")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class CurveData(BaseModel):
    """Precision-Recall or ROC curve data."""
    x_values: List[float] = Field(..., description="X-axis values (recall/fpr)")
    y_values: List[float] = Field(..., description="Y-axis values (precision/tpr)")
    thresholds: List[float] = Field(..., description="Threshold values")


class ConfusionMatrix(BaseModel):
    """Confusion matrix at a given threshold."""
    tn: int = Field(..., description="True negatives")
    fp: int = Field(..., description="False positives")
    fn: int = Field(..., description="False negatives")
    tp: int = Field(..., description="True positives")
    threshold: float = Field(..., description="Threshold used")
    accuracy: float = Field(..., description="Accuracy")
    precision: float = Field(..., description="Precision")
    recall: float = Field(..., description="Recall")
    f1_score: float = Field(..., description="F1 score")


class GeoPoint(BaseModel):
    """Geographic point with properties."""
    type: str = Field(default="Feature", description="GeoJSON feature type")
    geometry: Dict[str, Any] = Field(..., description="Point geometry")
    properties: Dict[str, Any] = Field(..., description="Point properties")


class GeoCollection(BaseModel):
    """GeoJSON FeatureCollection."""
    type: str = Field(default="FeatureCollection", description="GeoJSON collection type")
    features: List[GeoPoint] = Field(..., description="List of features")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
