#!/usr/bin/env python3
"""
Fire Forecasting Backend API
FastAPI server for serving ML pipeline results and triggering retraining
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import json
import pandas as pd
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fire Forecasting API",
    description="API for wildfire prediction models and metrics",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_project_root() -> Path:
    """Get the project root directory (parent of backend/)"""
    current_file = Path(__file__)
    return current_file.parent.parent

PROJECT_ROOT = get_project_root()

# Global state for training status
training_status = {"is_training": False, "last_completed": None, "error": None}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fire Forecasting API", "status": "running"}

@app.get("/api/status")
async def get_status():
    """Get API and training status"""
    return {
        "api_status": "running",
        "training_status": training_status,
        "artifacts_exist": {
            "global_metrics": (PROJECT_ROOT / "artifacts/metrics/global_metrics.json").exists(),
            "per_site_metrics": (PROJECT_ROOT / "artifacts/metrics/per_site_metrics.csv").exists(),
            "figures": len(list((PROJECT_ROOT / "artifacts/figures").glob("*.png"))) if (PROJECT_ROOT / "artifacts/figures").exists() else 0
        }
    }

@app.get("/api/data/firms")
async def get_firms_data():
    """Get FIRMS data status and count"""
    # Check for the consolidated FIRMS file first
    firms_path = PROJECT_ROOT / "data/firms/firms_consolidated.csv"
    if not firms_path.exists():
        # Fallback to the original file
        firms_path = PROJECT_ROOT / "data/firms/firms_20170101_20241231.csv"
        if not firms_path.exists():
            raise HTTPException(status_code=404, detail="FIRMS data not found")
    
    try:
        import pandas as pd
        df = pd.read_csv(firms_path)
        return {
            "count": len(df),
            "columns": list(df.columns),
            "date_range": {
                "min": df.get("acq_date", pd.Series()).min() if "acq_date" in df.columns else None,
                "max": df.get("acq_date", pd.Series()).max() if "acq_date" in df.columns else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading FIRMS data: {str(e)}")

@app.get("/api/data/raws")
async def get_raws_data():
    """Get RAWS data status and count"""
    raws_path = PROJECT_ROOT / "data/raws/SANTA BARBARA_20170101_20241231.csv"
    if not raws_path.exists():
        raise HTTPException(status_code=404, detail="RAWS data not found")
    
    try:
        import pandas as pd
        df = pd.read_csv(raws_path)
        return {
            "count": len(df),
            "columns": list(df.columns),
            "date_range": {
                "min": df.get("date", pd.Series()).min() if "date" in df.columns else None,
                "max": df.get("date", pd.Series()).max() if "date" in df.columns else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading RAWS data: {str(e)}")

@app.get("/api/models/status")
async def get_model_status():
    """Get model training status and artifacts"""
    model_path = PROJECT_ROOT / "artifacts/models/model.keras"
    metrics_path = PROJECT_ROOT / "artifacts/metrics/global_metrics.json"
    
    if not model_path.exists():
        return {
            "trained": False,
            "model_exists": False,
            "message": "Model not found - run training pipeline"
        }
    
    if not metrics_path.exists():
        return {
            "trained": False,
            "model_exists": True,
            "message": "Model exists but no metrics found - training may be incomplete"
        }
    
    try:
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return {
            "trained": True,
            "model_exists": True,
            "training_date": metrics.get("training_date", "Unknown"),
            "metrics": metrics,
            "message": "Model is fully trained and ready"
        }
    except Exception as e:
        return {
            "trained": False,
            "model_exists": True,
            "message": f"Model exists but error reading metrics: {str(e)}"
        }

def run_training_pipeline():
    """Run the training pipeline in background"""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["error"] = None
        
        logger.info("Starting training pipeline...")
        
        # Run the pipeline
        result = subprocess.run([
            sys.executable, "-m", "src.experiments.run_pipeline",
            "--config", "config.yaml"
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode == 0:
            training_status["last_completed"] = "success"
            logger.info("Training completed successfully")
        else:
            training_status["error"] = result.stderr
            logger.error(f"Training failed: {result.stderr}")
            
    except Exception as e:
        training_status["error"] = str(e)
        logger.error(f"Training error: {e}")
    finally:
        training_status["is_training"] = False

@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    background_tasks.add_task(run_training_pipeline)
    return {"message": "Retraining started", "status": "queued"}

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status"""
    return training_status

@app.get("/api/metrics/global")
async def get_global_metrics():
    """Get global model metrics"""
    metrics_path = PROJECT_ROOT / "artifacts/metrics/global_metrics.json"
    if not metrics_path.exists():
        # Return a mock response if metrics don't exist
        return {
            "status": "not_trained",
            "message": "Model not trained yet. Run training pipeline first.",
            "mock_data": True
        }
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Error reading global metrics: {e}")
        raise HTTPException(status_code=500, detail="Error reading metrics")

@app.get("/api/metrics/per-site")
async def get_per_site_metrics():
    """Get per-site metrics"""
    metrics_path = PROJECT_ROOT / "artifacts/metrics/per_site_metrics.csv"
    if not metrics_path.exists():
        # Return a mock response if metrics don't exist
        return {
            "status": "not_trained",
            "message": "Per-site metrics not available. Run training pipeline first.",
            "mock_data": True,
            "data": []
        }
    
    try:
        df = pd.read_csv(metrics_path)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading per-site metrics: {e}")
        raise HTTPException(status_code=500, detail="Error reading metrics")

@app.get("/api/confusion")
async def get_confusion_matrix(threshold: float = 0.5):
    """Get confusion matrix for given threshold"""
    # Return mock confusion matrix data
    return {
        "threshold": threshold,
        "confusion_matrix": {
            "true_negatives": 1500,
            "false_positives": 100,
            "false_negatives": 50,
            "true_positives": 200
        },
        "precision": 0.67,
        "recall": 0.80,
        "f1_score": 0.73,
        "mock_data": True
    }

@app.get("/api/curves/pr")
async def get_pr_curve():
    """Get Precision-Recall curve data"""
    # Return mock PR curve data
    return {
        "precision": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
        "recall": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "thresholds": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        "auc": 0.78,
        "mock_data": True
    }

@app.get("/api/curves/roc")
async def get_roc_curve():
    """Get ROC curve data"""
    # Return mock ROC curve data
    return {
        "fpr": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "tpr": [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
        "thresholds": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        "auc": 0.85,
        "mock_data": True
    }

@app.get("/api/figures/{figure_name}")
async def get_figure(figure_name: str):
    """Get a specific figure by name"""
    figure_path = PROJECT_ROOT / f"artifacts/figures/{figure_name}"
    if not figure_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure {figure_name} not found")
    
    return FileResponse(figure_path)

@app.get("/api/figures")
async def list_figures():
    """List all available figures"""
    figures_dir = PROJECT_ROOT / "artifacts/figures"
    if not figures_dir.exists():
        return {"figures": []}
    
    figures = [f.name for f in figures_dir.glob("*.png")]
    return {"figures": figures}

@app.get("/api/geo/frap")
async def get_frap_geojson():
    """Get FRAP fire perimeter data"""
    geo_path = PROJECT_ROOT / "artifacts/geo/frap_fire_perimeters.geojson"
    if not geo_path.exists():
        raise HTTPException(status_code=404, detail="FRAP GeoJSON not found — run `make fetch-frap`")
    
    return FileResponse(geo_path, media_type="application/geo+json")

@app.get("/api/geo/sites")
async def get_sites_geojson():
    """Get sites geojson data"""
    # Check multiple possible locations for sites data
    possible_paths = [
        PROJECT_ROOT / "artifacts/geo/sites.geojson",
        PROJECT_ROOT / "artifacts/geo/monitoring_sites.geojson",
        PROJECT_ROOT / "data/sites.geojson"
    ]
    
    for geo_path in possible_paths:
        if geo_path.exists():
            return FileResponse(geo_path, media_type="application/geo+json")
    
    # If no sites file exists, return a mock response with basic site data
    mock_sites = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-119.7, 34.4]},
                "properties": {"site": "Montecito", "lat": 34.4, "lon": -119.7}
            },
            {
                "type": "Feature", 
                "geometry": {"type": "Point", "coordinates": [-118.8, 34.1]},
                "properties": {"site": "Malibu", "lat": 34.1, "lon": -118.8}
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-118.9, 34.2]},
                "properties": {"site": "Thousand Oaks", "lat": 34.2, "lon": -118.9}
            }
        ]
    }
    
    return mock_sites

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
