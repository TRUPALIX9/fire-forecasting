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
            "global_metrics": Path("artifacts/metrics/global_metrics.json").exists(),
            "per_site_metrics": Path("artifacts/metrics/per_site_metrics.csv").exists(),
            "figures": len(list(Path("artifacts/figures").glob("*.png"))) if Path("artifacts/figures").exists() else 0
        }
    }

@app.get("/api/metrics/global")
async def get_global_metrics():
    """Get global model metrics"""
    metrics_path = Path("artifacts/metrics/global_metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Global metrics not found. Run training first.")
    
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
    metrics_path = Path("artifacts/metrics/per_site_metrics.csv")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Per-site metrics not found. Run training first.")
    
    try:
        df = pd.read_csv(metrics_path)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading per-site metrics: {e}")
        raise HTTPException(status_code=500, detail="Error reading metrics")

@app.get("/api/figures/{figure_name}")
async def get_figure(figure_name: str):
    """Get a specific figure by name"""
    figure_path = Path(f"artifacts/figures/{figure_name}")
    if not figure_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure {figure_name} not found")
    
    return FileResponse(figure_path)

@app.get("/api/figures")
async def list_figures():
    """List all available figures"""
    figures_dir = Path("artifacts/figures")
    if not figures_dir.exists():
        return {"figures": []}
    
    figures = [f.name for f in figures_dir.glob("*.png")]
    return {"figures": figures}

@app.get("/api/geo/frap")
async def get_frap_geojson():
    """Get FRAP fire perimeter data"""
    geo_path = Path("artifacts/geo/frap_fire_perimeters.geojson")
    if not geo_path.exists():
        raise HTTPException(status_code=404, detail="FRAP data not found. Run training first.")
    
    return FileResponse(geo_path)

@app.get("/api/geo/sites")
async def get_sites_geojson():
    """Get sites geojson data"""
    geo_path = Path("artifacts/geo/sites.geojson")
    if not geo_path.exists():
        raise HTTPException(status_code=404, detail="Sites data not found. Run training first.")
    
    return FileResponse(geo_path)

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
        ], capture_output=True, text=True, cwd=Path.cwd())
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
