# 🔥 Fire Forecasting

A machine learning system for predicting wildfire risk in the Tri-County area of California (Ventura, Santa Barbara, and Los Angeles counties).

## 🎯 Mission

Create a complete ML system that predicts "Fire tomorrow?" for selected sites across the Tri-County area using three free datasets:
1. **RAWS** (Remote Automated Weather Stations) — daily weather features
2. **NASA FIRMS** — active fire points used to build next-day labels  
3. **CAL FIRE FRAP** — historical fire perimeters for contextual mapping/EDA

Train compact neural nets (ANN primary, LSTM optional) on daily data (≈ 5k–12k rows) so a full run finishes on CPU in ~10 minutes. Expose a FastAPI backend for metrics & retraining; ship a Next.js frontend using Material UI and ApexCharts for a clean dashboard; include a Leaflet map with FRAP overlays.

## 🏗️ Architecture

```
fire-forecasting/
├── backend/                 # FastAPI backend
├── src/                     # ML pipeline source code
│   ├── data_sources/       # RAWS, FIRMS, FRAP data fetching
│   ├── features/           # Feature engineering & FWI calculation
│   ├── labeling/           # Fire label creation
│   ├── splits/             # Chronological data splitting
│   ├── models/             # ANN, LSTM, baseline models
│   └── experiments/        # Pipeline orchestration
├── frontend/               # Next.js + MUI + ApexCharts dashboard
├── artifacts/              # Models, metrics, plots, GeoJSON
├── data/                   # Cached raw data downloads
└── config.yaml            # Configuration file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- GitHub CLI (`gh`) for repository creation

### 1. Backend/ML Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd fire-forecasting

# Create virtual environment
make venv
make install

# Run first training (downloads data, trains ANN, writes artifacts)
make run-train

# Start FastAPI backend
make run-backend
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Access the System
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📊 Features

### ML Pipeline
- **Data Sources**: RAWS weather, FIRMS fire detection, FRAP historical fires
- **Feature Engineering**: Lags (1-7 days), rolling stats, seasonal features, neighbor signals
- **Labeling**: FIRMS proximity (15km buffer) or FWI threshold methods
- **Models**: ANN (256→128→64), LSTM (optional), baseline comparisons
- **Metrics**: PR-AUC (primary), ROC-AUC, Precision, Recall, F1
- **Performance**: CPU training in ~10 minutes, <12k rows

### Frontend Dashboard
- **KPI Cards**: PR-AUC, ROC-AUC, Precision, Recall, F1
- **Charts**: ApexCharts PR/ROC curves, confusion matrix
- **Controls**: Threshold slider, retrain button, layer toggles
- **Map**: Leaflet integration with site markers and FRAP overlays
- **Tables**: Per-site performance metrics with MUI DataGrid

### Backend API
- **Training**: POST `/api/train` for end-to-end pipeline
- **Metrics**: GET `/api/metrics/global` and `/api/metrics/sites`
- **Curves**: GET `/api/curves/pr` and `/api/curves/roc`
- **Geo**: GET `/api/geo/sites` and `/api/geo/frap`
- **Status**: GET `/api/status` for system health

## 🔧 Configuration

Edit `config.yaml` to customize:
- **Sites**: 20+ WUI-focused locations across Tri-County
- **Features**: Lag days, rolling windows, neighbor signals
- **Training**: Model type, epochs, batch size, learning rate
- **Labeling**: Buffer radius, confidence thresholds, FWI quantiles

## 📈 Model Performance

The system is designed to achieve:
- **PR-AUC**: >0.7 (excellent), >0.5 (good)
- **Training Time**: <10 minutes on CPU
- **Data Size**: 5k-12k rows (configurable)
- **Sites**: 20+ monitoring locations
- **Coverage**: 8 years of historical data

## 🗺️ Geographic Coverage

**Santa Barbara County**
- Montecito, Hope Ranch, San Roque, Santa Barbara Mesa

**Ventura County**  
- Thousand Oaks, Newbury Park, Moorpark, Simi Valley, Ojai Valley, Ventura Hillsides

**Los Angeles County**
- Malibu, Santa Monica Mountains, Glendale/Pasadena foothills, Bel Air/Brentwood, Chatsworth, Eagle Rock, Palos Verdes, Antelope Valley

## 🔬 Technical Details

### ML Stack
- **Framework**: TensorFlow 2.16
- **Architecture**: ANN (256→128→64) with Dropout(0.2)
- **Optimization**: Adam optimizer, early stopping, learning rate reduction
- **Regularization**: Class weights, chronological splits, feature scaling

### Backend Stack  
- **API**: FastAPI + Uvicorn
- **Data**: Pandas, NumPy, scikit-learn
- **Geo**: Shapely, GeoPandas
- **ML**: TensorFlow, scikit-learn

### Frontend Stack
- **Framework**: Next.js 14 (App Router)
- **UI**: Material UI (MUI) + Emotion
- **Charts**: ApexCharts + react-apexcharts
- **Maps**: Leaflet + react-leaflet
- **Data**: SWR for API state management

## 📚 Data Sources

### RAWS Weather Data
- **Source**: NOAA Remote Automated Weather Stations
- **Features**: TMAX, TMIN, RH, WIND_SPD, WIND_DIR, PRCP
- **Frequency**: Daily aggregated from hourly
- **Coverage**: Nearest station to each site (<60km)

### NASA FIRMS
- **Source**: Fire Information for Resource Management System
- **Data**: MODIS + VIIRS satellite fire detection
- **Features**: Lat/lon, confidence, FRP, timestamp
- **Usage**: Next-day fire labels (15km buffer)

### CAL FIRE FRAP
- **Source**: Fire and Resource Assessment Program
- **Data**: Historical fire perimeters (2017-2024)
- **Features**: Geometry, name, year, acres, county
- **Usage**: Contextual mapping and EDA

## 🚨 Limitations & Considerations

- **Satellite Latency**: FIRMS data may have 1-2 day delays
- **Weather Coverage**: Some sites far from RAWS stations
- **Climate Change**: Historical patterns may not represent future conditions
- **Human Factors**: Doesn't account for arson, accidents, etc.
- **Data Quality**: Mock data used for demonstration

## 🔮 Future Improvements

- **Real-time Data**: Hourly RAWS updates, live FIRMS feeds
- **Additional Features**: Fuel moisture, NDVI, topography, elevation
- **Model Enhancements**: LSTM variants, ensemble methods, transfer learning
- **Deployment**: Docker containers, cloud deployment, CI/CD pipeline
- **Monitoring**: Model drift detection, automated retraining, alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: NOAA RAWS, NASA FIRMS, CAL FIRE FRAP
- **ML Libraries**: TensorFlow, scikit-learn, pandas, numpy
- **Web Stack**: FastAPI, Next.js, Material UI, ApexCharts
- **Research**: Wildfire prediction literature and methodologies

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the configuration in `config.yaml`

---

**⚠️ Disclaimer**: This system is for research and demonstration purposes. Do not use for actual fire prediction without proper validation and emergency service coordination.
