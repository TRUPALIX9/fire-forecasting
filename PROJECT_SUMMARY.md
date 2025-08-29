# 🔥 Fire Forecasting Project Summary

## 🎯 Project Overview
**Fire Forecasting** is a complete ML system that predicts "Fire tomorrow?" for selected sites across Ventura, Santa Barbara, and Los Angeles counties using three free datasets:
1. **RAWS** (Remote Automated Weather Stations) — daily weather features
2. **NASA FIRMS** — active fire points used to build next-day labels
3. **CAL FIRE FRAP** — historical fire perimeters for contextual mapping/EDA

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **ML Pipeline**: TensorFlow ANN/LSTM models with CPU training <10 minutes
- **Data Sources**: RAWS weather data, FIRMS fire detection, FRAP historical perimeters
- **API Endpoints**: 9 REST endpoints for metrics, training, and geospatial data
- **Static Serving**: `/artifacts/` directory for model downloads and visualizations

### Frontend (Next.js + Material UI + ApexCharts)
- **Dashboard**: KPI cards, PR/ROC curves, confusion matrix, threshold tuning
- **Map View**: Interactive Leaflet map with site markers and FRAP overlays
- **Sites Table**: Per-site performance metrics with sorting/filtering
- **Responsive Design**: Material UI components with consistent theming

## 📊 Machine Learning Pipeline

### Data Processing
- **20+ WUI Sites**: Montecito, Malibu, Thousand Oaks canyons, Ojai Valley, etc.
- **Feature Engineering**: Daily weather + 7-day lags + rolling stats + seasonality + neighbor features
- **Labeling**: FIRMS proximity (default) or FWI threshold (ablation study)
- **Data Size**: 6k-12k rows (hard cap), 5-8 years of daily data

### Models & Training
- **Baseline**: Logistic Regression, Random Forest
- **Primary**: ANN (256→128→64→1 with Dropout)
- **Optional**: LSTM with 14-day lookback
- **Metrics**: PR-AUC (primary), ROC-AUC, Precision, Recall, F1
- **Runtime**: ANN ~1-2 min, LSTM ~5-8 min on CPU

### Data Splitting
- **Chronological**: 70/15/15 by date across all sites
- **No Leakage**: Target is strictly t+1, scaler fit on train only
- **Class Imbalance**: Automatic class weight computation

## 🗂️ Repository Structure

```
fire-forecasting/
├── README.md                 # Comprehensive project documentation
├── config.yaml              # Pre-filled Tri-County configuration
├── requirements.txt         # Python dependencies
├── Makefile                # Development commands
├── create_github_repo.sh   # GitHub repository creation script
├── .env.example            # Environment variables template
├── data/                   # Cached raw downloads (gitignored)
├── artifacts/              # Models, metrics, figures, geo data
├── backend/                # FastAPI application
├── src/                    # ML pipeline source code
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions
│   ├── geometry.py        # Geospatial operations
│   ├── features/          # Feature engineering
│   ├── data_sources/      # Data fetching modules
│   ├── labeling/          # Label generation
│   ├── splits/            # Data splitting
│   ├── models/            # Model implementations
│   └── experiments/       # Pipeline orchestration
└── frontend/              # Next.js application
    ├── app/               # App Router pages
    ├── components/        # React components
    └── package.json      # Node.js dependencies
```

## 🚀 Quick Start

### 1. Backend Setup
```bash
make venv          # Create virtual environment
make install       # Install Python dependencies
make run-train     # Run end-to-end ML pipeline
make run-backend   # Start FastAPI server
```

### 2. Frontend Setup
```bash
cd frontend
npm install        # Install Node.js dependencies
npm run dev        # Start Next.js development server
```

### 3. Access the System
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🔧 Key Features

### Data Integration
- **RAWS**: Automatic station lookup, daily aggregation, missing value imputation
- **FIRMS**: Tri-County bounding box filtering, proximity-based labeling
- **FRAP**: Historical fire perimeter simplification and GeoJSON export

### Feature Engineering
- **Temporal**: 7-day lags, 7-day rolling statistics
- **Seasonal**: Day-of-year sine/cosine transformations
- **Spatial**: Inverse-distance weighted neighbor features
- **Weather**: TMAX, TMIN, RH, wind speed/direction, precipitation

### Model Performance
- **PR-AUC**: Primary metric for class imbalance
- **Threshold Tuning**: Interactive precision/recall optimization
- **Per-Site Analysis**: Individual performance metrics and sparklines
- **Confusion Matrix**: Real-time threshold adjustment visualization

### Geospatial Visualization
- **Interactive Map**: Leaflet with custom markers and popups
- **Layer Toggles**: Sites, FRAP historical fires
- **Site Details**: Performance metrics and fire event history

## 📈 Performance Metrics

### Global Performance
- **PR-AUC**: Precision-Recall Area Under Curve
- **ROC-AUC**: Receiver Operating Characteristic
- **F1 Score**: Harmonic mean of precision and recall
- **Training Time**: Model training duration
- **Inference Time**: Prediction latency

### Per-Site Analysis
- **Individual Metrics**: PR-AUC, F1 score per site
- **Fire Events**: Number of positive samples
- **Data Coverage**: Total samples per site
- **Performance Ranking**: Site-by-site comparison

## 🎨 User Interface

### Dashboard Components
- **KPI Cards**: Key performance indicators with color coding
- **Charts**: ApexCharts for PR/ROC curves
- **Threshold Slider**: Interactive precision/recall tuning
- **Confusion Matrix**: Real-time matrix updates
- **Action Buttons**: Retrain model, download artifacts

### Navigation
- **Dashboard** (`/`): Main overview and controls
- **Map** (`/map`): Interactive geospatial visualization
- **Sites** (`/sites`): Per-site performance table
- **About** (`/about`): Project documentation and methodology

## 🔬 Technical Implementation

### Backend Architecture
- **FastAPI**: Modern Python web framework with automatic API docs
- **Background Tasks**: Non-blocking model training
- **Static File Serving**: Artifact downloads and visualizations
- **Pydantic Models**: Request/response validation

### Frontend Architecture
- **Next.js 14**: App Router with server-side rendering
- **Material UI**: Consistent design system and components
- **ApexCharts**: Interactive charts and visualizations
- **React-Leaflet**: Interactive maps with custom styling
- **SWR**: Data fetching with caching and revalidation

### ML Pipeline
- **Modular Design**: Separate modules for data, features, models
- **Configuration-Driven**: YAML-based parameter management
- **Mock Data**: Development-friendly data generation
- **Artifact Management**: Organized output structure

## 🌍 Geographic Coverage

### Tri-County Area
- **Ventura County**: Thousand Oaks, Moorpark, Ojai Valley, Ventura Hillsides
- **Santa Barbara County**: Montecito, Hope Ranch, San Roque
- **Los Angeles County**: Malibu, Santa Monica Mountains, Glendale foothills

### WUI Focus
- **Wildland-Urban Interface**: High-risk areas with development near wildlands
- **Historical Fires**: Areas with documented fire history
- **Weather Stations**: RAWS coverage within 60km of sites

## 📊 Data Sources

### RAWS (Remote Automated Weather Stations)
- **Coverage**: National network of automated weather stations
- **Variables**: Temperature, humidity, wind, precipitation
- **Frequency**: Hourly data aggregated to daily
- **Quality**: Missing value imputation and validation

### NASA FIRMS (Fire Information for Resource Management System)
- **Satellites**: MODIS and VIIRS fire detection
- **Coverage**: Global active fire monitoring
- **Latency**: Near real-time detection
- **Confidence**: Filterable by detection confidence

### CAL FIRE FRAP (Fire and Resource Assessment Program)
- **Coverage**: California fire perimeter database
- **History**: Decades of fire event data
- **Format**: Vector polygons with metadata
- **Simplification**: Optimized for web visualization

## 🚧 Limitations & Considerations

### Data Quality
- **Satellite Latency**: FIRMS detection delays
- **Detection Noise**: False positives/negatives in fire detection
- **Station Representativeness**: Point measurements vs. area conditions
- **Climate Drift**: Long-term weather pattern changes

### Model Constraints
- **CPU Training**: Limited to 10-minute runtime
- **Data Size**: Maximum 12,000 rows
- **Feature Complexity**: Balance between performance and interpretability
- **Temporal Dependencies**: Daily resolution limitations

### Geographic Scope
- **Tri-County Focus**: Limited to Southern California
- **Site Selection**: WUI areas with RAWS coverage
- **Buffer Distances**: Fixed proximity thresholds
- **Neighbor Features**: Limited to nearby sites

## 🔮 Future Improvements

### Data Enhancement
- **Hourly RAWS**: Higher temporal resolution
- **Fuel Moisture**: Live fuel moisture content data
- **NDVI**: Vegetation health indicators
- **Topography**: Elevation and slope data

### Model Advancement
- **Ensemble Methods**: Multiple model combination
- **Deep Learning**: More sophisticated architectures
- **Transfer Learning**: Pre-trained models
- **Online Learning**: Continuous model updates

### System Features
- **Real-time Updates**: Live data integration
- **Alert System**: Automated fire risk notifications
- **Mobile App**: Native mobile application
- **API Expansion**: Additional endpoints and integrations

## 🛠️ Development Tools

### Python Environment
- **Version**: Python 3.10+
- **Package Manager**: pip with requirements.txt
- **Virtual Environment**: venv with Makefile automation
- **Dependencies**: TensorFlow, scikit-learn, pandas, FastAPI

### Node.js Environment
- **Version**: Node.js 18+
- **Package Manager**: npm with package.json
- **Framework**: Next.js 14 with App Router
- **UI Library**: Material UI with ApexCharts

### Development Commands
```bash
make venv          # Create virtual environment
make install       # Install dependencies
make run-train     # Run ML pipeline
make run-backend   # Start backend server
make learning-curve # Generate learning curves
make threshold     # Threshold optimization
```

## 📚 Documentation

### Code Documentation
- **Inline Comments**: Comprehensive code explanations
- **Function Docstrings**: Parameter and return value documentation
- **Type Hints**: Python type annotations
- **README**: Project overview and setup instructions

### API Documentation
- **FastAPI Auto-Docs**: Interactive API documentation
- **Endpoint Descriptions**: Clear parameter and response documentation
- **Example Requests**: Sample API calls
- **Error Handling**: Comprehensive error responses

### User Guides
- **Setup Instructions**: Step-by-step installation
- **Usage Examples**: Common workflows and use cases
- **Troubleshooting**: Common issues and solutions
- **Configuration**: Parameter tuning and customization

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Code Standards
- **Python**: PEP 8 style guide
- **JavaScript**: ESLint configuration
- **TypeScript**: Strict type checking
- **Testing**: Unit tests for critical functions

### Review Process
- **Code Review**: All changes require review
- **Testing**: Automated testing on pull requests
- **Documentation**: Updated documentation for new features
- **Performance**: ML pipeline timing validation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RAWS**: NOAA for weather station data
- **FIRMS**: NASA for satellite fire detection
- **FRAP**: CAL FIRE for historical fire data
- **Open Source**: TensorFlow, FastAPI, Next.js, Material UI communities

---

**Fire Forecasting** - Predicting tomorrow's fires with today's data 🔥📊
