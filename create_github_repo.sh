#!/bin/bash

# Fire Forecasting GitHub Repository Creation Script
# This script creates and pushes the fire-forecasting repository to GitHub

set -e

echo "🔥 Creating Fire Forecasting GitHub Repository"
echo "=============================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if gh is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ GitHub CLI is not authenticated."
    echo "Please run: gh auth login"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "config.yaml" ] || [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the fire-forecasting directory root"
    exit 1
fi

echo "✅ GitHub CLI is ready"
echo ""

# Initialize git repository
echo "📁 Initializing git repository..."
git init

# Add all files
echo "📝 Adding files to git..."
git add .

# Initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: fire-forecasting (Tri-County wildfire risk with MUI + ApexCharts)

🔥 Complete ML system for wildfire prediction
- RAWS + FIRMS + FRAP data integration
- ANN/LSTM models with CPU training <10min
- FastAPI backend with comprehensive metrics
- Next.js frontend with MUI + ApexCharts
- Leaflet maps with FRAP overlays
- 20+ WUI sites across Tri-County area"

# Create GitHub repository
echo "🚀 Creating GitHub repository..."
gh repo create fire-forecasting \
    --public \
    --description "ML-powered wildfire prediction system for Tri-County California" \
    --homepage "http://localhost:3000" \
    --source=. \
    --remote=origin \
    --push

echo ""
echo "🎉 Repository created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Backend setup:"
echo "   make venv"
echo "   make install"
echo "   make run-train"
echo "   make run-backend"
echo ""
echo "2. Frontend setup:"
echo "   cd frontend"
echo "   npm install"
echo "   npm run dev"
echo ""
echo "3. Access the system:"
echo "   Frontend: http://localhost:3000"
echo "   Backend: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🌐 Repository URL: https://github.com/$(gh api user --jq .login)/fire-forecasting"
echo ""
echo "Happy fire forecasting! 🔥📊"
