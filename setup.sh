#!/bin/bash

# Violence Detection System - Setup Script
# This script helps set up the environment for the violence detection system

set -e  # Exit on error

echo "=========================================="
echo "Violence Detection System - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "🔍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Found Python $PYTHON_VERSION${NC}"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "env" ]; then
    python3 -m venv env
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source env/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p media/captured_db
mkdir -p media/face_db/sazib
mkdir -p media/face_db/shawon
mkdir -p media/face_db/rahat
mkdir -p media/results
mkdir -p models/dataset/violence_detection_run/weights
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Check for model files
echo "🔍 Checking for model files..."
MODEL_MISSING=0

if [ ! -f "models/yolo11n.pt" ]; then
    echo -e "${RED}❌ models/yolo11n.pt not found${NC}"
    MODEL_MISSING=1
fi

if [ ! -f "models/yolo11s.pt" ]; then
    echo -e "${RED}❌ models/yolo11s.pt not found${NC}"
    MODEL_MISSING=1
fi

if [ ! -f "models/dataset/violence_detection_run/weights/best.pt" ]; then
    echo -e "${RED}❌ models/dataset/violence_detection_run/weights/best.pt not found${NC}"
    MODEL_MISSING=1
fi

if [ $MODEL_MISSING -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Model files are missing!${NC}"
    echo "📥 Please download model files from Google Drive"
    echo "📖 See MODELS_README.md for download instructions"
    echo ""
else
    echo -e "${GREEN}✅ All required model files found${NC}"
fi
echo ""

# Run database migrations
echo "💾 Running database migrations..."
python manage.py makemigrations --noinput
python manage.py migrate --noinput
echo -e "${GREEN}✅ Database migrations completed${NC}"
echo ""

# Run model verification
echo "🔍 Running model verification..."
python verify_models.py
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""

if [ $MODEL_MISSING -eq 1 ]; then
    echo "1. 📥 Download model files from Google Drive (see MODELS_README.md)"
    echo "2. 📋 Run 'python verify_models.py' to verify installation"
    echo "3. 🚀 Start the server with: python manage.py runserver"
else
    echo "1. 🚀 Start the server with: python manage.py runserver"
    echo "2. 🌐 Open http://localhost:8000 in your browser"
fi
echo ""
echo "Optional:"
echo "  - Create superuser: python manage.py createsuperuser"
echo "  - Add face recognition images to media/face_db/"
echo ""
echo "For help, see README.md"
echo ""

