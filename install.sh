#!/bin/bash

# BulkRNA Agent - Installation & Setup Script

echo "🧬 BulkRNA Agent - Installation Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check Python
echo "1️⃣  Checking Python..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Step 2: Check Ollama
echo ""
echo "2️⃣  Checking Ollama..."
if command_exists ollama; then
    echo -e "${GREEN}✓${NC} Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama is running"
    else
        echo -e "${YELLOW}⚠${NC}  Ollama is not running"
        echo "   Starting Ollama in background..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
else
    echo -e "${RED}✗${NC} Ollama not found"
    echo "   Please install from: https://ollama.ai"
    exit 1
fi

# Step 3: Check Ollama models
echo ""
echo "3️⃣  Checking Ollama models..."

if ollama list | grep -q "gpt-oss:20b"; then
    echo -e "${GREEN}✓${NC} gpt-oss:20b found"
else
    echo -e "${YELLOW}⚠${NC}  gpt-oss:20b not found"
    read -p "   Pull gpt-oss:20b? This may take a while. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull gpt-oss:20b
    fi
fi

if ollama list | grep -q "cniongolo/biomistral"; then
    echo -e "${GREEN}✓${NC} cniongolo/biomistral found"
else
    echo -e "${YELLOW}⚠${NC}  cniongolo/biomistral not found"
    read -p "   Pull cniongolo/biomistral? This may take a while. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull cniongolo/biomistral
    fi
fi

# Step 4: Setup Python environment
echo ""
echo "4️⃣  Setting up Python environment..."

# Check if conda is available
if command_exists conda; then
    echo -e "${GREEN}✓${NC} Conda found"
    
    # Check if environment exists
    if conda env list | grep -q "bulkrna-agent"; then
        echo -e "${GREEN}✓${NC} Conda environment 'bulkrna-agent' exists"
    else
        echo "   Creating conda environment..."
        conda env create -f environment.yml
        echo -e "${GREEN}✓${NC} Conda environment created"
    fi
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate bulkrna-agent
    
else
    echo -e "${YELLOW}⚠${NC}  Conda not found, using venv instead"
    
    if [ -d "venv" ]; then
        echo -e "${GREEN}✓${NC} Virtual environment exists"
    else
        echo "   Creating virtual environment..."
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies with pip
    echo ""
    echo "5️⃣  Installing Python dependencies..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Dependencies installed"
    else
        echo -e "${RED}✗${NC} Failed to install dependencies"
        exit 1
    fi
fi

# Install package in development mode
echo "   Installing BulkRNA Agent..."
pip install -e .

# Step 6: Create directories
echo ""
echo "6️⃣  Creating directories..."
mkdir -p data/uploads data/outputs data/cache data/examples logs
echo -e "${GREEN}✓${NC} Directories created"

# Step 7: Check R (optional)
echo ""
echo "7️⃣  Checking R installation (optional)..."
if command_exists Rscript; then
    R_VERSION=$(Rscript -e "cat(paste0(R.version\$major, '.', R.version\$minor))" 2>/dev/null)
    echo -e "${GREEN}✓${NC} R found: $R_VERSION"
    
    # Check DESeq2
    if Rscript -e "library(DESeq2)" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} DESeq2 is installed"
    else
        echo -e "${YELLOW}⚠${NC}  DESeq2 not found"
        echo "   You can install it with:"
        echo "   Rscript -e \"install.packages('BiocManager'); BiocManager::install('DESeq2')\""
    fi
else
    echo -e "${YELLOW}⚠${NC}  R not installed (optional)"
    echo "   PyDESeq2 (Python) will be used instead"
fi

# Step 8: Generate example data
echo ""
echo "8️⃣  Generating example data..."
python examples/generate_example_data.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Example data generated"
else
    echo -e "${YELLOW}⚠${NC}  Could not generate example data"
fi

# Step 9: Run tests (optional)
echo ""
echo "9️⃣  Running tests (optional)..."
read -p "   Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/ -v
fi

# Summary
echo ""
echo "======================================"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "======================================"
echo ""
echo "To start BulkRNA Agent:"
echo "  ./start.sh"
echo ""
echo "Or manually:"
if command_exists conda && conda env list | grep -q "bulkrna-agent"; then
    echo "  conda activate bulkrna-agent"
else
    echo "  source venv/bin/activate"
fi
echo "  python run_agent.py"
echo ""
echo "Then open: http://localhost:7860"
echo ""
echo "Documentation:"
echo "  - README.md - Main documentation"
echo "  - docs/TUTORIAL.md - Step-by-step guide"
echo "  - QUICKREF.md - Quick reference"
echo ""
echo "Example data: data/examples/"
echo "Logs: logs/bulkrna_agent.log"
echo ""
echo "🧬 Happy analyzing! 🚀"
