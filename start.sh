#!/bin/bash

# Quick start script for BulkRNA Agent

echo "🧬 BulkRNA Agent - Quick Start"
echo "==============================="
echo ""

# Check if Ollama is running
echo "1️⃣ Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Ollama is running"
else
    echo "   ❌ Ollama is not running"
    echo "   Please start Ollama: ollama serve"
    exit 1
fi

# Check if models are available
echo ""
echo "2️⃣ Checking models..."
if ollama list | grep -q "gpt-oss:20b"; then
    echo "   ✅ gpt-oss:20b is available"
else
    echo "   ⚠️  gpt-oss:20b not found"
    echo "   Pulling model... (this may take a while)"
    ollama pull gpt-oss:20b
fi

if ollama list | grep -q "cniongolo/biomistral"; then
    echo "   ✅ cniongolo/biomistral is available"
else
    echo "   ⚠️  cniongolo/biomistral not found"
    echo "   Pulling model... (this may take a while)"
    ollama pull cniongolo/biomistral
fi

# Check Python environment
echo ""
echo "3️⃣ Checking Python environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    # Check if conda environment exists
    if conda env list | grep -q "bulkrna-agent"; then
        echo "   ✅ Conda environment 'bulkrna-agent' found"
        eval "$(conda shell.bash hook)"
        conda activate bulkrna-agent
    else
        echo "   ⚠️  Conda environment not found"
        echo "   Run ./install.sh first to create the environment"
        exit 1
    fi
elif [ -d "venv" ]; then
    echo "   ✅ Virtual environment found (venv)"
    source venv/bin/activate
else
    echo "   ⚠️  No Python environment found"
    echo "   Run ./install.sh first to set up the environment"
    exit 1
fi

# Create necessary directories
echo ""
echo "4️⃣ Creating directories..."
mkdir -p data/uploads data/outputs data/cache logs
echo "   ✅ Directories created"

# Check R (optional)
echo ""
echo "5️⃣ Checking R installation (optional)..."
if command -v Rscript &> /dev/null; then
    echo "   ✅ R is installed"
    echo "   Checking DESeq2..."
    if Rscript -e "library(DESeq2)" 2>/dev/null; then
        echo "   ✅ DESeq2 is installed"
    else
        echo "   ⚠️  DESeq2 not found"
        echo "   You can install it with:"
        echo "   Rscript -e \"install.packages('BiocManager'); BiocManager::install('DESeq2')\""
    fi
else
    echo "   ⚠️  R not installed (optional for MCP server)"
    echo "   PyDESeq2 will be used instead"
fi

# Launch agent
echo ""
echo "6️⃣ Launching BulkRNA Agent..."
echo ""
echo "🚀 Starting web interface at http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo ""

python run_agent.py
