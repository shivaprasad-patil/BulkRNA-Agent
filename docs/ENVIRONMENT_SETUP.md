# Environment Setup Guide

## Why Use a Separate Environment?

✅ **Isolation**: Prevents conflicts with other Python projects  
✅ **Reproducibility**: Ensures consistent package versions  
✅ **R Integration**: Conda can manage both Python and R packages  
✅ **Easy Cleanup**: Can delete the environment without affecting your system  

## Option 1: Conda (Recommended) 🌟

**Best for this project** - handles Python, R, and Bioconductor packages.

### Install Conda
If you don't have conda:
```bash
# Install Miniforge (lightweight, conda-forge default)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```

### Create Environment
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate
conda activate bulkrna-agent

# Install BulkRNA Agent
pip install -e .

# Verify installation
python -c "import bulkrna_agent; print('✓ BulkRNA Agent installed')"
```

### Usage
```bash
# Always activate before using
conda activate bulkrna-agent

# Run the agent
python run_agent.py

# Deactivate when done
conda deactivate
```

### Update Environment
```bash
conda activate bulkrna-agent
conda env update -f environment.yml --prune
```

### Remove Environment
```bash
conda deactivate
conda env remove -n bulkrna-agent
```

---

## Option 2: uv (Fastest) ⚡

**Pros**: Extremely fast, modern, great dependency resolution  
**Cons**: Doesn't handle R packages

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Environment
```bash
# Create environment with Python 3.11
uv venv --python 3.11 .venv

# Activate
source .venv/bin/activate

# Install dependencies (uv is much faster than pip)
uv pip install -r requirements.txt

# Install BulkRNA Agent
uv pip install -e .
```

### Usage
```bash
# Activate
source .venv/bin/activate

# Run
python run_agent.py

# Or use uv directly
uv run python run_agent.py
```

### Update Dependencies
```bash
uv pip install --upgrade -r requirements.txt
```

---

## Option 3: venv (Built-in) 📦

**Pros**: No extra tools needed, built into Python  
**Cons**: Slower than uv, doesn't handle R packages

### Create Environment
```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Usage
```bash
# Activate
source venv/bin/activate

# Run
python run_agent.py

# Deactivate
deactivate
```

---

## Comparison Table

| Feature | Conda | uv | venv |
|---------|-------|----|----- |
| Speed | Medium | **Very Fast** | Slow |
| R Support | ✅ Yes | ❌ No | ❌ No |
| Bioconductor | ✅ Yes | ❌ No | ❌ No |
| Python Packages | ✅ Yes | ✅ Yes | ✅ Yes |
| Built-in | ❌ No | ❌ No | ✅ Yes |
| Setup Complexity | Medium | Low | Low |
| **Recommendation** | **⭐ Best for this project** | Good for Python-only | Basic option |

---

## Recommended Workflow

### For Full Features (Python + R):
```bash
# Use Conda
conda env create -f environment.yml
conda activate bulkrna-agent
python run_agent.py
```

### For Python-only (faster):
```bash
# Use uv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e .
python run_agent.py
```

---

## Ollama Models (Required)

**Note**: Regardless of Python environment, you need Ollama models:

```bash
# Pull models (do this once)
ollama pull gpt-oss:20b
ollama pull cniongolo/biomistral

# Verify
ollama list
```

---

## Updated Install Script

The install.sh script uses venv by default. To use conda instead:

```bash
# Edit install.sh or run manually:
conda env create -f environment.yml
conda activate bulkrna-agent
pip install -e .
python examples/generate_example_data.py
python run_agent.py
```

---

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
# Edit as needed
```

---

## Troubleshooting

### Conda is slow
```bash
# Use mamba (faster conda alternative)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

### Python package conflicts
```bash
# With conda: use strict channel priority
conda config --set channel_priority strict

# With uv: it has better dependency resolution by default
```

### R packages won't install
```bash
# Conda approach (easiest):
conda install -c bioconda bioconductor-deseq2

# Or manual R installation:
Rscript -e "install.packages('BiocManager')"
Rscript -e "BiocManager::install('DESeq2')"
```

---

## My Recommendation for You 🎯

Since you're working on **transcriptomics with both Python and R**:

1. **Use Conda** - It's the most complete solution for bioinformatics
2. Already handles R + Bioconductor + Python packages
3. Widely used in the bioinformatics community
4. The environment.yml file I created includes everything

```bash
# Quick start
conda env create -f environment.yml
conda activate bulkrna-agent
./start.sh
```

If you only plan to use PyDESeq2 (Python) and want speed:
- Use **uv** - it's dramatically faster than pip

If you want minimal setup:
- Stick with **venv** - it's already in the install scripts
