# Troubleshooting Guide

## Common Issues and Solutions

### 1. Ollama Connection Errors

**Error**: `Connection refused to localhost:11434`

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Check if port is in use
lsof -i :11434
```

### 2. Model Not Found

**Error**: `model 'gpt-oss:20b' not found`

**Solutions**:
```bash
# List available models
ollama list

# Pull missing model
ollama pull gpt-oss:20b
ollama pull cniongolo/biomistral

# Verify models
ollama list | grep -E "gpt-oss|biomistral"
```

### 3. PyDESeq2 Installation Issues

**Error**: `ModuleNotFoundError: No module named 'pydeseq2'`

**Solutions**:
```bash
# Install PyDESeq2
pip install pydeseq2

# If that fails, install dependencies first
pip install numpy scipy pandas
pip install pydeseq2

# Alternative: use specific version
pip install pydeseq2==0.4.0
```

**Error**: `ValueError: numpy.dtype size changed`

**Solution**:
```bash
# Reinstall with compatible numpy
pip uninstall numpy pydeseq2
pip install numpy==1.24.0
pip install pydeseq2==0.4.0
```

### 4. R DESeq2 Errors

**Error**: `Error in library(DESeq2) : there is no package called 'DESeq2'`

**Solutions**:
```bash
# Install DESeq2
Rscript -e "install.packages('BiocManager')"
Rscript -e "BiocManager::install('DESeq2')"

# If that fails, update R
brew install r  # macOS
# or download from https://cran.r-project.org/

# Install with specific version
Rscript -e "BiocManager::install('DESeq2', version='3.18')"
```

**Error**: `Rscript: command not found`

**Solution**:
```bash
# Add R to PATH (macOS)
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify R installation
which R
R --version
```

### 5. Gradio Interface Issues

**Error**: `Address already in use`

**Solutions**:
```bash
# Find process using port 7860
lsof -ti:7860

# Kill the process
kill -9 $(lsof -ti:7860)

# Or use different port
python run_agent.py --port 7861
```

**Error**: `Gradio interface not loading`

**Solutions**:
```bash
# Clear browser cache
# Or open in incognito/private mode

# Check firewall settings
# On macOS: System Preferences > Security & Privacy > Firewall

# Try different browser
# Or access from different device using --host 0.0.0.0
```

### 6. File Upload Issues

**Error**: `Permission denied when uploading files`

**Solutions**:
```bash
# Check directory permissions
ls -la data/uploads

# Create directories with correct permissions
mkdir -p data/uploads data/outputs data/cache
chmod 755 data/uploads data/outputs data/cache

# Change ownership if needed
sudo chown -R $USER data/
```

**Error**: `File too large`

**Solution**:
```python
# Increase max file size in config
config = BulkRNAConfig()
config.data.max_file_size_mb = 1000  # Increase to 1GB
```

### 7. Memory Issues

**Error**: `MemoryError` or `Killed`

**Solutions**:
```bash
# Check available memory
free -h  # Linux
vm_stat | grep free  # macOS

# Reduce data size
# Filter genes before upload
# Or increase system memory

# Use more efficient data types
# Load data in chunks
```

**For DESeq2**:
```python
# Use PyDESeq2 instead of R (lower memory)
de_tool.execute(..., use_mcp=False)

# Or filter data more aggressively
qc_tool.execute(..., min_counts=20, min_samples=3)
```

### 8. LLM Response Issues

**Error**: `Timeout waiting for LLM response`

**Solutions**:
```python
# Increase timeout
config.llm.timeout = 600  # 10 minutes

# Reduce max_tokens
config.llm.max_tokens = 2048

# Use smaller model
config.llm.reasoning_model = "llama2:7b"
```

**Error**: `LLM giving incorrect responses`

**Solutions**:
```python
# Adjust temperature
config.llm.temperature = 0.0  # More deterministic

# Try different model
config.llm.reasoning_model = "mistral:latest"

# Check model is loaded
ollama list
```

### 9. Enrichment Analysis Issues

**Error**: `Connection timeout to Enrichr`

**Solutions**:
```bash
# Check internet connection
ping maayanlab.cloud

# Try again later (API may be down)

# Use VPN if blocked

# Alternative: use local enrichment
# Install gprofiler2
pip install gprofiler-official
```

**Error**: `No enrichment results found`

**Solutions**:
```python
# Check gene names format
# Enrichr expects gene symbols, not Ensembl IDs

# Convert Ensembl to gene symbols first
# Or use different organism
enrich_tool.execute(gene_list, organism="mouse")
```

### 10. Logging Issues

**Error**: `Permission denied: logs/bulkrna_agent.log`

**Solutions**:
```bash
# Create logs directory
mkdir -p logs
chmod 755 logs

# Or change log file location
config.log_file = "/tmp/bulkrna_agent.log"
```

**Error**: `Log file too large`

**Solutions**:
```bash
# Rotate log files
mv logs/bulkrna_agent.log logs/bulkrna_agent.log.old
gzip logs/bulkrna_agent.log.old

# Or configure log rotation
# Use Python's RotatingFileHandler
```

### 11. Data Format Issues

**Error**: `ValueError: could not convert string to float`

**Solutions**:
```python
# Check CSV format
# Ensure first column is gene IDs
# Ensure other columns are numeric

# Fix format:
import pandas as pd
df = pd.read_csv("counts.csv")
df.set_index(df.columns[0], inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.to_csv("counts_fixed.csv")
```

**Error**: `Samples don't match between counts and metadata`

**Solutions**:
```python
# Ensure sample names match exactly
counts = pd.read_csv("counts.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)

print("Counts samples:", counts.columns.tolist())
print("Metadata samples:", metadata.index.tolist())

# Reorder metadata to match counts
metadata = metadata.loc[counts.columns]
metadata.to_csv("metadata_fixed.csv")
```

## Getting Help

If you're still having issues:

1. **Check Logs**: Look at `logs/bulkrna_agent.log` for detailed error messages

2. **Enable Debug Logging**:
   ```python
   config.log_level = "DEBUG"
   ```

3. **Test Components Individually**:
   ```bash
   # Test Ollama
   ollama run gpt-oss:20b "Hello"
   
   # Test Python dependencies
   python -c "import pandas, numpy, gradio, pydeseq2"
   
   # Test R DESeq2
   Rscript -e "library(DESeq2)"
   ```

4. **Open GitHub Issue**: Provide:
   - Error message
   - Log file excerpt
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "gradio|pandas|pydeseq2"`
   - OS and version

5. **Check System Requirements**:
   - Python 3.9+
   - 8GB+ RAM recommended
   - 10GB+ free disk space
   - Internet connection for Enrichr
