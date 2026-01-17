# 🧬 BulkRNA Agent

AI-powered bulk RNA-seq analysis tool with interactive web interface. Perform comprehensive transcriptomics analysis from QC to enrichment analysis with dual LLM system for intelligent insights.

## 🎥 Demo


https://github.com/user-attachments/assets/88d02e07-7f4f-4166-b426-4f34d3c0fca6

> Watch the complete workflow from data upload to enrichment analysis.

## ✨ Features

- **Quality Control**: Automated filtering, PCA visualization, and RNAseqQC plots
- **Differential Expression**: DESeq2/PyDESeq2 with AI-powered design suggestions
- **Enrichment Analysis**: GO, KEGG, Reactome gene set analysis
- **AI Chat**: Ask questions about your data (powered by biomistral + gpt-oss)
- **Interactive Plots**: Volcano, MA, PCA scatter plots with customization
- **Web Interface**: User-friendly Gradio interface

## 🚀 Quick Start

### Prerequisites

1. **Ollama** with models:
   ```bash
   ollama pull gpt-oss:20b
   ollama pull cniongolo/biomistral
   ```

2. **Python 3.9+** and **Conda** (recommended)

3. **R + DESeq2** (optional, for R-based analysis)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Tx_AI_agent.git
cd Tx_AI_agent

# Quick setup with script
./install.sh

# Or manual setup
conda env create -f environment.yml
conda activate bulkrna-agent
pip install -e .
```

### Launch

```bash
./start.sh
# Or: python run_agent.py
```

Open browser to http://localhost:7860

**See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for detailed comparison and recommendations.**

### Development Install

```bash
pip install -e ".[dev]"
```

## 💻 Usage

### Launch Web Interface

```bash
# Start the Gradio interface
bulkrna-agent

# Or with custom settings
python -m bulkrna_agent.web_interface --host 0.0.0.0 --port 7860 --share
```

### Command Line Options

```bash
bulkrna-agent --help

Options:
  --host HOST       Host to run on (default: 127.0.0.1)
  --port PORT       Port to run on (default: 7860)
  --share           Create public Gradio link
```

### Using the Web Interface

1. **Upload Data**
   - Upload count matrix (genes × samples CSV/TSV)
   - Upload sample metadata (samples × conditions CSV/TSV)


## 📊 Usage

1. **Upload Data**: Count matrix (CSV) and metadata (CSV)
2. **Run QC**: Filter low-count genes and visualize PCA
3. **Differential Expression**: AI suggests design formula → Run DESeq2
4. **Enrichment**: Automatic GO/KEGG/Reactome analysis
5. **Explore**: Interactive plots and AI chat for insights

### Example Data Format

**counts.csv:**
```csv
gene_id,sample1,sample2,sample3
ENSG00000000003,2000,1500,2200
ENSG00000000005,100,80,110
```

**metadata.csv:**
```csv
sample_id,condition,batch
sample1,control,A
sample2,treated,A
sample3,treated,B
```

## 🔧 Advanced

### Custom Configuration
```python
from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.web_interface import BulkRNAWebInterface

config = BulkRNAConfig()
config.analysis.fdr_threshold = 0.01
config.llm.reasoning_model = "gpt-oss:20b"

app = BulkRNAWebInterface(config)
app.launch()
```

### R Integration
For R DESeq2 support, install Bioconductor packages:
```bash
Rscript -e "install.packages('BiocManager')"
Rscript -e "BiocManager::install(c('DESeq2', 'RNAseqQC'))"
```

### LLM Routing

The agent automatically routes queries:
- **Biomedical LLM**: Gene function, pathways, biological interpretation
- **Reasoning LLM**: Tool selection, analysis planning, statistical decisions

### Logging

All operations are logged to:
- Console (INFO level)
- File: `./logs/bulkrna_agent.log` (detailed logs)

Check logs for debugging:
```bash
tail -f logs/bulkrna_agent.log
```

## 🐛 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve
```

### Model Not Found

```bash
# Pull missing models
ollama pull gpt-oss:20b
ollama pull cniongolo/biomistral
```

## 📚 Documentation

- [Getting Started](GETTING_STARTED.md) - Detailed installation guide
- [Quick Reference](QUICKREF.md) - Command cheat sheet
- [Contributing](CONTRIBUTING.md) - How to contribute
- [API Docs](docs/API.md) - Python API reference
- [RNAseqQC Integration](docs/RNASEQQC_INTEGRATION.md) - QC plots guide
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) and [Ollama](https://ollama.ai/)
- Uses [DESeq2](https://bioconductor.org/packages/DESeq2/), [PyDESeq2](https://github.com/owkin/PyDESeq2), [RNAseqQC](https://cran.r-project.org/package=RNAseqQC)
- Inspired by [Biomni](https://github.com/yourusername/biomni) framework

---

**Questions?** Open an issue or check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- [ ] Interactive plots in web interface
- [ ] Export analysis reports (PDF/HTML)
- [ ] Docker container for easy deployment

---

**Built with ❤️ for the transcriptomics community**
