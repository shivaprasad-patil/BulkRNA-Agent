# BulkRNA Agent - Quick Reference

## 🚀 Quick Start

```bash
# 1. Start Ollama (if not running)
ollama serve

# 2. Launch agent
./start.sh
# OR
python run_agent.py

# 3. Open browser
# http://localhost:7860
```

## 📁 Data Formats

### Count Matrix (CSV/TSV)
```
gene_id,sample1,sample2,sample3
ENSG00000000003,2000,1500,2200
ENSG00000000005,100,80,110
```

### Metadata (CSV/TSV)
```
sample_id,condition,batch
sample1,control,batch1
sample2,treated,batch1
```

## 🔧 Configuration

```python
from bulkrna_agent import BulkRNAConfig

config = BulkRNAConfig()

# LLM Settings
config.llm.reasoning_model = "gpt-oss:20b"
config.llm.biomedical_model = "cniongolo/biomistral"
config.llm.temperature = 0.1

# Analysis Thresholds
config.analysis.fdr_threshold = 0.05      # FDR < 0.05
config.analysis.log2fc_threshold = 1.0    # |log2FC| > 1
config.analysis.min_count_threshold = 10  # Min counts
```

## 📊 Common Design Formulas

```r
# Simple comparison
~ condition

# With batch effect
~ batch + condition

# Multiple factors
~ sex + condition

# Interaction
~ condition + time + condition:time

# Complex
~ batch + sex + condition
```

## 🛠️ Command Line

```bash
# Launch with options
bulkrna-agent --host 0.0.0.0 --port 7860 --share

# Generate example data
python examples/generate_example_data.py

# Run tests
pytest tests/

# View logs
tail -f logs/bulkrna_agent.log
```

## 🐍 Programmatic Usage

### Quick Analysis

```python
from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.web_interface import BulkRNAWebInterface

config = BulkRNAConfig()
app = BulkRNAWebInterface(config)
app.launch()
```

### Full Control

```python
from bulkrna_agent.tools import *

config = BulkRNAConfig()

# QC
qc = QualityControlTool(config)
qc_result = qc.execute(
    counts_file="counts.csv",
    metadata_file="metadata.csv"
)

# DE
de = DifferentialExpressionTool(config)
de_result = de.execute(
    counts_file=qc_result["filtered_counts_path"],
    metadata_file="metadata.csv",
    design_formula="~ condition"
)

# Enrichment
enrich = EnrichmentAnalysisTool(config)
enrich_result = enrich.execute(
    gene_list=["TP53", "BRCA1", "EGFR"]
)
```

## 💬 Example Chat Questions

- "What are the most significant genes?"
- "Explain the enriched pathways"
- "Should I account for batch effects?"
- "What does log2 fold change mean?"
- "How many upregulated genes do I have?"
- "What is the biological significance of these results?"

## 🔍 Analysis Workflow

```
1. Upload Data → 2. QC → 3. Design Suggestion → 
4. DE Analysis → 5. Enrichment → 6. Chat/Interpret
```

## ⚙️ Adjusting Stringency

**More Stringent** (fewer false positives):
```python
config.analysis.fdr_threshold = 0.01
config.analysis.log2fc_threshold = 2.0
```

**Less Stringent** (more discoveries):
```python
config.analysis.fdr_threshold = 0.1
config.analysis.log2fc_threshold = 0.5
```

## 📈 Output Files

```
data/outputs/
├── qc/
│   ├── filtered_counts.csv
│   └── qc_metrics.json
├── de_analysis/
│   ├── de_results.csv
│   ├── significant_genes.csv
│   ├── normalized_counts.csv
│   └── plots/
└── enrichment/
    ├── enrichment_GO_*.csv
    ├── enrichment_KEGG_*.csv
    └── enrichment_Reactome_*.csv
```

## 🐛 Quick Troubleshooting

### Ollama not responding
```bash
curl http://localhost:11434/api/tags
ollama serve
```

### Model not found
```bash
ollama pull gpt-oss:20b
ollama pull cniongolo/biomistral
```

### PyDESeq2 errors
```bash
pip install --upgrade pydeseq2
pip install numpy==1.24.0
```

### Port in use
```bash
lsof -ti:7860 | xargs kill -9
# OR
python run_agent.py --port 7861
```

## 📚 Documentation

- [Full README](README.md)
- [Tutorial](docs/TUTORIAL.md)
- [API Reference](docs/API.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🎯 Key Features

✅ Quality Control with filtering
✅ DESeq2 (Python or R)
✅ Gene set enrichment (GO, KEGG, Reactome)
✅ AI-powered design suggestions
✅ Chat interface for questions
✅ Dual LLM system (reasoning + biomedical)
✅ Gradio web interface
✅ Comprehensive logging
✅ MCP server integration

## 💡 Tips

- Always run QC before DE analysis
- Use design suggestion for complex experiments
- Check logs if something fails
- Start with stringent thresholds
- Validate top genes with literature
- Use chat for biological interpretation

## 🆘 Support

- Logs: `logs/bulkrna_agent.log`
- Issues: GitHub Issues
- Docs: `docs/` directory
