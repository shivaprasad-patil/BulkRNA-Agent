# API Documentation

## BulkRNA Agent API

### Core Classes

#### BulkRNAConfig

Configuration class for the agent.

```python
from bulkrna_agent import BulkRNAConfig

config = BulkRNAConfig()

# LLM settings
config.llm.reasoning_model = "gpt-oss:20b"
config.llm.biomedical_model = "cniongolo/biomistral"
config.llm.base_url = "http://localhost:11434"
config.llm.temperature = 0.1

# Analysis settings
config.analysis.fdr_threshold = 0.05
config.analysis.log2fc_threshold = 1.0
config.analysis.min_count_threshold = 10

# Logging
config.log_level = "INFO"
config.log_file = "./logs/agent.log"
```

#### DualLLMManager

Manages two LLMs: reasoning and biomedical.

```python
from bulkrna_agent.llm import DualLLMManager

llm_manager = DualLLMManager(config)

# Generate response (auto-routes to appropriate LLM)
response = llm_manager.generate("What is differential expression?")

# Explicitly choose LLM
response = llm_manager.generate(
    "Explain DESeq2 normalization",
    llm_type="biomedical"
)

# Chat interface
messages = [
    {"role": "user", "content": "What is RNA-seq?"}
]
response = llm_manager.chat(messages)
```

#### BulkRNAAgent

Main ReAct agent for analysis.

```python
from bulkrna_agent.agent import BulkRNAAgent

agent = BulkRNAAgent(config, llm_manager, tools)

# Chat with agent
response = agent.chat("Analyze my data")

# Suggest design matrix
suggestion = agent.suggest_design_matrix("metadata.csv")

# Reset conversation
agent.reset()
```

### Tools

#### QualityControlTool

```python
from bulkrna_agent.tools import QualityControlTool

qc_tool = QualityControlTool(config)

result = qc_tool.execute(
    counts_file="counts.csv",
    metadata_file="metadata.csv",
    min_counts=10,
    min_samples=2
)

# Result structure:
# {
#   "status": "success",
#   "metrics": {...},
#   "filtered_counts_path": "...",
#   "n_genes_before": 20000,
#   "n_genes_after": 15000
# }
```

#### DifferentialExpressionTool

```python
from bulkrna_agent.tools import DifferentialExpressionTool

de_tool = DifferentialExpressionTool(config)

result = de_tool.execute(
    counts_file="filtered_counts.csv",
    metadata_file="metadata.csv",
    design_formula="~ condition",
    use_mcp=False  # True for R DESeq2, False for PyDESeq2
)

# Result structure:
# {
#   "status": "success",
#   "results_path": "...",
#   "significant_genes_path": "...",
#   "n_significant": 500,
#   "n_upregulated": 250,
#   "n_downregulated": 250
# }
```

#### EnrichmentAnalysisTool

```python
from bulkrna_agent.tools import EnrichmentAnalysisTool

enrich_tool = EnrichmentAnalysisTool(config)

result = enrich_tool.execute(
    gene_list=["TP53", "BRCA1", "EGFR"],
    databases=[
        "GO_Biological_Process_2021",
        "KEGG_2021_Human"
    ]
)

# Result structure:
# {
#   "status": "success",
#   "output_dir": "...",
#   "databases": [...],
#   "n_genes": 3
# }
```

#### DesignMatrixSuggestionTool

```python
from bulkrna_agent.tools import DesignMatrixSuggestionTool

design_tool = DesignMatrixSuggestionTool(config, llm_manager)

result = design_tool.execute(metadata_file="metadata.csv")

# Result structure:
# {
#   "status": "success",
#   "metadata_summary": {...},
#   "suggested_design": "~ batch + condition",
#   "explanation": "...",
#   "possible_contrasts": [...]
# }
```

### Web Interface

#### BulkRNAWebInterface

```python
from bulkrna_agent.web_interface import BulkRNAWebInterface

# Initialize
app = BulkRNAWebInterface(config)

# Launch
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Create public link
)
```

### MCP Server

#### RTranscriptomicsMCPServer

```python
from bulkrna_agent.mcp_server import RTranscriptomicsMCPServer

mcp_server = RTranscriptomicsMCPServer(config)

# Run DESeq2 via R
result = mcp_server.run_deseq2(
    counts_file="counts.csv",
    metadata_file="metadata.csv",
    design_formula="~ condition",
    contrast=["condition", "treated", "control"]
)
```

### Example Workflows

#### Complete Analysis Workflow

```python
from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.llm import DualLLMManager
from bulkrna_agent.agent import BulkRNAAgent
from bulkrna_agent.tools import *

# Initialize
config = BulkRNAConfig()
llm_manager = DualLLMManager(config)

tools = {
    "qc": QualityControlTool(config),
    "de": DifferentialExpressionTool(config),
    "enrich": EnrichmentAnalysisTool(config),
    "design": DesignMatrixSuggestionTool(config, llm_manager)
}

agent = BulkRNAAgent(config, llm_manager, tools)

# QC
qc_result = tools["qc"].execute(
    counts_file="counts.csv",
    metadata_file="metadata.csv"
)

# Design suggestion
design_result = tools["design"].execute(
    metadata_file="metadata.csv"
)

# DE analysis
de_result = tools["de"].execute(
    counts_file=qc_result["filtered_counts_path"],
    metadata_file="metadata.csv",
    design_formula=design_result["suggested_design"]
)

# Enrichment
import pandas as pd
sig_genes = pd.read_csv(de_result["significant_genes_path"], index_col=0)
enrich_result = tools["enrich"].execute(
    gene_list=sig_genes.index.tolist()
)

# Chat about results
response = agent.chat("What are the key findings?")
print(response)
```

## Error Handling

All tools return a status field:

```python
result = tool.execute(...)

if result["status"] == "success":
    # Process results
    data = result["data"]
elif result["status"] == "error":
    # Handle error
    print(f"Error: {result['message']}")
```

## Logging

All operations are logged:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# View logs
tail -f logs/bulkrna_agent.log
```
