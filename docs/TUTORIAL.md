# BulkRNA Agent Tutorial

Complete walkthrough for analyzing bulk RNA-seq data with BulkRNA Agent.

## Prerequisites

Before starting, ensure you have:
- ✅ Ollama installed with models: `gpt-oss:20b` and `cniongolo/biomistral`
- ✅ Python 3.9+ installed
- ✅ BulkRNA Agent installed
- ✅ Your RNA-seq count data and metadata files

## Tutorial: Analyzing Example Dataset

### Step 1: Generate Example Data

Let's start by generating synthetic data for this tutorial:

```bash
# Generate example data
python examples/generate_example_data.py
```

This creates:
- `data/examples/example_counts.csv` - Count matrix (1000 genes × 6 samples)
- `data/examples/example_metadata.csv` - Sample metadata

### Step 2: Start BulkRNA Agent

Launch the web interface:

```bash
# Quick start
./start.sh

# Or manually
python run_agent.py
```

Open your browser to: http://localhost:7860

### Step 3: Upload Data

1. Go to **"📁 Data Upload"** tab
2. Upload count matrix:
   - Click **"Count Matrix"** upload button
   - Select `data/examples/example_counts.csv`
   - Wait for confirmation ✅
3. Upload metadata:
   - Click **"Sample Metadata"** upload button
   - Select `data/examples/example_metadata.csv`
   - Wait for confirmation ✅

**What you should see:**
- Preview of your data
- Dimensions: genes × samples
- Sample names and conditions

### Step 4: Quality Control

1. Go to **"🔍 Quality Control"** tab
2. Click **"Run Quality Control"**
3. Review QC metrics:
   - Total genes before/after filtering
   - Library sizes per sample
   - Number of detected genes
   - Filtered count matrix location

**Expected Output:**
```
✅ Quality Control Complete

Summary:
- Total genes (before filtering): 1000
- Total genes (after filtering): ~850
- Genes removed: ~150
- Number of samples: 6

Library Sizes:
- Median: 150,000 reads

Per-Sample Metrics:
- control_1: 145,230 reads, 842 genes
- control_2: 152,100 reads, 855 genes
- ...
```

**What's happening:**
- Filtering genes with low counts
- Calculating library sizes
- Assessing data quality
- Saving filtered count matrix

### Step 5: Design Matrix Suggestion

1. Go to **"📊 Differential Expression"** tab
2. Click **"Suggest Design Matrix"**
3. Review AI suggestion:
   - Recommended design formula
   - Explanation of the design
   - Possible contrasts to test

**Expected Output:**
```
Suggested Design Matrix

Design Formula: `~ condition`

Explanation:
This dataset has a simple experimental design with one 
factor (condition) with two levels (control and treated). 
The design formula `~ condition` will compare treated vs 
control samples while accounting for baseline expression.

Possible Contrasts:
- ["condition", "treated", "control"]
```

**Understanding the design:**
- `~ condition`: Tests effect of treatment
- `~ batch + condition`: Accounts for batch effects
- `~ condition + condition:time`: Tests interaction

### Step 6: Differential Expression Analysis

1. In the **"📊 Differential Expression"** tab
2. Enter design formula: `~ condition`
3. Choose analysis method:
   - ☐ **Unchecked**: PyDESeq2 (Python) - faster, lower memory
   - ☑ **Checked**: R DESeq2 (via MCP) - more features, requires R
4. Click **"Run Differential Expression"**

**Expected Output:**
```
✅ Differential Expression Analysis Complete

Design Formula: `~ condition`

Results:
- Total significant genes: 98
- Upregulated: 49
- Downregulated: 49

Thresholds:
- FDR < 0.05
- |log2FC| > 1.0

📁 Results saved to: data/outputs/de_analysis/de_results.csv
📁 Significant genes: data/outputs/de_analysis/significant_genes.csv
```

**Understanding the results:**
- **Upregulated**: Higher in treated vs control
- **Downregulated**: Lower in treated vs control
- **FDR (False Discovery Rate)**: Adjusted p-value
- **log2FC**: Log2 fold change

**Interpreting Results:**

View significant genes:
```bash
# Top upregulated genes
head -20 data/outputs/de_analysis/significant_genes.csv

# Or in Python:
import pandas as pd
df = pd.read_csv("data/outputs/de_analysis/significant_genes.csv", index_col=0)
df_up = df[df['log2FoldChange'] > 0].sort_values('padj')
print(df_up.head(10))
```

### Step 7: Enrichment Analysis

1. Go to **"🎯 Enrichment Analysis"** tab
2. Click **"Run Enrichment Analysis"**
3. Wait for analysis to complete
4. Review enriched pathways

**Expected Output:**
```
✅ Enrichment Analysis Complete

Analyzed 98 significant genes

Databases queried:
- GO_Biological_Process_2021
- GO_Molecular_Function_2021
- KEGG_2021_Human
- Reactome_2022

📁 Results saved to: data/outputs/enrichment/
```

**View enrichment results:**
```bash
# View enriched GO terms
cat data/outputs/enrichment/enrichment_GO_Biological_Process_2021.csv

# Or in Python:
import pandas as pd
df = pd.read_csv("data/outputs/enrichment/enrichment_GO_Biological_Process_2021.csv")
print(df[['term', 'adjusted_p_value']].head(10))
```

### Step 8: Chat with the Agent

1. Go to **"💬 Chat with Agent"** tab
2. Ask questions about your data and results

**Example Questions:**

**Q1: "What are the most significant genes in my analysis?"**

Expected response explaining top DE genes with statistics.

**Q2: "What biological processes are enriched?"**

Expected response summarizing key pathways and their significance.

**Q3: "Explain what log2 fold change means"**

Expected response with clear explanation of the metric.

**Q4: "Should I be concerned about batch effects?"**

Expected response analyzing your design and suggesting whether batch correction is needed.

**Q5: "What do these enriched pathways tell me about my experiment?"**

Expected biological interpretation of enrichment results.

## Advanced Usage

### Custom Design Matrices

For complex designs:

**Example 1: Batch Effect Correction**
```
Design: ~ batch + condition
```

**Example 2: Multiple Factors**
```
Design: ~ sex + condition
```

**Example 3: Interaction Terms**
```
Design: ~ condition + time + condition:time
```

### Adjusting Thresholds

Modify analysis thresholds:

```python
from bulkrna_agent import BulkRNAConfig

config = BulkRNAConfig()

# More stringent (fewer false positives)
config.analysis.fdr_threshold = 0.01
config.analysis.log2fc_threshold = 2.0

# Less stringent (more discoveries)
config.analysis.fdr_threshold = 0.1
config.analysis.log2fc_threshold = 0.5
```

### Programmatic Analysis

Run analysis without web interface:

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
}

# Run analysis
qc_result = tools["qc"].execute(
    counts_file="data/examples/example_counts.csv",
    metadata_file="data/examples/example_metadata.csv"
)

de_result = tools["de"].execute(
    counts_file=qc_result["filtered_counts_path"],
    metadata_file="data/examples/example_metadata.csv",
    design_formula="~ condition"
)

print(f"Found {de_result['n_significant']} significant genes")
```

## Analyzing Your Own Data

### Data Preparation

**Count Matrix Format:**
```csv
gene_id,sample1,sample2,sample3,...
ENSG00000000003,2000,1500,2200,...
ENSG00000000005,100,80,110,...
```

**Requirements:**
- First column: Gene IDs (Ensembl or gene symbols)
- Other columns: Raw counts (integers)
- Column names: Sample IDs matching metadata

**Metadata Format:**
```csv
sample_id,condition,batch,other_factors...
sample1,control,batch1,...
sample2,treated,batch1,...
```

**Requirements:**
- First column: Sample IDs (matching count matrix)
- Other columns: Experimental factors
- At least one condition column

### Best Practices

1. **Quality Control:**
   - Remove samples with very low library sizes
   - Filter genes with low counts across most samples
   - Check for outliers

2. **Design Matrix:**
   - Include batch if present
   - Order factors: batch → biological effects
   - Avoid over-parameterization

3. **Interpretation:**
   - Consider biological significance, not just statistical
   - Validate key findings with additional experiments
   - Use enrichment to understand biological context

4. **Reporting:**
   - Document all parameters used
   - Report both up and down-regulated genes
   - Include QC metrics in publications

## Troubleshooting

### Common Issues

**Issue**: "No significant genes found"

**Solutions:**
- Check design matrix is correct
- Reduce stringency (FDR = 0.1, log2FC = 0.5)
- Ensure biological replicates are present
- Check for batch effects or outliers

**Issue**: "Too many significant genes (>5000)"

**Solutions:**
- Increase stringency (FDR = 0.01, log2FC = 2)
- Check for data quality issues
- Verify samples are correctly labeled

**Issue**: "Enrichment finds nothing"

**Solutions:**
- Check gene ID format (symbols vs Ensembl)
- Try different databases
- Need at least 10-20 genes for enrichment
- Verify genes are expressed in relevant organism

## Next Steps

After completing this tutorial:

1. **Analyze Your Own Data**: Follow the same steps with your data
2. **Explore Results**: Use chat interface to dig deeper
3. **Customize Analysis**: Adjust parameters for your needs
4. **Visualize Results**: Create plots of top genes and pathways
5. **Share Results**: Export results for publication

## Resources

- [API Documentation](docs/API.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Example Scripts](examples/)
- [DESeq2 Documentation](https://bioconductor.org/packages/release/bioc/html/DESeq2.html)
- [Enrichr Documentation](https://maayanlab.cloud/Enrichr/)

## Getting Help

Questions or issues?
- Check the troubleshooting guide
- Review log files: `logs/bulkrna_agent.log`
- Open a GitHub issue with details

Happy analyzing! 🧬
