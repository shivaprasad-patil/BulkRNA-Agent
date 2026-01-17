# RNAseqQC Package Integration

## Overview
Successfully integrated the RNAseqQC R package to generate comprehensive quality control plots during differential expression analysis. These plots provide deeper insights into sample quality, library characteristics, and data structure.

## Implementation Summary

### 1. R Script Enhancement (data/cache/run_deseq2.R)
Added RNAseqQC plot generation after DESeq2 analysis:

#### QC Plots Generated:
1. **Total Sample Counts** (`plot_total_counts`)
   - Shows total read counts per sample
   - Helps identify samples with unusually low/high sequencing depth
   - Title: "Total sample counts"

2. **Library Complexity** (`plot_library_complexity`)
   - Displays what fraction of counts is taken up by what fraction of genes
   - Identifies samples with different complexity patterns
   - Title: "Library complexity"

3. **Variance Stabilization** (`mean_sd_plot`)
   - Mean-SD plot after variance stabilizing transformation (VST)
   - Verifies that variance is stabilized across expression levels
   - Uses `vst(dds)` transformation

4. **Sample Clustering** (`plot_sample_clustering`)
   - Hierarchical clustering heatmap with distance matrix
   - **Automatically uses ALL metadata variables** for annotation
   - Uses Euclidean distance
   - Set seed for reproducible annotation colors

5. **PCA Scatters** (`plot_pca_scatters`)
   - Matrix of scatter plots for multiple principal components (5 PCs)
   - Color by first metadata variable, shape by second (if available)
   - Comprehensive view of sample relationships across multiple PCs

### 2. Python Tool Enhancement (bulkrna_agent/tools.py)
Added `get_rnaseqqc_plots()` method to `DifferentialExpressionTool` class:
- Retrieves paths to all generated QC plots
- Checks plot availability
- Returns structured dictionary with plot paths
- Handles missing plots gracefully

### 3. Web Interface Enhancement (bulkrna_agent/web_interface.py)

#### New Method: `get_rnaseqqc_plots_html()`
- Converts PNG plot images to base64-encoded HTML
- Generates formatted HTML for each QC plot
- Provides informative messages when plots are unavailable
- Returns tuple of 6 HTML strings for all QC plots

#### New UI Components in DE Tab:
Added 6 new tabs within the Differential Expression section:
- 📊 QC: Total Counts
- 📊 QC: Library Complexity
- 📊 QC: Variance Stabilization
- 📊 QC: Sample Clustering
- 📊 QC: PCA Scatters

#### New Functionality:
1. **Manual Refresh Button**: "🔄 Show/Refresh QC Plots"
   - Allows users to manually load/refresh QC plots
   
2. **Auto-loading**: QC plots are automatically loaded when DE analysis completes
   - Integrated into `run_de_and_update()` function
   - Plots appear immediately after successful DE analysis

## Usage Instructions

### Prerequisites
Install the RNAseqQC R package:
```r
install.packages("RNAseqQC")
```

### In the Web Interface
1. Upload your count matrix and metadata files
2. Run Quality Control (optional, for basic filtering)
3. Navigate to "📊 Differential Expression" tab
4. Configure and run DE analysis
5. **QC plots will automatically appear** in the new QC tabs
6. Use the "🔄 Show/Refresh QC Plots" button to manually refresh if needed

### Clustering Variables
The clustering plot automatically uses **all available metadata columns** for annotation. To control which variables are shown:
- Ensure your metadata file contains only the columns you want to see
- All categorical/factor columns in metadata will be used for heatmap annotation

## Output Location
All RNAseqQC plots are saved to:
```
data/outputs/de_analysis/qc_plots/
```

Individual plot files:
- `total_counts.png`
- `library_complexity.png`
- `variance_stabilization.png`
- `sample_clustering.png`
- `pca_scatters.png`

## Important Notes

### Gene ID Requirements
RNAseqQC requires **ENSEMBL gene IDs** as row names in the count matrix. If your data uses other identifiers (e.g., gene symbols), the biotype-related plots may not work correctly.

### Graceful Degradation
If RNAseqQC is not installed or if gene annotations are missing:
- The R script will continue and complete DESeq2 analysis
- Standard DESeq2 plots will still be generated
- The web interface will show informative messages for unavailable plots
- No errors will interrupt the analysis pipeline

### Metadata Considerations
- Sample clustering uses all metadata columns automatically
- For PCA scatters, the first two categorical metadata columns are used for coloring and shaping
- Ensure metadata column names are informative (e.g., "treatment", "mutation", "replicate")

## Benefits

1. **Comprehensive QC**: Five different perspectives on data quality
2. **Automated Integration**: Plots generated during normal DE workflow
3. **No Manual Intervention**: All metadata variables used automatically
4. **Publication-Ready**: High-quality plots suitable for reports/papers
5. **Interactive Exploration**: Easy access through web interface tabs
6. **Robust Error Handling**: Continues even if QC package unavailable

## Troubleshooting

### Plots Not Showing
1. Verify RNAseqQC package is installed in R
2. Check that gene IDs are in ENSEMBL format
3. Ensure DE analysis has been run successfully
4. Click "🔄 Show/Refresh QC Plots" to manually refresh

### Biotype Plot Issues
- Requires ENSEMBL gene IDs with version numbers (e.g., "ENSG00000000003.15")
- If using gene symbols, this plot may not generate

### Clustering Plot Missing Variables
- Check that metadata file has the columns you expect
- All columns in metadata are automatically included

## Future Enhancements

Potential improvements for future versions:
1. Add user-selectable variables for clustering (multi-select dropdown)
2. Add more RNAseqQC plots (e.g., `plot_gene_detection`, `plot_chromosome`)
3. Interactive plot options (zoom, pan, select samples)
4. Export plots in multiple formats (PDF, SVG, PNG)
5. Batch download all QC plots as ZIP file

## References

- RNAseqQC Package: https://cran.r-project.org/web/packages/RNAseqQC/
- RNAseqQC Vignette: https://cran.r-project.org/web/packages/RNAseqQC/vignettes/introduction.html
- DESeq2: https://bioconductor.org/packages/release/bioc/html/DESeq2.html
