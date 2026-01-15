# RNAseqQC Integration - Quick Reference

## What Changed

### 📁 Files Modified

1. **data/cache/run_deseq2.R**
   - Added RNAseqQC library loading with graceful fallback
   - Created `qc_plots` subdirectory in output
   - Added 6 QC plot generation steps after DESeq2 analysis
   - All plots saved as PNG files with appropriate titles

2. **bulkrna_agent/tools.py**
   - Added `get_rnaseqqc_plots()` method to `DifferentialExpressionTool`
   - Returns paths to all 6 QC plots
   - Checks availability and handles missing plots

3. **bulkrna_agent/web_interface.py**
   - Added `get_rnaseqqc_plots_html()` method
   - Added 6 new tabs in DE section for QC plots
   - Added refresh button for QC plots
   - Auto-loads QC plots after DE analysis completes

## 🎨 New UI Elements

In the **Differential Expression** tab, after "Top Genes" tab:

```
📊 Differential Expression
├── Volcano Plot
├── MA Plot  
├── Top Genes
├── 📊 QC: Total Counts          [NEW]
├── 📊 QC: Library Complexity    [NEW]
├── 📊 QC: Biotypes              [NEW]
├── 📊 QC: Variance Stabilization [NEW]
├── 📊 QC: Sample Clustering     [NEW]
└── 📊 QC: PCA Scatters          [NEW]

[🔄 Show/Refresh QC Plots]       [NEW BUTTON]
```

## 🔧 R Code Added

```r
# Load RNAseqQC
library(RNAseqQC)

# Generate plots
plot_total_counts(dds) + ggtitle("Total sample counts")
plot_library_complexity(dds) + ggtitle("Library complexity")
plot_biotypes(dds) + ggtitle("Gene biotypes")

# Variance stabilization
vsd <- vst(dds, blind=FALSE)
mean_sd_plot(vsd) + ggtitle("Variance stabilization")

# Clustering (uses ALL metadata variables automatically)
set.seed(1)
plot_sample_clustering(vsd, anno_vars = names(metadata), distance = "euclidean")

# PCA scatters
plot_pca_scatters(vsd, n_PCs = 5, color_by = metadata[1], shape_by = metadata[2])
```

## 📊 Plot Descriptions

| Plot Name | Purpose | Key Insight |
|-----------|---------|-------------|
| **Total Counts** | Library size per sample | Identify sequencing depth issues |
| **Library Complexity** | Gene fraction vs count fraction | Detect library prep problems |
| **Biotypes** | Distribution of gene types | Check RNA composition |
| **Variance Stabilization** | Mean-SD relationship | Verify VST normalization |
| **Sample Clustering** | Hierarchical clustering | See sample relationships |
| **PCA Scatters** | Multiple PC comparisons | Comprehensive dimensionality view |

## ⚙️ Key Features

### Automatic Variable Selection
- **Clustering**: Uses ALL metadata columns automatically
- **PCA**: Uses first two metadata columns for color/shape
- No manual configuration needed!

### Error Handling
- Continues if RNAseqQC not installed
- Shows helpful messages for missing plots
- Doesn't break DE analysis pipeline

### Output Location
```
data/outputs/de_analysis/qc_plots/
├── total_counts.png
├── library_complexity.png
├── biotypes.png
├── variance_stabilization.png
├── sample_clustering.png
└── pca_scatters.png
```

## 🚀 Usage Flow

```
1. Upload counts + metadata
        ↓
2. Run DE Analysis
        ↓
3. QC plots auto-generate in R
        ↓
4. QC plots auto-display in UI
        ↓
5. Explore in 6 new tabs!
```

## ✅ Requirements Met

- ✅ `plot_total_counts(dds)` with title "Total sample counts"
- ✅ `plot_library_complexity(dds)` with title "Library complexity"
- ✅ `plot_biotypes(dds)` with title "Gene biotypes"
- ✅ `vsd <- vst(dds)` + `mean_sd_plot(vsd)` for variance stabilization
- ✅ `plot_sample_clustering(vsd, anno_vars = c(...))` with ALL metadata variables
- ✅ `plot_pca_scatters(vsd, n_PCs = 5, color_by = ..., shape_by = ...)`
- ✅ All shown in DE & results tab
- ✅ Clustering uses user's metadata variables automatically

## 💡 Pro Tips

1. **Gene IDs**: Use ENSEMBL IDs for best biotype results
2. **Metadata**: Name columns descriptively (treatment, genotype, etc.)
3. **Refresh**: Click refresh button if plots don't load initially
4. **Install**: Run `install.packages("RNAseqQC")` in R beforehand

## 🐛 Troubleshooting

**Issue**: Plots say "not available"
**Solution**: Install RNAseqQC R package

**Issue**: Biotype plot missing
**Solution**: Ensure gene IDs are ENSEMBL format

**Issue**: Want different clustering variables
**Solution**: Modify metadata file to include only desired columns

## 📚 Documentation

See `RNASEQQC_INTEGRATION.md` for comprehensive documentation.
