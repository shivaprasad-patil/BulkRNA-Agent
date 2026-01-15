"""
MCP (Model Context Protocol) Server for R-based transcriptomics tools
This server provides access to DESeq2 and other Bioconductor packages
"""
import logging
import json
import subprocess
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPServer:
    """Base MCP Server"""
    
    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.process = None
        
        logger.info(f"Initialized MCP Server: {name}")
    
    def start(self):
        """Start the MCP server"""
        try:
            logger.info(f"Starting MCP server: {self.name}")
            # Implementation would start the actual MCP server process
            # For now, this is a placeholder
            logger.info(f"MCP server {self.name} started")
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            raise
    
    def stop(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            logger.info(f"MCP server {self.name} stopped")
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool via MCP"""
        raise NotImplementedError


class RTranscriptomicsMCPServer(MCPServer):
    """
    MCP Server for R-based transcriptomics tools
    Provides access to DESeq2, edgeR, limma-voom, etc.
    """
    
    def __init__(self, config):
        server_config = config.mcp_servers.get("r_transcriptomics", {})
        super().__init__(
            name="r_transcriptomics",
            command=server_config.get("command", "Rscript"),
            args=server_config.get("args", ["./mcp_servers/r_transcriptomics_server.R"])
        )
        self.config = config
    
    def run_deseq2(
        self,
        counts_file: str,
        metadata_file: str,
        design_formula: str,
        contrast: List[str]
    ) -> Dict[str, Any]:
        """
        Run DESeq2 analysis via R
        """
        try:
            logger.info("Running DESeq2 via R")
            
            # Create R script
            r_script = self._create_deseq2_script(
                counts_file, metadata_file, design_formula, contrast
            )
            
            # Save script
            script_path = Path(self.config.data.cache_dir) / "run_deseq2.R"
            with open(script_path, 'w') as f:
                f.write(r_script)
            
            # Run R script
            result = subprocess.run(
                [self.command, str(script_path)],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Check if output files were created (success indicator)
            output_dir = Path(self.config.data.output_dir) / "de_analysis"
            contrasts_file = output_dir / "available_contrasts.txt"
            
            if not contrasts_file.exists():
                logger.error(f"DESeq2 failed - contrasts file not created")
                logger.error(f"stderr: {result.stderr[-1000:]}")  # Last 1000 chars
                return {
                    "status": "error",
                    "message": result.stderr[-500:] if result.stderr else "Unknown error"
                }
            
            # Success - results were created
            logger.info("DESeq2 completed successfully")
            if result.stdout:
                logger.info(f"R output: {result.stdout[-2000:]}")  # Last 2000 chars
            if result.stderr:
                logger.info(f"R stderr: {result.stderr[-1000:]}")  # Last 1000 chars
            
            # Parse output - get first contrast file for backward compatibility
            output_dir = Path(self.config.data.output_dir) / "de_analysis"
            
            # Read contrast names to find first results file
            contrasts_file = output_dir / "available_contrasts.txt"
            with open(contrasts_file) as f:
                contrasts = [line.strip() for line in f.readlines()]
            
            # Use first non-Intercept contrast if available
            first_contrast = contrasts[1] if len(contrasts) > 1 and contrasts[0] == "Intercept" else contrasts[0]
            first_contrast_clean = first_contrast.replace(" ", "_").replace("(", "").replace(")", "")
            
            # Find the actual file (it may have different cleaning)
            import glob
            contrast_files = glob.glob(str(output_dir / "*.csv"))
            contrast_files = [f for f in contrast_files if not f.endswith("_significant.csv") and 
                            not f.endswith("normalized_counts.csv")]
            
            results_path = contrast_files[0] if contrast_files else str(output_dir / f"{first_contrast_clean}.csv")
            
            return {
                "status": "success",
                "results_path": results_path,
                "normalized_counts_path": str(output_dir / "normalized_counts.csv"),
                "stdout": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            logger.error("DESeq2 execution timed out")
            return {"status": "error", "message": "Analysis timed out"}
        except Exception as e:
            logger.error(f"Error running DESeq2: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _create_deseq2_script(
        self,
        counts_file: str,
        metadata_file: str,
        design_formula: str,
        contrast: List[str]
    ) -> str:
        """Create R script for DESeq2 analysis"""
        
        output_dir = Path(self.config.data.output_dir) / "de_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        script = f"""
# DESeq2 Analysis Script with RNAseqQC
# Generated by BulkRNA Agent

library(DESeq2)
library(ggplot2)

# Load RNAseqQC package
tryCatch({{
  library(RNAseqQC)
  cat("RNAseqQC package loaded successfully\\n")
  rnaseqqc_available <- TRUE
}}, error = function(e) {{
  cat("Warning: RNAseqQC package not available. Install with: install.packages('RNAseqQC')\\n")
  rnaseqqc_available <- FALSE
}})

# Set output directory
output_dir <- "{output_dir}"
qc_plots_dir <- file.path(output_dir, "qc_plots")
if (!dir.exists(qc_plots_dir)) {{
  dir.create(qc_plots_dir, recursive = TRUE)
}}

# Load data
cat("Loading count data...\\n")
counts <- read.csv("{counts_file}", row.names=1, check.names=FALSE)
metadata <- read.csv("{metadata_file}", row.names=1, check.names=FALSE, stringsAsFactors=TRUE)

# Check if sample names match
cat("Count matrix samples:", paste(colnames(counts), collapse=", "), "\\n")
cat("Metadata samples:", paste(rownames(metadata), collapse=", "), "\\n")

# Ensure sample order matches
common_samples <- intersect(colnames(counts), rownames(metadata))
if (length(common_samples) == 0) {{
  stop("Error: No matching samples between counts and metadata. Check that sample names match.")
}}

if (length(common_samples) < ncol(counts)) {{
  cat("Warning: Some samples in counts not found in metadata. Using only common samples.\\n")
  counts <- counts[, common_samples]
}}

metadata <- metadata[common_samples, , drop=FALSE]
cat("Using", length(common_samples), "samples for analysis\\n")

# Create DESeq2 dataset
cat("Creating DESeq2 dataset...\\n")
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = metadata,
  design = as.formula("{design_formula}")
)

# Run DESeq2
cat("Running DESeq2...\\n")
dds <- DESeq(dds)

# Get all available contrasts
cat("\\nExtracting all available contrasts...\\n")
result_names <- resultsNames(dds)
cat("Available contrasts:", paste(result_names, collapse=", "), "\\n")

# Save list of contrasts to a file
writeLines(result_names, file.path(output_dir, "available_contrasts.txt"))

# Extract results for each contrast
for (contrast_name in result_names) {{
  cat("\\nProcessing contrast:", contrast_name, "\\n")
  
  # Get results for this contrast
  res <- results(dds, name=contrast_name)
  
  # Save results with contrast-specific filename
  res_df <- as.data.frame(res)
  res_df <- res_df[order(res_df$padj), ]
  
  # Clean up contrast name for filename
  clean_name <- gsub("[^a-zA-Z0-9_]", "_", contrast_name)
  output_file <- file.path(output_dir, paste0(clean_name, ".csv"))
  write.csv(res_df, output_file)
  cat("Saved:", output_file, "\\n")
  
  # Save significant genes for this contrast
  sig_genes <- res_df[which(res_df$padj < {self.config.analysis.fdr_threshold} & 
                             abs(res_df$log2FoldChange) > {self.config.analysis.log2fc_threshold}), ]
  if (nrow(sig_genes) > 0) {{
    sig_file <- file.path(output_dir, paste0(clean_name, "_significant.csv"))
    write.csv(sig_genes, sig_file)
    cat("Saved significant genes:", sig_file, "\\n")
  }}
  
  # Summary for this contrast
  n_sig <- sum(res_df$padj < {self.config.analysis.fdr_threshold}, na.rm=TRUE)
  n_up <- sum(res_df$padj < {self.config.analysis.fdr_threshold} & 
              res_df$log2FoldChange > {self.config.analysis.log2fc_threshold}, na.rm=TRUE)
  n_down <- sum(res_df$padj < {self.config.analysis.fdr_threshold} & 
                res_df$log2FoldChange < -{self.config.analysis.log2fc_threshold}, na.rm=TRUE)
  
  cat("  Total significant genes:", n_sig, "\\n")
  cat("  Upregulated:", n_up, "\\n")
  cat("  Downregulated:", n_down, "\\n")
}}

# Save normalized counts
norm_counts <- counts(dds, normalized=TRUE)
write.csv(norm_counts, file.path(output_dir, "normalized_counts.csv"))

# Overall Summary
cat("\\n=== DESeq2 Analysis Complete ===\\n")
cat("Total contrasts analyzed:", length(result_names), "\\n")
cat("Contrasts:", paste(result_names, collapse=", "), "\\n")
cat("Results saved with contrast-specific filenames\\n")

# Generate RNAseqQC plots
cat("\\n=== Generating RNAseqQC Plots ===\\n")

if (rnaseqqc_available) {{
  # 1. Total sample counts
  tryCatch({{
    cat("Generating Total Counts plot...\\n")
    png(file.path(qc_plots_dir, "total_counts.png"), width=800, height=600, res=100)
    print(plot_total_counts(dds) + ggtitle("Total sample counts") + theme_minimal())
    dev.off()
    cat("  ✓ Total Counts plot saved\\n")
  }}, error = function(e) {{
    cat("  ✗ Total Counts plot failed:", conditionMessage(e), "\\n")
  }})
  
  # 2. Library complexity
  tryCatch({{
    cat("Generating Library Complexity plot...\\n")
    png(file.path(qc_plots_dir, "library_complexity.png"), width=800, height=600, res=100)
    print(plot_library_complexity(dds) + ggtitle("Library complexity") + theme_minimal())
    dev.off()
    cat("  ✓ Library Complexity plot saved\\n")
  }}, error = function(e) {{
    cat("  ✗ Library Complexity plot failed:", conditionMessage(e), "\\n")
  }})
  
  # 3. Variance stabilization
  tryCatch({{
    cat("Generating Variance Stabilization plot...\\n")
    vsd <- vst(dds, blind=FALSE)
    png(file.path(qc_plots_dir, "variance_stabilization.png"), width=800, height=600, res=100)
    print(mean_sd_plot(vsd) + ggtitle("Variance stabilization") + theme_minimal())
    dev.off()
    cat("  ✓ Variance Stabilization plot saved\\n")
  }}, error = function(e) {{
    cat("  ✗ Variance Stabilization plot failed:", conditionMessage(e), "\\n")
  }})
  
  # 5. Sample clustering
  tryCatch({{
    cat("Generating Sample Clustering plot...\\n")
    if (!exists("vsd")) {{
      vsd <- vst(dds, blind=FALSE)
    }}
    set.seed(1)
    anno_vars <- names(metadata)
    cat("  Using annotation variables:", paste(anno_vars, collapse=", "), "\\n")
    png(file.path(qc_plots_dir, "sample_clustering.png"), width=1000, height=800, res=100)
    print(plot_sample_clustering(vsd, anno_vars = anno_vars, distance = "euclidean"))
    dev.off()
    cat("  ✓ Sample Clustering plot saved\\n")
  }}, error = function(e) {{
    cat("  ✗ Sample Clustering plot failed:", conditionMessage(e), "\\n")
  }})
  
  # 6. PCA scatters
  tryCatch({{
    cat("Generating PCA Scatters plot...\\n")
    if (!exists("vsd")) {{
      vsd <- vst(dds, blind=FALSE)
    }}
    anno_vars <- names(metadata)
    color_by <- if(length(anno_vars) >= 1) anno_vars[1] else NULL
    shape_by <- if(length(anno_vars) >= 2) anno_vars[2] else NULL
    
    if (!is.null(color_by)) {{
      cat("  Color by:", color_by, "\\n")
      if (!is.null(shape_by)) {{
        cat("  Shape by:", shape_by, "\\n")
        png(file.path(qc_plots_dir, "pca_scatters.png"), width=1200, height=1000, res=100)
        print(plot_pca_scatters(vsd, n_PCs = 5, color_by = color_by, shape_by = shape_by))
        dev.off()
      }} else {{
        png(file.path(qc_plots_dir, "pca_scatters.png"), width=1200, height=1000, res=100)
        print(plot_pca_scatters(vsd, n_PCs = 5, color_by = color_by))
        dev.off()
      }}
      cat("  ✓ PCA Scatters plot saved\\n")
    }} else {{
      cat("  ⚠ Skipping PCA Scatters (no categorical variables in metadata)\\n")
    }}
  }}, error = function(e) {{
    cat("  ✗ PCA Scatters plot failed:", conditionMessage(e), "\\n")
  }})
  
  cat("\\n✓ RNAseqQC plotting complete!\\n")
  cat("  Plots saved to:", qc_plots_dir, "\\n")
}}
}} else {{
  cat("Skipping RNAseqQC plots (package not available)\\n")
}}

# Generate standard DESeq2 plots for default contrast
cat("\\nGenerating standard DESeq2 plots for default contrast...\\n")

# MA plot
pdf(file.path(output_dir, "ma_plot.pdf"))
plotMA(res, main="MA Plot")
dev.off()

# Volcano plot
pdf(file.path(output_dir, "volcano_plot.pdf"))
with(res_df, plot(log2FoldChange, -log10(pvalue), 
     pch=20, main="Volcano Plot",
     xlab="log2 Fold Change", ylab="-log10(p-value)"))
with(subset(res_df, padj < {self.config.analysis.fdr_threshold}), 
     points(log2FoldChange, -log10(pvalue), pch=20, col="red"))
dev.off()

# PCA plot (may fail with small datasets)
cat("\\nGenerating PCA plot...\\n")
tryCatch({{
  vsd <- vst(dds, blind=FALSE)
  pdf(file.path(output_dir, "pca_plot.pdf"))
  print(plotPCA(vsd, intgroup=names(metadata)[1]))
  dev.off()
  cat("PCA plot generated successfully\\n")
}}, error = function(e) {{
  cat("Warning: PCA plot generation failed (may need more samples/genes):", conditionMessage(e), "\\n")
}})

cat("\\nAnalysis complete!\\n")
cat("Results saved to:", output_dir, "\\n")
"""
        
        return script


class MCPManager:
    """Manage multiple MCP servers"""
    
    def __init__(self, config):
        self.config = config
        self.servers = {}
        
        if config.use_mcp_server:
            self._initialize_servers()
    
    def _initialize_servers(self):
        """Initialize configured MCP servers"""
        if "r_transcriptomics" in self.config.mcp_servers:
            self.servers["r_transcriptomics"] = RTranscriptomicsMCPServer(
                self.config
            )
            logger.info("Initialized R Transcriptomics MCP Server")
    
    def start_all(self):
        """Start all MCP servers"""
        for name, server in self.servers.items():
            try:
                server.start()
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
    
    def stop_all(self):
        """Stop all MCP servers"""
        for name, server in self.servers.items():
            try:
                server.stop()
            except Exception as e:
                logger.error(f"Failed to stop {name}: {e}")
    
    def get_server(self, name: str) -> MCPServer:
        """Get MCP server by name"""
        return self.servers.get(name)
