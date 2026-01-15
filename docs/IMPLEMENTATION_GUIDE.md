"""
IMPLEMENTATION GUIDE: Enhanced Web Interface Features

This guide explains the changes needed to implement the three requested features.
The implementation is broken down into manageable steps.

======================================================================================
FEATURE 1: Progress Bars
======================================================================================

Add gr.Progress to show progress during long-running operations.

Changes needed in web_interface.py:

1. Update run_qc method signature:
   def run_qc(self, progress=gr.Progress()) -> str:
       progress(0, desc="Loading data...")
       # existing code
       progress(0.5, desc="Filtering low-count genes...")
       # existing code
       progress(1.0, desc="Complete!")

2. Update run_de_analysis method signature:
   def run_de_analysis(self, design_formula: str, use_r: bool, contrast: str, progress=gr.Progress()) -> str:
       progress(0, desc="Initializing DESeq2...")
       progress(0.3, desc="Estimating size factors...")
       progress(0.6, desc="Testing for differential expression...")
       progress(1.0, desc="Complete!")

3. Update run_enrichment method signature:
   def run_enrichment(self, progress=gr.Progress()) -> str:
       progress(0, desc="Preparing gene list...")
       progress(0.5, desc="Querying databases...")
       progress(1.0, desc="Complete!")

======================================================================================
FEATURE 2: Contrast Selection and DE Visualization
======================================================================================

Add UI for contrast selection and display DE results with plots.

New methods to add to BulkRNAWebInterface class:

```python
def detect_contrasts(self) -> List[str]:
    \"\"\"Detect possible contrasts from metadata\"\"\"
    if self.metadata_file is None:
        return []
    
    try:
        metadata = pd.read_csv(self.metadata_file, index_col=0)
        contrasts = []
        
        for col in metadata.columns:
            unique_vals = metadata[col].unique()
            if 2 <= len(unique_vals) <= 10:
                for i, val1 in enumerate(unique_vals):
                    for val2 in unique_vals[i+1:]:
                        contrasts.append(f"{col}: {val1} vs {val2}")
        
        return contrasts if contrasts else ["No contrasts available"]
    except:
        return ["Error detecting contrasts"]

def get_contrast_dropdown_choices(self):
    \"\"\"Update contrast dropdown after metadata upload\"\"\"
    contrasts = self.detect_contrasts()
    return gr.Dropdown(choices=contrasts, value=contrasts[0] if contrasts else None)

def load_de_results_for_contrast(self, contrast_name: str):
    \"\"\"Load and display DE results for selected contrast\"\"\"
    try:
        # Parse contrast name: "factor: level1 vs level2"
        parts = contrast_name.split(": ")
        if len(parts) != 2:
            return None, None, "Invalid contrast format"
        
        factor = parts[0]
        levels = parts[1].split(" vs ")
        
        # Load results file
        results_file = Path(self.config.data.output_dir) / "de_analysis" / f"de_results_{factor}_{levels[0]}_vs_{levels[1]}.csv"
        
        if not results_file.exists():
            return None, None, f"Results not found for {contrast_name}"
        
        df = pd.read_csv(results_file, index_col=0)
        
        # Create plots
        volcano_html = self.create_volcano_plot(df, contrast_name)
        ma_html = self.create_ma_plot(df, contrast_name)
        table_html = self.format_top_genes(df)
        
        return volcano_html, ma_html, table_html
    except Exception as e:
        return None, None, f"Error loading results: {e}"

def create_volcano_plot(self, df: pd.DataFrame, title: str) -> str:
    \"\"\"Create volcano plot HTML\"\"\"
    # Implementation using plotly (see web_interface_enhanced.py)
    pass

def create_ma_plot(self, df: pd.DataFrame, title: str) -> str:
    \"\"\"Create MA plot HTML\"\"\"
    # Implementation using plotly (see web_interface_enhanced.py)
    pass

def format_top_genes(self, df: pd.DataFrame, n: int = 50) -> str:
    \"\"\"Format top genes as HTML table\"\"\"
    top_df = df.nsmallest(n, 'padj')[['baseMean', 'log2FoldChange', 'pvalue', 'padj']]
    return top_df.to_html(classes='table table-striped')
```

UI Changes in create_interface():

In the Differential Expression tab, add:
```python
with gr.Tab("🔬 Differential Expression"):
    gr.Markdown("### Select Contrasts and Run DE Analysis")
    
    with gr.Row():
        contrast_dropdown = gr.Dropdown(
            label="Select Contrast",
            choices=[],
            interactive=True
        )
        refresh_contrasts_btn = gr.Button("🔄 Detect Contrasts")
    
    with gr.Row():
        design_input = gr.Textbox(...)
        use_r_checkbox = gr.Checkbox(...)
    
    run_de_btn = gr.Button("Run DE Analysis")
    de_output = gr.Textbox(...)
    
    # Results visualization
    gr.Markdown("### DE Results")
    with gr.Tabs():
        with gr.Tab("Volcano Plot"):
            volcano_plot = gr.HTML()
        with gr.Tab("MA Plot"):
            ma_plot = gr.HTML()
        with gr.Tab("Top Genes"):
            top_genes_table = gr.HTML()
    
    # Wire up callbacks
    refresh_contrasts_btn.click(
        fn=self.get_contrast_dropdown_choices,
        outputs=contrast_dropdown
    )
    
    contrast_dropdown.change(
        fn=self.load_de_results_for_contrast,
        inputs=[contrast_dropdown],
        outputs=[volcano_plot, ma_plot, top_genes_table]
    )
```

======================================================================================
FEATURE 3: Fix Enrichment Output and Display
======================================================================================

1. Fix enrichment tool to ensure files are saved (tools.py):

In EnrichmentAnalysisTool.execute():
```python
# Save results - ADD THIS
output_dir = Path(self.config.data.output_dir) / "enrichment"
output_dir.mkdir(parents=True, exist_ok=True)

all_results = []
for db in databases:
    logger.info(f"Querying {database}")
    enrich_results = self._query_enrichr(gene_list, db)
    
    if enrich_results:
        df = pd.DataFrame(enrich_results)
        output_file = output_dir / f"enrichment_{db}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved enrichment results to {output_file}")
        all_results.append(df)

return {
    "status": "success",
    "output_dir": str(output_dir),
    "databases": databases,
    "n_genes": len(gene_list),
    "results_files": [str(output_dir / f"enrichment_{db}.csv") for db in databases]
}
```

2. Add enrichment display to web interface:

New method:
```python
def display_enrichment_results(self) -> str:
    \"\"\"Display enrichment results as HTML tables\"\"\"
    if self.enrichment_results is None or self.enrichment_results.get('status') != 'success':
        return "<p>No enrichment results available</p>"
    
    output_dir = Path(self.enrichment_results['output_dir'])
    databases = self.enrichment_results.get('databases', [])
    
    html = "<h3>Enrichment Analysis Results</h3>"
    
    for db in databases:
        file_path = output_dir / f"enrichment_{db}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if not df.empty:
                # Show top 20 results
                top_df = df.head(20)
                html += f"<h4>{db}</h4>"
                html += top_df.to_html(classes='table table-striped', index=False)
        else:
            html += f"<p>No results for {db}</p>"
    
    return html
```

UI Changes in Enrichment tab:
```python
with gr.Tab("📊 Enrichment Analysis"):
    gr.Markdown("### Gene Set Enrichment Analysis")
    
    run_enrich_btn = gr.Button("Run Enrichment")
    enrich_output = gr.Textbox(...)
    
    # Add results display
    gr.Markdown("### Enrichment Results")
    enrichment_results_html = gr.HTML()
    
    # Wire up
    run_enrich_btn.click(
        fn=self.run_enrichment,
        outputs=[enrich_output, enrichment_results_html]
    )
```

Update run_enrichment to return both text and HTML:
```python
def run_enrichment(self, progress=gr.Progress()):
    # ... existing code ...
    
    if result["status"] == "success":
        results_html = self.display_enrichment_results()
        return success_message, results_html
    else:
        return error_message, ""
```

======================================================================================
INSTALLATION REQUIREMENTS
======================================================================================

Add to requirements.txt:
- plotly>=5.17.0  (already included)

No additional dependencies needed!

======================================================================================
TESTING
======================================================================================

1. Test progress bars:
   - Upload data
   - Watch for progress indicators during QC, DE, enrichment

2. Test contrast selection:
   - Upload metadata with multiple factors
   - Click "Detect Contrasts"
   - Select different contrasts from dropdown
   - Verify plots and tables update

3. Test enrichment display:
   - Run enrichment after DE
   - Check data/outputs/enrichment/ for CSV files
   - Verify results display in web interface

======================================================================================
"""
