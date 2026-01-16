"""
Gradio Web Interface for BulkRNA Agent
Provides file upload, chat interface, and visualization
"""
import gradio as gr
import logging
from pathlib import Path
import pandas as pd
import json
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .config import BulkRNAConfig
from .llm import DualLLMManager
from .agent import BulkRNAAgent
from .tools import (
    QualityControlTool,
    DifferentialExpressionTool,
    EnrichmentAnalysisTool,
    DesignMatrixSuggestionTool
)
from .mcp_server import MCPManager

logger = logging.getLogger(__name__)


class BulkRNAWebInterface:
    """Gradio web interface for BulkRNA Agent"""
    
    def __init__(self, config: Optional[BulkRNAConfig] = None):
        if config is None:
            config = BulkRNAConfig()
        
        self.config = config
        
        # Initialize components
        logger.info("Initializing BulkRNA Agent components...")
        self.llm_manager = DualLLMManager(config)
        
        # Initialize MCP servers
        self.mcp_manager = MCPManager(config)
        self.mcp_manager.start_all()
        
        # Initialize tools
        self.tools = {
            "quality_control": QualityControlTool(config),
            "differential_expression": DifferentialExpressionTool(config, self.mcp_manager),
            "enrichment_analysis": EnrichmentAnalysisTool(config),
            "design_matrix_suggestion": DesignMatrixSuggestionTool(config, self.llm_manager)
        }
        
        # Initialize agent
        self.agent = BulkRNAAgent(config, self.llm_manager, self.tools)
        
        # State
        self.counts_file = None
        self.metadata_file = None
        self.qc_results = None
        self.de_results = None
        self.available_contrasts = []
        self.current_contrast = None
        
        logger.info("BulkRNA Agent initialized successfully")
    
    def upload_counts(self, file) -> str:
        """Handle counts file upload"""
        try:
            if file is None:
                return "No file uploaded"
            
            # Save file
            file_path = Path(self.config.data.upload_dir) / Path(file.name).name
            
            # Copy file
            import shutil
            shutil.copy(file.name, file_path)
            
            self.counts_file = str(file_path)
            
            # Preview data
            df = pd.read_csv(file_path, index_col=0, nrows=5)
            
            logger.info(f"Uploaded counts file: {file_path}")
            
            return f"""
✅ Counts file uploaded successfully!

**File:** {Path(file.name).name}
**Dimensions:** {df.shape[0]}+ genes × {df.shape[1]} samples

**Preview:**
```
{df.to_string()}
```
"""
        except Exception as e:
            logger.error(f"Error uploading counts: {e}", exc_info=True)
            return f"❌ Error uploading file: {str(e)}"
    
    def upload_metadata(self, file) -> Tuple[str, gr.Dropdown, gr.Dropdown]:
        """Handle metadata file upload and return dropdown updates"""
        try:
            if file is None:
                return "No file uploaded", gr.Dropdown(), gr.Dropdown()
            
            # Save file
            file_path = Path(self.config.data.upload_dir) / Path(file.name).name
            
            import shutil
            shutil.copy(file.name, file_path)
            
            self.metadata_file = str(file_path)
            
            # Preview data
            df = pd.read_csv(file_path, index_col=0)
            
            logger.info(f"Uploaded metadata file: {file_path}")
            
            # Get column names for dropdowns
            col_names = list(df.columns)
            
            # Update dropdowns with column names
            color_dropdown = gr.Dropdown(choices=col_names, value=col_names[0] if col_names else None)
            shape_dropdown = gr.Dropdown(choices=col_names, value=col_names[1] if len(col_names) > 1 else None)
            
            status = f"""
✅ Metadata file uploaded successfully!

**File:** {Path(file.name).name}
**Samples:** {len(df)}
**Columns:** {', '.join(df.columns)}

**Preview:**
```
{df.to_string()}
```
"""
            return status, color_dropdown, shape_dropdown
        except Exception as e:
            logger.error(f"Error uploading metadata: {e}", exc_info=True)
            return f"❌ Error uploading file: {str(e)}", gr.Dropdown(), gr.Dropdown()
    
    def run_qc(self, progress=gr.Progress()) -> Tuple[str, str]:
        """Run quality control"""
        try:
            if self.counts_file is None:
                return "❌ Please upload counts file first", ""
            
            progress(0, desc="Starting QC analysis...")
            logger.info("Running QC analysis...")
            
            progress(0.3, desc="Loading count data...")
            tool = self.tools["quality_control"]
            
            progress(0.5, desc="Filtering low-count genes...")
            result = tool.execute(
                counts_file=self.counts_file,
                metadata_file=self.metadata_file,
                min_counts=self.config.analysis.min_count_threshold
            )
            
            progress(0.9, desc="Calculating metrics...")
            self.qc_results = result
            progress(1.0, desc="QC complete!")
            
            if result["status"] == "success":
                metrics = result["metrics"]
                
                response = f"""
✅ **Quality Control Complete**

**Summary:**
- Total genes (before filtering): {result['n_genes_before']}
- Total genes (after filtering): {result['n_genes_after']}
- Genes removed: {result['n_genes_before'] - result['n_genes_after']}
- Number of samples: {result['n_samples']}

**Library Sizes:**
- Median: {metrics.get('median_library_size', 'N/A'):,.0f} reads

**Per-Sample Metrics:**
"""
                for sample, sample_metrics in metrics.get('sample_metrics', {}).items():
                    response += f"\n- **{sample}**: {sample_metrics['total_counts']:,} reads, {sample_metrics['detected_genes']:,} genes"
                
                response += f"\n\n📁 **Filtered counts saved to:** `{result['filtered_counts_path']}`"
                
                # Generate PCA plot
                pca_html = self._create_pca_plot()
                
                return response, pca_html
            else:
                return f"❌ QC failed: {result.get('message', 'Unknown error')}", ""
                
        except Exception as e:
            logger.error(f"Error in QC: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", ""
    
    def _create_pca_plot(self) -> str:
        """Create PCA plot from QC results"""
        try:
            logger.info("Starting PCA plot generation...")
            
            if self.qc_results is None or self.metadata_file is None:
                logger.warning(f"Missing data for PCA: qc_results={self.qc_results is not None}, metadata={self.metadata_file is not None}")
                return "<p>Run QC and upload metadata first to see PCA plot</p>"
            
            # Load filtered counts
            filtered_counts_path = self.qc_results.get('filtered_counts_path')
            logger.info(f"Filtered counts path: {filtered_counts_path}")
            
            if not filtered_counts_path or not Path(filtered_counts_path).exists():
                logger.error(f"Filtered counts file not found at: {filtered_counts_path}")
                return "<p>Filtered counts file not found</p>"
            
            # Load data
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            counts = pd.read_csv(filtered_counts_path, index_col=0)
            metadata = pd.read_csv(self.metadata_file, index_col=0)
            logger.info(f"Loaded counts: {counts.shape}, metadata: {metadata.shape}")
            
            # Match samples
            common_samples = list(set(counts.columns) & set(metadata.index))
            logger.info(f"Found {len(common_samples)} common samples")
            
            if len(common_samples) < 2:
                logger.warning("Not enough samples for PCA")
                return "<p>Need at least 2 samples for PCA</p>"
            
            counts = counts[common_samples]
            metadata = metadata.loc[common_samples]
            
            # Log transform and transpose
            log_counts = np.log2(counts + 1).T
            
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(log_counts)
            
            # PCA
            pca = PCA(n_components=min(2, len(common_samples)))
            pca_result = pca.fit_transform(scaled_data)
            logger.info(f"PCA complete. Result shape: {pca_result.shape}, explained variance: {pca.explained_variance_ratio_}")
            
            # Get color by first metadata column
            color_column = metadata.columns[0] if len(metadata.columns) > 0 else None
            if color_column:
                colors = metadata[color_column].astype(str)
                color_map = {val: idx for idx, val in enumerate(colors.unique())}
                color_values = [color_map[c] for c in colors]
            else:
                colors = ['sample'] * len(common_samples)
                color_values = [0] * len(common_samples)
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1] if pca_result.shape[1] > 1 else [0] * len(pca_result),
                mode='markers+text',
                text=common_samples,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
            
            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100 if len(pca.explained_variance_ratio_) > 1 else 0
            
            fig.update_layout(
                title='PCA Plot - Sample Clustering',
                xaxis_title=f'PC1 ({pc1_var:.1f}% variance)',
                yaxis_title=f'PC2 ({pc2_var:.1f}% variance)',
                height=500,
                width=800,
                template='plotly_white',
                hovermode='closest'
            )
            
            # Generate unique div ID and use iframe rendering for Gradio compatibility
            import uuid
            import base64
            
            # Create complete standalone HTML
            plot_html = fig.to_html(include_plotlyjs='cdn', full_html=True, config={'responsive': True})
            
            # Encode as base64 for iframe
            html_bytes = plot_html.encode('utf-8')
            html_b64 = base64.b64encode(html_bytes).decode('utf-8')
            
            logger.info(f"PCA plot generated successfully, HTML length: {len(plot_html)}")
            
            # Return iframe with base64 encoded HTML
            return f'<iframe src="data:text/html;base64,{html_b64}" width="100%" height="550" frameborder="0"></iframe>'
            
        except Exception as e:
            logger.error(f"Error creating PCA plot: {e}", exc_info=True)
            return f"<p>Could not generate PCA plot: {str(e)}</p>"
    
    def suggest_design(self) -> str:
        """Suggest design matrix"""
        try:
            if self.metadata_file is None:
                return "❌ Please upload metadata file first"
            
            logger.info("Suggesting design matrix...")
            
            return self.agent.suggest_design_matrix(self.metadata_file)
            
        except Exception as e:
            logger.error(f"Error suggesting design: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def _extract_contrasts(self, design_formula: str = None) -> List[str]:
        """Extract available contrasts from DESeq2 output file"""
        try:
            # Check if contrasts file exists from DESeq2 run
            contrasts_file = Path(self.config.data.output_dir) / "de_analysis" / "available_contrasts.txt"
            
            if contrasts_file.exists():
                with open(contrasts_file, 'r') as f:
                    contrasts = [line.strip() for line in f if line.strip()]
                logger.info(f"Found {len(contrasts)} contrasts from DESeq2")
                return contrasts if contrasts else ["Intercept"]
            else:
                logger.warning("Contrasts file not found, using default")
                return ["Intercept"]
            
        except Exception as e:
            logger.error(f"Error extracting contrasts: {e}")
            return ["Intercept"]
    
    def run_de_analysis(self, design_formula: str, use_r: bool, progress=gr.Progress()) -> str:
        """Run differential expression analysis"""
        try:
            if self.qc_results is None:
                return "❌ Please run QC first"
            
            if self.metadata_file is None:
                return "❌ Please upload metadata file"
            
            if not design_formula.strip():
                return "❌ Please provide a design formula"
            
            progress(0, desc="Initializing DESeq2...")
            logger.info(f"Running DE analysis with design: {design_formula}")
            
            progress(0.2, desc="Loading filtered counts...")
            tool = self.tools["differential_expression"]
            
            # Use filtered counts from QC
            counts_file = self.qc_results.get("filtered_counts_path", self.counts_file)
            
            progress(0.4, desc="Estimating size factors...")
            result = tool.execute(
                counts_file=counts_file,
                metadata_file=self.metadata_file,
                design_formula=design_formula,
                use_mcp=use_r
            )
            
            progress(0.9, desc="Generating results...")
            self.de_results = result
            
            # Extract available contrasts from DESeq2 output
            self.available_contrasts = self._extract_contrasts()
            if self.available_contrasts:
                self.current_contrast = self.available_contrasts[0]
            
            progress(1.0, desc="DE analysis complete!")
            
            if result["status"] == "success":
                contrasts_info = ""
                if len(self.available_contrasts) > 1:
                    contrasts_info = f"\n\n**Available Contrasts ({len(self.available_contrasts)}):**\n"
                    for contrast in self.available_contrasts:
                        contrasts_info += f"- `{contrast}`\n"
                
                response = f"""
✅ **Differential Expression Analysis Complete**

**Design Formula:** `{result.get('design_formula', design_formula)}`

**Results:**
- Total significant genes: {result['n_significant']}
- Upregulated: {result['n_upregulated']}
- Downregulated: {result['n_downregulated']}

**Thresholds:**
- FDR < {self.config.analysis.fdr_threshold}
- |log2FC| > {self.config.analysis.log2fc_threshold}
{contrasts_info}
📁 **Results saved to:** `{result['results_path']}`
📁 **Significant genes:** `{result['significant_genes_path']}`

💡 **Tip:** Select different contrasts from the dropdown to view specific comparisons.
"""
                return response
            else:
                return f"❌ DE analysis failed: {result.get('message', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in DE analysis: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
    def run_enrichment(self, contrast: str = None, progress=gr.Progress()) -> Tuple[str, gr.Dropdown, str]:
        """Run enrichment analysis for selected contrast"""
        try:
            progress(0, desc="Preparing gene list...")
            if self.de_results is None:
                return "❌ Please run differential expression analysis first", gr.Dropdown(choices=[]), ""
            
            # Use selected contrast or first available
            if not contrast and self.available_contrasts:
                contrast = self.available_contrasts[0]
            
            if not contrast:
                return "❌ No contrasts available", gr.Dropdown(choices=[]), ""
            
            logger.info(f"Running enrichment analysis for contrast: {contrast}")
            
            # Load significant genes from contrast-specific file
            clean_name = contrast.replace(" ", "_").replace("/", "_")
            clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
            sig_genes_file = f"{clean_name}_significant.csv"
            sig_genes_path = Path(self.config.data.output_dir) / "de_analysis" / sig_genes_file
            
            if not sig_genes_path.exists():
                return f"❌ Significant genes file not found for contrast: {contrast}", gr.Dropdown(choices=[]), ""
            
            df = pd.read_csv(sig_genes_path, index_col=0)
            gene_list = df.index.tolist()
            
            if len(gene_list) == 0:
                return f"❌ No significant genes found for contrast: {contrast}", gr.Dropdown(choices=[]), ""
            
            progress(0.3, desc="Querying enrichment databases...")
            tool = self.tools["enrichment_analysis"]
            result = tool.execute(gene_list=gene_list)
            
            progress(0.9, desc="Formatting results...")
            self.enrichment_results = result
            
            if result["status"] == "success":
                progress(1.0, desc="Enrichment complete!")
                
                # Check if we got any results
                has_results = any(len(r) > 0 for r in result.get('results', {}).values())
                
                # Get list of databases with results
                db_list = [db for db, data in result.get('results', {}).items() if len(data) > 0]
                
                if has_results:
                    message = f"""
✅ **Enrichment Analysis Complete**

**Contrast:** `{contrast}`
**Genes analyzed:** {result['n_genes']}
**Databases queried:** {', '.join(result['databases'])}
**Results saved to:** `{result['output_dir']}`
"""
                    # Update dropdown with available databases
                    db_dropdown = gr.Dropdown(choices=db_list, value=db_list[0] if db_list else None)
                    # Generate HTML display for first database
                    html_results = self._format_single_database_html(result, db_list[0] if db_list else None)
                    return message, db_dropdown, html_results
                else:
                    message = f"""
⚠️ **Enrichment Analysis Complete - No Significant Terms Found**

**Contrast:** `{contrast}`
**Genes analyzed:** {result['n_genes']}
**Databases queried:** {', '.join(result['databases'])}

No significantly enriched terms were found in any database. This could mean:
- The gene set is too small
- The genes don't cluster into known pathways
- Try adjusting your DE thresholds to get more genes
"""
                    return message, gr.Dropdown(choices=[]), ""
            else:
                return f"❌ Enrichment analysis failed: {result.get('message', 'Unknown error')}", gr.Dropdown(choices=[]), ""
                
        except Exception as e:
            logger.error(f"Error in enrichment: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", gr.Dropdown(choices=[]), ""
    
    def display_enrichment_database(self, database: str) -> str:
        """Display results for a specific enrichment database"""
        try:
            if self.enrichment_results is None:
                return "<p>No enrichment results available. Please run enrichment analysis first.</p>"
            
            if database is None:
                return "<p>Please select a database to view results.</p>"
            
            return self._format_single_database_html(self.enrichment_results, database)
            
        except Exception as e:
            logger.error(f"Error displaying database {database}: {e}", exc_info=True)
            return f"<p>Error displaying results: {str(e)}</p>"
    
    def _format_single_database_html(self, results: Dict[str, Any], database: str) -> str:
        """Format enrichment results for a single database as HTML"""
        try:
            if 'results' not in results or database not in results['results']:
                return "<p>No results available for this database</p>"
            
            enrich_data = results['results'][database]
            
            if not enrich_data:
                return f"<p>No enrichment terms found in {database}</p>"
            
            df = pd.DataFrame(enrich_data)
            if df.empty:
                return f"<p>No enrichment terms found in {database}</p>"
            
            # Ensure sorted by p-value (ascending - most significant first)
            if 'p_value' in df.columns:
                df = df.sort_values('p_value')
            
            # Format columns for better display
            display_df = df.copy()
            
            if 'adjusted_p_value' in display_df.columns:
                display_df['Adjusted P-value'] = display_df['adjusted_p_value'].apply(lambda x: f"{x:.2e}")
                display_df = display_df.drop('adjusted_p_value', axis=1)
            if 'p_value' in display_df.columns:
                display_df['P-value'] = display_df['p_value'].apply(lambda x: f"{x:.2e}")
                display_df = display_df.drop('p_value', axis=1)
            if 'combined_score' in display_df.columns:
                display_df['Combined Score'] = display_df['combined_score'].apply(lambda x: f"{x:.2f}")
                display_df = display_df.drop('combined_score', axis=1)
            
            # Rename columns for clarity
            if 'term' in display_df.columns:
                display_df = display_df.rename(columns={'term': 'Term'})
            if 'genes' in display_df.columns:
                display_df = display_df.rename(columns={'genes': 'Genes'})
            
            # Create HTML with styling
            html = f"""
            <style>
                .enrich-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 12px;
                }}
                .enrich-table th {{
                    background: #3498db;
                    color: white;
                    padding: 12px 8px;
                    text-align: left;
                    border-bottom: 2px solid #2980b9;
                    position: sticky;
                    top: 0;
                    font-weight: 600;
                    z-index: 10;
                }}
                .enrich-table td {{
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                    background: white;
                }}
                .enrich-table tr:hover td {{
                    background: #f0f8ff;
                }}
            </style>
            <h3>{database} ({len(display_df)} terms, sorted by P-value)</h3>
            <div style="max-height: 600px; overflow-y: auto;">
            """
            
            html += display_df.to_html(
                classes='enrich-table',
                border=0,
                index=False,
                escape=False
            )
            
            html += "</div>"
            
            return html
            
        except Exception as e:
            logger.error(f"Error formatting database {database}: {e}", exc_info=True)
            return f"<p>Error formatting results: {str(e)}</p>"
    
    def _format_enrichment_html(self, results: Dict[str, Any]) -> str:
        """Format enrichment results as HTML with tabs for each database"""
        try:
            if 'results' not in results:
                return "<p>No detailed results available</p>"
            
            # Debug: Log what databases we have and their sizes
            logger.info(f"Formatting enrichment results for databases: {list(results['results'].keys())}")
            for db, data in results['results'].items():
                logger.info(f"  {db}: {len(data)} results")
            
            # Check if any database returned results
            has_results = any(len(r) > 0 for r in results['results'].values())
            
            if not has_results:
                return """
                <div style='padding: 20px; background-color: #f8f9fa; border-radius: 8px;'>
                    <h3>No Enrichment Terms Returned</h3>
                    <p>The Enrichr API did not return any terms for the queried databases.</p>
                    <p><strong>Possible reasons:</strong></p>
                    <ul>
                        <li>Gene list may be too small (recommended: >20 genes)</li>
                        <li>Gene symbols may not match database nomenclature</li>
                        <li>Try a different contrast with more significant genes</li>
                    </ul>
                </div>
                """
            
            # Create tabbed interface using CSS-only approach with radio buttons
            html = """
            <style>
                .enrich-tab-container {
                    width: 100%;
                }
                .enrich-tab-radio {
                    display: none;
                }
                .enrich-tabs {
                    display: flex;
                    border-bottom: 2px solid #ddd;
                    margin-bottom: 20px;
                    flex-wrap: wrap;
                }
                .enrich-tab-label {
                    padding: 10px 20px;
                    cursor: pointer;
                    background: #f8f9fa;
                    margin-right: 2px;
                    border-radius: 5px 5px 0 0;
                    user-select: none;
                    transition: background 0.2s;
                }
                .enrich-tab-label:hover {
                    background: #e9ecef;
                }
                .enrich-tab-radio:checked + .enrich-tab-label {
                    background: white;
                    font-weight: bold;
                    border-bottom: 2px solid white;
                }
                .enrich-content {
                    display: none;
                }
                .enrich-tab-radio:checked ~ .enrich-tab-contents .enrich-content[data-tab] {
                    display: none;
                }
                .enrich-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 12px;
                }
                .enrich-table th {
                    background: #3498db;
                    color: white;
                    padding: 12px 8px;
                    text-align: left;
                    border-bottom: 2px solid #2980b9;
                    position: sticky;
                    top: 0;
                    font-weight: 600;
                    z-index: 10;
                }
                .enrich-table td {
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                    background: white;
                }
                .enrich-table tr:hover td {
                    background: #f0f8ff;
                }
            </style>
            <div class="enrich-tab-container">
            <div class="enrich-tabs">
            """
            
            # Create tabs with radio buttons
            db_list = [db for db, data in results['results'].items() if len(data) > 0]
            for idx, database in enumerate(db_list):
                safe_db_name = database.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                checked = 'checked' if idx == 0 else ''
                html += f'''
                <input type="radio" id="tab_{safe_db_name}" name="enrich-tabs" class="enrich-tab-radio" {checked}>
                <label for="tab_{safe_db_name}" class="enrich-tab-label">{database}</label>
                '''
            
            html += "</div><div class='enrich-tab-contents'>"
            
            # Create content for each database - USE THE SAME db_list TO ENSURE CONSISTENCY
            for idx, database in enumerate(db_list):
                enrich_data = results['results'][database]
                
                # Debug log
                logger.info(f"Creating content for {database}: {len(enrich_data)} results")
                if enrich_data:
                    first_term = enrich_data[0].get('term', 'N/A') if enrich_data else 'N/A'
                    logger.info(f"  First term for {database}: {first_term}")
                
                df = pd.DataFrame(enrich_data)
                if df.empty:
                    continue
                
                safe_db_name = database.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                
                # Show ALL results (not just top 50), already sorted by p-value from tool
                display_df = df.copy()
                
                # Ensure sorted by p-value (ascending - most significant first)
                if 'p_value' in display_df.columns:
                    display_df = display_df.sort_values('p_value')
                
                # Format columns for better display
                if 'adjusted_p_value' in display_df.columns:
                    display_df['Adjusted P-value'] = display_df['adjusted_p_value'].apply(lambda x: f"{x:.2e}")
                    display_df = display_df.drop('adjusted_p_value', axis=1)
                if 'p_value' in display_df.columns:
                    display_df['P-value'] = display_df['p_value'].apply(lambda x: f"{x:.2e}")
                    display_df = display_df.drop('p_value', axis=1)
                if 'combined_score' in display_df.columns:
                    display_df['Combined Score'] = display_df['combined_score'].apply(lambda x: f"{x:.2f}")
                    display_df = display_df.drop('combined_score', axis=1)
                
                # Rename columns for clarity
                if 'term' in display_df.columns:
                    display_df = display_df.rename(columns={'term': 'Term'})
                if 'genes' in display_df.columns:
                    display_df = display_df.rename(columns={'genes': 'Genes'})
                
                html += f'<div class="enrich-content" data-tab="{safe_db_name}" style="display: {"block" if idx == 0 else "none"};">'
                html += f"<h3>{database} ({len(display_df)} terms, sorted by P-value)</h3>"
                html += f'<p style="font-size: 10px; color: #666;">DEBUG: Database={database}, SafeName={safe_db_name}, First term={display_df.iloc[0]["Term"] if "Term" in display_df.columns and len(display_df) > 0 else "N/A"}</p>'
                html += '<div style="max-height: 600px; overflow-y: auto;">'
                html += display_df.to_html(
                    classes='enrich-table',
                    border=0,
                    index=False,
                    escape=False
                )
                html += "</div></div>"
            
            html += "</div></div>"
            
            # Add JavaScript to handle tab switching (fallback if CSS doesn't work)
            html += """
            <script>
            (function() {
                var radios = document.querySelectorAll('input[name="enrich-tabs"]');
                radios.forEach(function(radio) {
                    radio.addEventListener('change', function() {
                        // Hide all content
                        var contents = document.querySelectorAll('.enrich-content');
                        contents.forEach(function(content) {
                            content.style.display = 'none';
                        });
                        
                        // Show the selected content
                        var tabId = this.id.replace('tab_', '');
                        var targetContent = document.querySelector('.enrich-content[data-tab="' + tabId + '"]');
                        if (targetContent) {
                            targetContent.style.display = 'block';
                        }
                    });
                });
            })();
            </script>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error formatting enrichment HTML: {e}")
            return f"<p>Error formatting results: {str(e)}</p>"
    
    def _create_volcano_plot(self, contrast_name: str = None) -> str:
        """Create volcano plot from DE results"""
        try:
            if self.de_results is None or self.de_results.get('status') != 'success':
                return "<p>No DE results available</p>"
            
            # Get the results file for the selected contrast
            if contrast_name:
                clean_name = contrast_name.replace(" ", "_").replace("/", "_")
                # Remove any special characters
                clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                results_file = f"{clean_name}.csv"
                results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
            else:
                # Fallback to first available contrast
                if self.available_contrasts:
                    clean_name = self.available_contrasts[0].replace(" ", "_").replace("/", "_")
                    clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                    results_file = f"{clean_name}.csv"
                    results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
                else:
                    return "<p>No contrast results available</p>"
            
            if not results_path.exists():
                return f"<p>Results file not found: {results_file}</p>"
            
            df = pd.read_csv(results_path, index_col=0)
            
            # Create plotly volcano plot
            import plotly.graph_objects as go
            import numpy as np
            
            df['significant'] = (
                (df['padj'] < self.config.analysis.fdr_threshold) &
                (df['log2FoldChange'].abs() > self.config.analysis.log2fc_threshold)
            )
            df['direction'] = 'Not Significant'
            df.loc[(df['significant']) & (df['log2FoldChange'] > 0), 'direction'] = 'Upregulated'
            df.loc[(df['significant']) & (df['log2FoldChange'] < 0), 'direction'] = 'Downregulated'
            
            fig = go.Figure()
            
            colors = {'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'lightgray'}
            
            for direction in ['Not Significant', 'Downregulated', 'Upregulated']:
                df_subset = df[df['direction'] == direction]
                fig.add_trace(go.Scatter(
                    x=df_subset['log2FoldChange'],
                    y=-np.log10(df_subset['pvalue'].clip(lower=1e-300)),
                    mode='markers',
                    name=direction,
                    marker=dict(color=colors[direction], size=4, opacity=0.6),
                    text=df_subset.index,
                    hovertemplate='<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10(p): %{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Volcano Plot - {contrast_name if contrast_name else "Default Contrast"}',
                xaxis_title='log2 Fold Change',
                yaxis_title='-log10(p-value)',
                hovermode='closest',
                height=500,
                template='plotly_white'
            )
            
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", annotation_text="p=0.05")
            fig.add_vline(x=self.config.analysis.log2fc_threshold, line_dash="dash", line_color="gray")
            fig.add_vline(x=-self.config.analysis.log2fc_threshold, line_dash="dash", line_color="gray")
            
            # Use iframe rendering for Gradio compatibility
            import base64
            plot_html = fig.to_html(include_plotlyjs='cdn', full_html=True, config={'responsive': True})
            html_b64 = base64.b64encode(plot_html.encode('utf-8')).decode('utf-8')
            
            return f'<iframe src="data:text/html;base64,{html_b64}" width="100%" height="550" frameborder="0"></iframe>'
            
        except Exception as e:
            logger.error(f"Error creating volcano plot: {e}")
            return f"<p>Error creating plot: {str(e)}</p>"
    
    def _create_ma_plot(self, contrast_name: str = None) -> str:
        """Create MA plot from DE results"""
        try:
            if self.de_results is None or self.de_results.get('status') != 'success':
                return "<p>No DE results available</p>"
            
            # Get the results file for the selected contrast
            if contrast_name:
                clean_name = contrast_name.replace(" ", "_").replace("/", "_")
                clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                results_file = f"{clean_name}.csv"
                results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
            else:
                # Fallback to first available contrast
                if self.available_contrasts:
                    clean_name = self.available_contrasts[0].replace(" ", "_").replace("/", "_")
                    clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                    results_file = f"{clean_name}.csv"
                    results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
                else:
                    return "<p>No contrast results available</p>"
            
            if not results_path.exists():
                return f"<p>Results file not found: {results_file}</p>"
            
            df = pd.read_csv(results_path, index_col=0)
            
            # Create plotly MA plot
            import plotly.graph_objects as go
            import numpy as np
            
            df['significant'] = (
                (df['padj'] < self.config.analysis.fdr_threshold) &
                (df['log2FoldChange'].abs() > self.config.analysis.log2fc_threshold)
            )
            
            fig = go.Figure()
            
            # Non-significant genes
            df_ns = df[~df['significant']]
            fig.add_trace(go.Scatter(
                x=np.log10(df_ns['baseMean'].clip(lower=1)),
                y=df_ns['log2FoldChange'],
                mode='markers',
                name='Not Significant',
                marker=dict(color='lightgray', size=3, opacity=0.4),
                text=df_ns.index,
                hovertemplate='<b>%{text}</b><br>baseMean: %{x:.1f}<br>log2FC: %{y:.2f}<extra></extra>'
            ))
            
            # Significant genes
            df_sig = df[df['significant']]
            fig.add_trace(go.Scatter(
                x=np.log10(df_sig['baseMean'].clip(lower=1)),
                y=df_sig['log2FoldChange'],
                mode='markers',
                name='Significant',
                marker=dict(color='red', size=4, opacity=0.7),
                text=df_sig.index,
                hovertemplate='<b>%{text}</b><br>baseMean: %{x:.1f}<br>log2FC: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'MA Plot - {contrast_name if contrast_name else "Default Contrast"}',
                xaxis_title='log10(baseMean)',
                yaxis_title='log2 Fold Change',
                hovermode='closest',
                height=500,
                template='plotly_white'
            )
            
            fig.add_hline(y=self.config.analysis.log2fc_threshold, line_dash="dash", line_color="gray")
            fig.add_hline(y=-self.config.analysis.log2fc_threshold, line_dash="dash", line_color="gray")
            fig.add_hline(y=0, line_color="black", line_width=0.5)
            
            # Use iframe rendering for Gradio compatibility
            import base64
            plot_html = fig.to_html(include_plotlyjs='cdn', full_html=True, config={'responsive': True})
            html_b64 = base64.b64encode(plot_html.encode('utf-8')).decode('utf-8')
            
            return f'<iframe src="data:text/html;base64,{html_b64}" width="100%" height="550" frameborder="0"></iframe>'
            
        except Exception as e:
            logger.error(f"Error creating MA plot: {e}")
            return f"<p>Error creating plot: {str(e)}</p>"
    
    def _format_top_genes(self, n: int = 50, contrast: str = None) -> str:
        """Format top DE genes as HTML table"""
        try:
            if self.de_results is None or self.de_results.get('status') != 'success':
                return "<p>No DE results available</p>"
            
            # Get the results file for the selected contrast
            if contrast:
                clean_name = contrast.replace(" ", "_").replace("/", "_")
                clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                results_file = f"{clean_name}.csv"
                results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
            else:
                # Fallback to first available contrast
                if self.available_contrasts:
                    clean_name = self.available_contrasts[0].replace(" ", "_").replace("/", "_")
                    clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)
                    results_file = f"{clean_name}.csv"
                    results_path = Path(self.config.data.output_dir) / "de_analysis" / results_file
                else:
                    return "<p>No contrast results available</p>"
            
            if not results_path.exists():
                return f"<p>Results file not found: {results_file}</p>"
            
            df = pd.read_csv(results_path, index_col=0)
            
            # Get top genes by adjusted p-value
            top_results = df.nsmallest(n, 'padj')
            
            # Format for display
            display_df = top_results[['baseMean', 'log2FoldChange', 'pvalue', 'padj']].copy()
            display_df['baseMean'] = display_df['baseMean'].apply(lambda x: f"{x:.1f}")
            display_df['log2FoldChange'] = display_df['log2FoldChange'].apply(lambda x: f"{x:.3f}")
            display_df['pvalue'] = display_df['pvalue'].apply(lambda x: f"{x:.2e}")
            display_df['padj'] = display_df['padj'].apply(lambda x: f"{x:.2e}")
            
            contrast_label = contrast if contrast else "Default Contrast"
            html = f"<h3>Top {len(top_results)} Genes - {contrast_label}</h3>"
            html += f"<div style='max-height: 500px; overflow-y: auto;'>"
            html += display_df.to_html(classes='dataframe', border=0, escape=False)
            html += "</div>"
            return html
            
        except Exception as e:
            logger.error(f"Error formatting top genes: {e}")
            return f"<p>Error formatting results: {str(e)}</p>"
    
    def refresh_de_visualizations(self, contrast: str = None) -> Tuple[str, str, str]:
        """Refresh all DE visualization plots and tables"""
        try:
            # Use the selected contrast or current contrast
            if contrast:
                self.current_contrast = contrast
            
            contrast_name = self.current_contrast if self.current_contrast else self.available_contrasts[0] if self.available_contrasts else None
            
            volcano = self._create_volcano_plot(contrast_name)
            ma = self._create_ma_plot(contrast_name)
            table = self._format_top_genes(contrast=contrast_name)
            return volcano, ma, table
        except Exception as e:
            logger.error(f"Error refreshing visualizations: {e}")
            error_msg = f"<p>Error: {str(e)}</p>"
            return error_msg, error_msg, error_msg
    
    def get_rnaseqqc_plots_html(self) -> Tuple[str, str, str, str, str]:
        """
        Get HTML for RNAseqQC plots
        
        Returns:
            Tuple of HTML strings for (total_counts, library_complexity,
                                      variance_stabilization, sample_clustering, pca_scatters)
        """
        try:
            # Get plot paths from DE tool
            tool = self.tools.get("differential_expression")
            if tool is None:
                error_msg = "<p>Differential Expression tool not available</p>"
                return tuple([error_msg] * 5)
            
            result = tool.get_rnaseqqc_plots()
            
            if result["status"] != "success":
                error_msg = f"<p>Error: {result.get('message', 'Unknown error')}</p>"
                return tuple([error_msg] * 5)
            
            plots = result["plots"]
            
            # Create HTML for each plot
            html_outputs = []
            plot_titles = {
                "total_counts": "Total Sample Counts",
                "library_complexity": "Library Complexity",
                "variance_stabilization": "Variance Stabilization",
                "sample_clustering": "Sample Clustering",
                "pca_scatters": "PCA Scatters (Multiple PCs)"
            }
            
            for plot_key in ["total_counts", "library_complexity",
                           "variance_stabilization", "sample_clustering", "pca_scatters"]:
                plot_path = plots.get(plot_key)
                if plot_path and Path(plot_path).exists():
                    # Convert image to base64 for embedding
                    import base64
                    with open(plot_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    html = f"""
                    <div style="text-align: center; padding: 10px;">
                        <h3>{plot_titles[plot_key]}</h3>
                        <img src="data:image/png;base64,{img_data}" 
                             style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;"/>
                    </div>
                    """
                    html_outputs.append(html)
                else:
                    html_outputs.append(f"""
                    <div style="text-align: center; padding: 20px; color: #666;">
                        <p>⚠️ {plot_titles[plot_key]} not available</p>
                        <p style="font-size: 0.9em;">This may be because:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>RNAseqQC R package is not installed</li>
                            <li>Gene IDs are not in ENSEMBL format</li>
                            <li>Differential expression analysis hasn't been run yet</li>
                        </ul>
                    </div>
                    """)
            
            return tuple(html_outputs)
            
        except Exception as e:
            logger.error(f"Error getting RNAseqQC plots: {e}", exc_info=True)
            error_msg = f"<p>Error loading plots: {str(e)}</p>"
            return tuple([error_msg] * 5)
    
    def get_metadata_columns(self) -> List[str]:
        """Get list of metadata column names"""
        try:
            if self.metadata_file is None:
                return []
            
            import pandas as pd
            metadata = pd.read_csv(self.metadata_file, index_col=0)
            return list(metadata.columns)
        except Exception as e:
            logger.error(f"Error getting metadata columns: {e}")
            return []
    
    def regenerate_pca_plot(self, color_by: str, shape_by: str) -> str:
        """Regenerate PCA scatter plot with selected variables"""
        try:
            if not color_by:
                return "<p>⚠️ Please select a variable for coloring</p>"
            
            # Get output directory
            output_dir = Path(self.config.data.output_dir) / "de_analysis"
            qc_plots_dir = output_dir / "qc_plots"
            
            if not qc_plots_dir.exists():
                return "<p>⚠️ Run DE analysis first to generate QC plots</p>"
            
            # Call MCP server to regenerate PCA plot with custom variables
            mcp_server = self.tools.get("differential_expression").mcp_server
            
            # Create R script to regenerate just the PCA plot
            r_script = f"""
library(DESeq2)
library(RNAseqQC)

# Load the saved dds object (if available) or reconstruct
output_dir <- "data/outputs/de_analysis"
qc_plots_dir <- file.path(output_dir, "qc_plots")

# Load count data
counts <- read.csv("data/outputs/qc/filtered_counts.csv", row.names=1, check.names=FALSE)
metadata <- read.csv("data/uploads/metadata.csv", row.names=1, check.names=FALSE, stringsAsFactors=TRUE)

# Ensure sample order matches
common_samples <- intersect(colnames(counts), rownames(metadata))
counts <- counts[, common_samples]
metadata <- metadata[common_samples, , drop=FALSE]

# Create DESeq2 dataset with simple design
dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = metadata,
  design = ~ 1
)

# Run DESeq2 (minimal)
dds <- DESeq(dds)

# Variance stabilization
vsd <- vst(dds, blind=FALSE)

# Generate PCA plot with custom variables
cat("Generating PCA Scatters plot...\\n")
color_by <- "{color_by}"
shape_by <- {'"' + shape_by + '"' if shape_by else 'NULL'}

tryCatch({{
  if (!is.null(shape_by) && shape_by != "") {{
    png(file.path(qc_plots_dir, "pca_scatters.png"), width=1200, height=1000, res=100)
    print(plot_pca_scatters(vsd, n_PCs = 5, color_by = color_by, shape_by = shape_by))
    dev.off()
  }} else {{
    png(file.path(qc_plots_dir, "pca_scatters.png"), width=1200, height=1000, res=100)
    print(plot_pca_scatters(vsd, n_PCs = 5, color_by = color_by))
    dev.off()
  }}
  cat("✓ PCA Scatters plot regenerated\\n")
}}, error = function(e) {{
  cat("Error:", conditionMessage(e), "\\n")
}})
"""
            
            # Execute R script
            import subprocess
            r_executable = mcp_server.r_path
            result = subprocess.run(
                [r_executable, "-e", r_script],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            if result.returncode != 0:
                logger.error(f"R script failed: {result.stderr}")
                return f"<p>❌ Error regenerating PCA plot: {result.stderr[-200:]}</p>"
            
            # Now load the regenerated plot
            pca_plot_path = qc_plots_dir / "pca_scatters.png"
            if pca_plot_path.exists():
                import base64
                with open(pca_plot_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                
                return f"""
                <div style="text-align: center; padding: 10px;">
                    <h3>PCA Scatters (Multiple PCs)</h3>
                    <p style="font-size: 0.9em;">Color by: <strong>{color_by}</strong>{f' | Shape by: <strong>{shape_by}</strong>' if shape_by else ''}</p>
                    <img src="data:image/png;base64,{img_data}" 
                         style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;"/>
                </div>
                """
            else:
                return "<p>❌ PCA plot file not found after regeneration</p>"
            
        except Exception as e:
            logger.error(f"Error regenerating PCA plot: {e}", exc_info=True)
            return f"<p>❌ Error: {str(e)}</p>"
    
    def chat_with_data(self, message: str, history: List) -> List:
        """Chat interface handler"""
        return self.chat(message, history)
    
    def chat(self, message: str, history: List) -> Tuple[str, List]:
        """Handle chat messages"""
        try:
            # Get response from agent
            response = self.agent.chat(message)
            
            # Update history with proper message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def reset_chat(self) -> List:
        """Reset chat history"""
        self.agent.reset()
        return []
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        
        # Create theme with standard font (will be passed to launch in Gradio 6.0+)
        self.theme = gr.themes.Default(
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Courier New", "monospace"]
        )
        
        with gr.Blocks(title="BulkRNA Agent") as interface:
            gr.Markdown("""
# 🧬 BulkRNA Agent
### AI-Powered Bulk RNA-seq Analysis

Upload your data, run analyses, and chat with the agent to explore your transcriptomics results.
            """)
            
            with gr.Tabs():
                # Tab 1: Data Upload
                with gr.Tab("📁 Data Upload"):
                    gr.Markdown("### Upload Expression Data and Metadata")
                    
                    with gr.Row():
                        with gr.Column():
                            counts_upload = gr.File(
                                label="Count Matrix (CSV/TSV)",
                                file_types=[".csv", ".tsv", ".txt"]
                            )
                            counts_status = gr.Markdown()
                            
                        with gr.Column():
                            metadata_upload = gr.File(
                                label="Sample Metadata (CSV/TSV)",
                                file_types=[".csv", ".tsv", ".txt"]
                            )
                            metadata_status = gr.Markdown()
                    
                    counts_upload.change(
                        fn=self.upload_counts,
                        inputs=[counts_upload],
                        outputs=[counts_status]
                    )
                
                # Tab 2: Quality Control
                with gr.Tab("🔍 Quality Control"):
                    gr.Markdown("### Run QC on uploaded data")
                    
                    qc_button = gr.Button("Run Quality Control", variant="primary")
                    qc_output = gr.Markdown()
                    
                    # PCA Plot Display
                    gr.Markdown("### PCA Plot")
                    pca_plot_html = gr.HTML(label="Sample Clustering")
                    
                    qc_button.click(
                        fn=self.run_qc,
                        inputs=[],
                        outputs=[qc_output, pca_plot_html],
                        show_progress="full"
                    )
                
                # Tab 3: Differential Expression
                with gr.Tab("📊 Differential Expression"):
                    gr.Markdown("### Design Matrix and DE Analysis")
                    
                    suggest_button = gr.Button("Suggest Design Matrix")
                    design_output = gr.Markdown()
                    
                    design_formula = gr.Textbox(
                        label="Design Formula",
                        placeholder="e.g., ~ condition",
                        value="~ condition"
                    )
                    
                    use_r = gr.Checkbox(
                        label="Use R DESeq2 (via MCP server)",
                        value=True,
                        info="If unchecked, uses PyDESeq2"
                    )
                    
                    de_button = gr.Button("Run Differential Expression", variant="primary")
                    de_output = gr.Markdown()
                    
                    # DE Results Visualization
                    gr.Markdown("### Differential Expression Results")
                    gr.Markdown("_Select a contrast to view specific comparison results_")
                    
                    with gr.Row():
                        contrast_dropdown = gr.Dropdown(
                            label="Select Contrast",
                            choices=["Intercept"],
                            value="Intercept",
                            info="Choose which contrast to visualize"
                        )
                        refresh_viz_button = gr.Button("🔄 Show/Refresh Visualizations", size="sm")
                    
                    with gr.Tabs():
                        with gr.Tab("Volcano Plot"):
                            volcano_plot_html = gr.HTML()
                        with gr.Tab("MA Plot"):
                            ma_plot_html = gr.HTML()
                        with gr.Tab("Top Genes"):
                            top_genes_html = gr.HTML()
                        
                        # RNAseqQC plots tabs
                        with gr.Tab("📊 QC: Total Counts"):
                            gr.Markdown("_Total read counts per sample (from RNAseqQC)_")
                            qc_total_counts_html = gr.HTML()
                        with gr.Tab("📊 QC: Library Complexity"):
                            gr.Markdown("_Library complexity analysis (from RNAseqQC)_")
                            qc_library_complexity_html = gr.HTML()
                        with gr.Tab("📊 QC: Variance Stabilization"):
                            gr.Markdown("_Mean-SD plot after variance stabilization (from RNAseqQC)_")
                            qc_variance_html = gr.HTML()
                        with gr.Tab("📊 QC: Sample Clustering"):
                            gr.Markdown("_Hierarchical clustering of samples (from RNAseqQC)_")
                            qc_clustering_html = gr.HTML()
                        with gr.Tab("📊 QC: PCA Scatters"):
                            gr.Markdown("_Multiple PC scatter plots (from RNAseqQC)_")
                            
                            with gr.Row():
                                pca_color_dropdown = gr.Dropdown(
                                    label="Color by",
                                    choices=[],
                                    value=None,
                                    interactive=True
                                )
                                pca_shape_dropdown = gr.Dropdown(
                                    label="Shape by (optional)",
                                    choices=[],
                                    value=None,
                                    interactive=True
                                )
                                pca_update_button = gr.Button("🔄 Update PCA Plot", size="sm")
                            
                            qc_pca_scatters_html = gr.HTML()
                    
                    # Button to refresh QC plots
                    refresh_qc_button = gr.Button("🔄 Show/Refresh QC Plots", size="sm")
                    
                    # Event handlers
                    metadata_upload.change(
                        fn=self.upload_metadata,
                        inputs=[metadata_upload],
                        outputs=[metadata_status, pca_color_dropdown, pca_shape_dropdown]
                    )
                    
                    suggest_button.click(
                        fn=self.suggest_design,
                        inputs=[],
                        outputs=[design_output]
                    )
                    
                    refresh_viz_button.click(
                        fn=self.refresh_de_visualizations,
                        inputs=[contrast_dropdown],
                        outputs=[volcano_plot_html, ma_plot_html, top_genes_html]
                    )
                    
                    refresh_qc_button.click(
                        fn=self.get_rnaseqqc_plots_html,
                        inputs=[],
                        outputs=[qc_total_counts_html, qc_library_complexity_html,
                                qc_variance_html, qc_clustering_html, qc_pca_scatters_html]
                    )
                    
                    pca_update_button.click(
                        fn=self.regenerate_pca_plot,
                        inputs=[pca_color_dropdown, pca_shape_dropdown],
                        outputs=[qc_pca_scatters_html]
                    )
                
                # Tab 4: Enrichment Analysis
                with gr.Tab("🎯 Enrichment Analysis"):
                    gr.Markdown("### Gene Set Enrichment")
                    gr.Markdown("_Run enrichment analysis on significant genes from a specific contrast_")
                    
                    enrich_contrast_dropdown = gr.Dropdown(
                        label="Select Contrast for Enrichment",
                        choices=["Intercept"],
                        value="Intercept",
                        info="Choose which contrast's significant genes to analyze"
                    )
                    
                    enrich_button = gr.Button("Run Enrichment Analysis", variant="primary")
                    enrich_output = gr.Markdown()
                    
                    # Enrichment Results Display with Database Selector
                    gr.Markdown("### Enrichment Results")
                    
                    with gr.Row():
                        enrich_db_dropdown = gr.Dropdown(
                            label="Select Database",
                            choices=["GO_Biological_Process_2021"],
                            value="GO_Biological_Process_2021",
                            info="Choose which enrichment database to view",
                            interactive=True
                        )
                    
                    enrichment_results_html = gr.HTML()
                    
                    enrich_button.click(
                        fn=self.run_enrichment,
                        inputs=[enrich_contrast_dropdown],
                        outputs=[enrich_output, enrich_db_dropdown, enrichment_results_html],
                        show_progress="full"
                    )
                    
                    enrich_db_dropdown.change(
                        fn=self.display_enrichment_database,
                        inputs=[enrich_db_dropdown],
                        outputs=[enrichment_results_html]
                    )
                
                # Now set up the DE button handler after enrichment dropdown is defined
                # Run DE and update contrast dropdowns and auto-generate visualizations
                def run_de_and_update(design_formula, use_r, progress=gr.Progress()):
                    result = self.run_de_analysis(design_formula, use_r, progress)
                    # Update dropdown with available contrasts
                    updated_dropdown = gr.Dropdown(
                        choices=self.available_contrasts if self.available_contrasts else ["Intercept"],
                        value=self.available_contrasts[0] if self.available_contrasts else "Intercept"
                    )
                    # Same dropdown for enrichment
                    updated_enrich_dropdown = gr.Dropdown(
                        choices=self.available_contrasts if self.available_contrasts else ["Intercept"],
                        value=self.available_contrasts[0] if self.available_contrasts else "Intercept"
                    )
                    # Auto-generate visualizations for first contrast
                    if self.de_results and self.de_results.get('status') == 'success':
                        volcano, ma, top_genes = self.refresh_de_visualizations()
                        # Also load RNAseqQC plots
                        qc_plots = self.get_rnaseqqc_plots_html()
                    else:
                        volcano = ma = top_genes = "<p>Run DE analysis first</p>"
                        qc_plots = tuple(["<p>Run DE analysis first</p>"] * 5)
                    
                    return (result, updated_dropdown, volcano, ma, top_genes, updated_enrich_dropdown) + qc_plots
                
                de_button.click(
                    fn=run_de_and_update,
                    inputs=[design_formula, use_r],
                    outputs=[de_output, contrast_dropdown, volcano_plot_html, ma_plot_html, top_genes_html, enrich_contrast_dropdown,
                            qc_total_counts_html, qc_library_complexity_html,
                            qc_variance_html, qc_clustering_html, qc_pca_scatters_html],
                    show_progress="full"
                )
                
                # Tab 5: Chat Interface
                with gr.Tab("💬 Chat with Agent"):
                    gr.Markdown("### Ask questions about your data and analysis")
                    
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me anything about your RNA-seq data...",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear Chat")
                    
                    submit.click(
                        fn=self.chat,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                    
                    msg.submit(
                        fn=self.chat,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                    
                    clear.click(
                        fn=self.reset_chat,
                        inputs=[],
                        outputs=[chatbot]
                    )
            
            gr.Markdown("""
---
### 📖 Quick Guide:
1. **Upload Data**: Start by uploading your count matrix and metadata
2. **Quality Control**: Run QC to filter low-count genes
3. **Differential Expression**: Get design suggestions and run DE analysis
4. **Enrichment**: Analyze enriched pathways in your significant genes
5. **Chat**: Ask the agent questions about your results

**Powered by:** Ollama (gpt-oss:20b + cniongolo/biomistral) | DESeq2 | Enrichr
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the interface"""
        interface = self.create_interface()
        # Pass theme to launch for Gradio 6.0+ compatibility
        if not kwargs.get('theme'):
            kwargs['theme'] = self.theme
        interface.launch(**kwargs)
        
        # Cleanup on exit
        self.mcp_manager.stop_all()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BulkRNA Agent Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run on")
    parser.add_argument("--port", default=7860, type=int, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Initialize and launch
    app = BulkRNAWebInterface()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
