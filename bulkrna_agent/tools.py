"""
Transcriptomics tools for BulkRNA Agent
Implements QC, differential expression, and enrichment analysis
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
import mygene

logger = logging.getLogger(__name__)


class RNASeqTool:
    """Base class for RNA-seq analysis tools"""
    
    def __init__(self, config):
        self.config = config
        self.name = "base_tool"
        self.description = "Base RNA-seq tool"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        raise NotImplementedError


class QualityControlTool(RNASeqTool):
    """Quality control for RNA-seq data"""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "quality_control"
        self.description = """
        Perform quality control on RNA-seq count data.
        
        Parameters:
        - counts_file: Path to count matrix (CSV/TSV)
        - metadata_file: Path to sample metadata
        
        Returns:
        - QC metrics: library size, detected genes, alignment stats
        - QC plots: distribution, PCA, correlation heatmap
        - Filtered data if requested
        """
    
    def execute(
        self,
        counts_file: str,
        metadata_file: Optional[str] = None,
        min_counts: int = 10,
        min_samples: int = 2
    ) -> Dict[str, Any]:
        """
        Execute quality control
        """
        try:
            logger.info(f"Starting QC on {counts_file}")
            
            # Load count data
            counts_df = pd.read_csv(counts_file, index_col=0)
            logger.info(f"Loaded count matrix: {counts_df.shape}")
            
            # Load metadata if provided
            metadata_df = None
            if metadata_file:
                metadata_df = pd.read_csv(metadata_file, index_col=0)
                logger.info(f"Loaded metadata: {metadata_df.shape}")
            
            # Calculate QC metrics
            qc_metrics = self._calculate_qc_metrics(counts_df)
            
            # Filter low count genes
            filtered_counts = self._filter_low_counts(
                counts_df, min_counts, min_samples
            )
            
            # Save results
            output_dir = Path(self.config.data.output_dir) / "qc"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filtered_counts.to_csv(output_dir / "filtered_counts.csv")
            
            with open(output_dir / "qc_metrics.json", 'w') as f:
                json.dump(qc_metrics, f, indent=2)
            
            result = {
                "status": "success",
                "metrics": qc_metrics,
                "filtered_counts_path": str(output_dir / "filtered_counts.csv"),
                "n_genes_before": counts_df.shape[0],
                "n_genes_after": filtered_counts.shape[0],
                "n_samples": counts_df.shape[1]
            }
            
            logger.info(f"QC completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in QC: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _calculate_qc_metrics(self, counts_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate QC metrics"""
        metrics = {}
        
        # Per-sample metrics
        sample_metrics = {}
        for sample in counts_df.columns:
            sample_counts = counts_df[sample]
            sample_metrics[sample] = {
                "total_counts": int(sample_counts.sum()),
                "detected_genes": int((sample_counts > 0).sum()),
                "median_counts": float(sample_counts.median()),
                "mean_counts": float(sample_counts.mean())
            }
        
        metrics["sample_metrics"] = sample_metrics
        
        # Overall metrics
        metrics["total_genes"] = counts_df.shape[0]
        metrics["total_samples"] = counts_df.shape[1]
        metrics["median_library_size"] = np.median([
            m["total_counts"] for m in sample_metrics.values()
        ])
        
        return metrics
    
    def _filter_low_counts(
        self,
        counts_df: pd.DataFrame,
        min_counts: int,
        min_samples: int
    ) -> pd.DataFrame:
        """Filter genes with low counts"""
        # Keep genes with at least min_counts in at least min_samples
        keep = (counts_df >= min_counts).sum(axis=1) >= min_samples
        filtered = counts_df[keep]
        
        logger.info(
            f"Filtered {counts_df.shape[0] - filtered.shape[0]} genes "
            f"with low counts"
        )
        
        return filtered


class DifferentialExpressionTool(RNASeqTool):
    """Differential expression analysis using DESeq2"""
    
    def __init__(self, config, mcp_manager=None):
        super().__init__(config)
        self.name = "differential_expression"
        self.mcp_manager = mcp_manager
        self.description = """
        Perform differential expression analysis using DESeq2.
        
        Parameters:
        - counts_file: Path to filtered count matrix
        - metadata_file: Path to sample metadata with conditions
        - design_formula: DESeq2 design formula (e.g., "~ condition")
        - contrast: Comparison to make (e.g., ["condition", "treated", "control"])
        
        Returns:
        - Differential expression results with log2FC, p-values, FDR
        - Normalized counts
        - Diagnostic plots
        """
    
    def execute(
        self,
        counts_file: str,
        metadata_file: str,
        design_formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        use_mcp: bool = True
    ) -> Dict[str, Any]:
        """
        Execute differential expression analysis
        """
        try:
            logger.info(f"Starting DE analysis on {counts_file}")
            
            if use_mcp and self.config.use_mcp_server:
                # Use MCP server for DESeq2 analysis
                return self._execute_via_mcp(
                    counts_file, metadata_file, design_formula, contrast
                )
            else:
                # Use PyDESeq2 (Python implementation)
                return self._execute_pydeseq2(
                    counts_file, metadata_file, design_formula, contrast
                )
                
        except Exception as e:
            logger.error(f"Error in DE analysis: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _execute_pydeseq2(
        self,
        counts_file: str,
        metadata_file: str,
        design_formula: Optional[str],
        contrast: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute using PyDESeq2"""
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
            
            # Load data
            counts_df = pd.read_csv(counts_file, index_col=0)
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            
            logger.info(f"Loaded counts shape: {counts_df.shape}")
            logger.info(f"Loaded metadata shape: {metadata_df.shape}")
            
            # PyDESeq2/AnnData expects: X = samples (rows) × genes (cols), obs = samples (rows) × conditions (cols)
            # Our counts file has genes (rows) × samples (cols), so we need to TRANSPOSE
            
            logger.info(f"Count columns (samples): {list(counts_df.columns[:5])}")
            logger.info(f"Metadata index (samples): {list(metadata_df.index[:5])}")
            
            # Match samples between counts and metadata
            common_samples = list(set(counts_df.columns) & set(metadata_df.index))
            if len(common_samples) == 0:
                raise ValueError("No matching samples between counts and metadata. Check sample names.")
            
            logger.info(f"Found {len(common_samples)} common samples")
            
            # Reorder to match
            counts_df = counts_df[common_samples]
            metadata_df = metadata_df.loc[common_samples]
            
            # TRANSPOSE: genes×samples → samples×genes for PyDESeq2
            counts_df = counts_df.T
            
            logger.info(f"Final counts shape after transpose (samples × genes): {counts_df.shape}")
            logger.info(f"Final metadata shape (samples × conditions): {metadata_df.shape}")
            
            # Ensure counts is numeric and has integer counts
            counts_df = counts_df.astype(int)
            
            # Set default design
            if design_formula is None:
                design_formula = "~ condition"
            
            # Create DESeq2 dataset
            dds = DeseqDataSet(
                counts=counts_df,
                metadata=metadata_df,
                design=design_formula
            )
            
            # Run DESeq2
            dds.deseq2()
            
            # Get results
            stat_res = DeseqStats(dds)
            stat_res.summary()
            results_df = stat_res.results_df
            
            # Save results
            output_dir = Path(self.config.data.output_dir) / "de_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_df.to_csv(output_dir / "de_results.csv")
            
            # Get significant genes
            sig_genes = results_df[
                (results_df['padj'] < self.config.analysis.fdr_threshold) &
                (abs(results_df['log2FoldChange']) > self.config.analysis.log2fc_threshold)
            ]
            
            sig_genes.to_csv(output_dir / "significant_genes.csv")
            
            result = {
                "status": "success",
                "results_path": str(output_dir / "de_results.csv"),
                "significant_genes_path": str(output_dir / "significant_genes.csv"),
                "n_significant": len(sig_genes),
                "n_upregulated": len(sig_genes[sig_genes['log2FoldChange'] > 0]),
                "n_downregulated": len(sig_genes[sig_genes['log2FoldChange'] < 0]),
                "design_formula": design_formula
            }
            
            logger.info(f"DE analysis completed: {result['n_significant']} significant genes")
            return result
            
        except ImportError:
            logger.error("PyDESeq2 not installed. Install with: pip install pydeseq2")
            return {
                "status": "error",
                "message": "PyDESeq2 not installed. Please install pydeseq2."
            }
        except Exception as e:
            logger.error(f"Error in PyDESeq2 analysis: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _execute_via_mcp(
        self,
        counts_file: str,
        metadata_file: str,
        design_formula: Optional[str],
        contrast: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute via MCP server (R DESeq2)"""
        try:
            if self.mcp_manager is None:
                logger.warning("MCP manager not available, falling back to PyDESeq2")
                return self._execute_pydeseq2(
                    counts_file, metadata_file, design_formula, contrast
                )
            
            # Get R transcriptomics server
            r_server = self.mcp_manager.get_server("r_transcriptomics")
            if r_server is None:
                logger.warning("R transcriptomics server not found, falling back to PyDESeq2")
                return self._execute_pydeseq2(
                    counts_file, metadata_file, design_formula, contrast
                )
            
            logger.info("Executing DESeq2 via R MCP server")
            
            # Run DESeq2 via R
            result = r_server.run_deseq2(
                counts_file=counts_file,
                metadata_file=metadata_file,
                design_formula=design_formula,
                contrast=contrast or []
            )
            
            if result["status"] != "success":
                logger.error(f"R DESeq2 failed: {result.get('message')}")
                logger.info("Falling back to PyDESeq2")
                return self._execute_pydeseq2(
                    counts_file, metadata_file, design_formula, contrast
                )
            
            # Parse R results
            import pandas as pd
            results_df = pd.read_csv(result["results_path"], index_col=0)
            
            # Filter significant genes
            sig_genes = results_df[
                (results_df["padj"] < self.config.analysis.fdr_threshold) &
                (results_df["log2FoldChange"].abs() > self.config.analysis.log2fc_threshold)
            ]
            
            # Save significant genes
            output_dir = Path(self.config.data.output_dir) / "de_analysis"
            sig_genes_path = output_dir / "significant_genes.csv"
            sig_genes.to_csv(sig_genes_path)
            
            n_up = sum(sig_genes["log2FoldChange"] > 0)
            n_down = sum(sig_genes["log2FoldChange"] < 0)
            
            logger.info(f"Found {len(sig_genes)} significant genes ({n_up} up, {n_down} down)")
            
            return {
                "status": "success",
                "design_formula": design_formula,
                "results_path": result["results_path"],
                "normalized_counts_path": result["normalized_counts_path"],
                "significant_genes_path": str(sig_genes_path),
                "n_significant": len(sig_genes),
                "n_upregulated": int(n_up),
                "n_downregulated": int(n_down),
                "method": "R DESeq2"
            }
            
        except Exception as e:
            logger.error(f"Error in R DESeq2: {e}", exc_info=True)
            logger.info("Falling back to PyDESeq2")
            return self._execute_pydeseq2(
                counts_file, metadata_file, design_formula, contrast
            )
    
    def get_rnaseqqc_plots(self) -> Dict[str, Any]:
        """
        Get paths to RNAseqQC plots generated during DE analysis
        
        Returns:
            Dictionary with plot paths and availability status
        """
        try:
            output_dir = Path(self.config.data.output_dir) / "de_analysis" / "qc_plots"
            
            # Define expected QC plots
            qc_plots = {
                "total_counts": "total_counts.png",
                "library_complexity": "library_complexity.png",
                "variance_stabilization": "variance_stabilization.png",
                "sample_clustering": "sample_clustering.png",
                "pca_scatters": "pca_scatters.png"
            }
            
            # Check which plots exist
            available_plots = {}
            for plot_name, filename in qc_plots.items():
                plot_path = output_dir / filename
                if plot_path.exists():
                    available_plots[plot_name] = str(plot_path)
                else:
                    available_plots[plot_name] = None
            
            # Count available plots
            n_available = sum(1 for v in available_plots.values() if v is not None)
            
            return {
                "status": "success",
                "plots": available_plots,
                "n_available": n_available,
                "qc_dir": str(output_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting RNAseqQC plots: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "plots": {}
            }


class EnrichmentAnalysisTool(RNASeqTool):
    """Gene set enrichment analysis"""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "enrichment_analysis"
        self.description = """
        Perform gene set enrichment analysis on differentially expressed genes.
        
        Parameters:
        - gene_list: List of gene symbols or path to file
        - background: Background gene list (optional)
        - databases: Enrichment databases to query
        
        Returns:
        - Enriched pathways/gene sets with p-values and FDR
        - Enrichment plots
        """
    
    def _convert_ensembl_to_symbols(self, ensembl_ids: List[str]) -> List[str]:
        """
        Convert Ensembl gene IDs to gene symbols using mygene
        """
        logger.info(f"Converting {len(ensembl_ids)} Ensembl IDs to gene symbols...")
        
        try:
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                ensembl_ids,
                scopes='ensembl.gene',
                fields='symbol',
                species='human',
                returnall=True
            )
            
            # Extract symbols from results
            gene_symbols = []
            unmapped = 0
            
            for query_result in results['out']:
                if 'symbol' in query_result:
                    gene_symbols.append(query_result['symbol'])
                else:
                    unmapped += 1
                    logger.debug(f"Could not map: {query_result.get('query', 'unknown')}")
            
            logger.info(f"Successfully converted {len(gene_symbols)} IDs to symbols")
            if unmapped > 0:
                logger.warning(f"Could not map {unmapped} Ensembl IDs to symbols")
            
            return gene_symbols
            
        except Exception as e:
            logger.error(f"Error converting Ensembl IDs: {e}")
            logger.warning("Proceeding with original IDs (may result in no enrichment results)")
            return ensembl_ids
    
    def execute(
        self,
        gene_list: List[str],
        databases: Optional[List[str]] = None,
        organism: str = "human"
    ) -> Dict[str, Any]:
        """
        Execute enrichment analysis using Enrichr API
        """
        try:
            logger.info(f"Starting enrichment analysis for {len(gene_list)} genes")
            logger.info(f"First 5 genes: {gene_list[:5]}")
            
            # Check if genes are Ensembl IDs (start with ENSG)
            ensembl_count = sum(1 for g in gene_list[:20] if g.startswith('ENSG'))
            if ensembl_count > 10:
                logger.info("Detected Ensembl IDs - converting to gene symbols...")
                gene_list = self._convert_ensembl_to_symbols(gene_list)
                logger.info(f"After conversion, first 5 genes: {gene_list[:5]}")
                if len(gene_list) == 0:
                    return {
                        "status": "error",
                        "message": "Failed to convert Ensembl IDs to gene symbols",
                        "results": {}
                    }
            
            if databases is None:
                databases = self.config.analysis.enrichment_databases
            
            results = {}
            
            for database in databases:
                logger.info(f"Querying {database}")
                enrich_results = self._query_enrichr(gene_list, database)
                results[database] = enrich_results
            
            # Save results
            output_dir = Path(self.config.data.output_dir) / "enrichment"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            for db, res in results.items():
                df = pd.DataFrame(res)
                if not df.empty:
                    output_file = output_dir / f"enrichment_{db}.csv"
                    df.to_csv(output_file, index=False)
                    saved_files.append(str(output_file))
                    logger.info(f"Saved {len(df)} enrichment results to {output_file}")
                else:
                    logger.warning(f"No enrichment results for {db}")
            
            result = {
                "status": "success",
                "output_dir": str(output_dir),
                "databases": databases,
                "n_genes": len(gene_list),
                "saved_files": saved_files,
                "results": results
            }
            
            logger.info("Enrichment analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in enrichment analysis: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _query_enrichr(self, gene_list: List[str], database: str) -> List[Dict]:
        """Query Enrichr API"""
        try:
            import requests
            
            # Submit gene list
            ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/addList'
            genes_str = '\n'.join(gene_list)
            payload = {
                'list': (None, genes_str),
            }
            
            response = requests.post(ENRICHR_URL, files=payload)
            if not response.ok:
                raise Exception('Error analyzing gene list')
            
            data = response.json()
            user_list_id = data['userListId']
            
            # Get enrichment results
            ENRICHR_RESULT_URL = 'https://maayanlab.cloud/Enrichr/enrich'
            query_string = f'?userListId={user_list_id}&backgroundType={database}'
            
            response = requests.get(ENRICHR_RESULT_URL + query_string)
            if not response.ok:
                raise Exception('Error fetching enrichment results')
            
            data = response.json()
            
            # Log API response structure for debugging
            logger.debug(f"API response keys: {list(data.keys())}")
            logger.debug(f"Full response: {str(data)[:500]}...")
            
            # Parse results - get ALL results, not filtered
            results = []
            db_results = data.get(database, [])
            logger.info(f"Enrichr returned {len(db_results)} terms for {database}")
            
            for entry in db_results:
                results.append({
                    'term': entry[1],
                    'p_value': entry[2],
                    'adjusted_p_value': entry[6],
                    'genes': entry[5],
                    'combined_score': entry[4]
                })
            
            # Sort by p-value (most significant first)
            results = sorted(results, key=lambda x: x['p_value'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying Enrichr: {e}")
            return []


class DesignMatrixSuggestionTool(RNASeqTool):
    """Suggest design matrix for DESeq2 analysis"""
    
    def __init__(self, config, llm_manager):
        super().__init__(config)
        self.llm_manager = llm_manager
        self.name = "design_matrix_suggestion"
        self.description = """
        Suggest appropriate design matrix formula for DESeq2 based on metadata.
        
        Parameters:
        - metadata_file: Path to sample metadata
        
        Returns:
        - Suggested design formula
        - Explanation of the design
        - Possible contrasts
        """
    
    def execute(self, metadata_file: str) -> Dict[str, Any]:
        """
        Analyze metadata and suggest design matrix
        """
        try:
            logger.info(f"Analyzing metadata from {metadata_file}")
            
            # Load metadata
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            
            # Analyze metadata structure
            analysis = self._analyze_metadata(metadata_df)
            
            # Use LLM to suggest design
            suggestion = self._llm_suggest_design(analysis, metadata_df)
            
            result = {
                "status": "success",
                "metadata_summary": analysis,
                "suggested_design": suggestion["design_formula"],
                "explanation": suggestion["explanation"],
                "possible_contrasts": suggestion["contrasts"]
            }
            
            logger.info(f"Design suggestion completed: {suggestion['design_formula']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in design suggestion: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    def _analyze_metadata(self, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze metadata structure"""
        analysis = {
            "n_samples": len(metadata_df),
            "columns": list(metadata_df.columns),
            "column_info": {}
        }
        
        for col in metadata_df.columns:
            unique_vals = metadata_df[col].unique()
            analysis["column_info"][col] = {
                "n_unique": len(unique_vals),
                "values": list(unique_vals)[:10],  # Limit to 10
                "dtype": str(metadata_df[col].dtype)
            }
        
        return analysis
    
    def _llm_suggest_design(
        self,
        analysis: Dict[str, Any],
        metadata_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Use LLM to suggest design formula"""
        
        prompt = f"""
You are analyzing RNA-seq experimental metadata to suggest an appropriate DESeq2 design formula.

Metadata Summary:
- Number of samples: {analysis['n_samples']}
- Available columns: {', '.join(analysis['columns'])}

Column Details:
{json.dumps(analysis['column_info'], indent=2)}

Sample Metadata (first few rows):
{metadata_df.head().to_string()}

Please suggest:
1. An appropriate DESeq2 design formula (e.g., "~ condition" or "~ batch + condition")
2. A clear explanation of why this design is appropriate
3. Possible contrasts that could be tested

Consider:
- Which columns represent experimental conditions
- Whether batch effects need to be accounted for
- Whether interactions should be included
- Sample size for each group

Respond in JSON format:
{{
    "design_formula": "~ ...",
    "explanation": "...",
    "contrasts": [["column", "group1", "group2"], ...]
}}
"""
        
        try:
            response = self.llm_manager.generate(
                prompt=prompt,
                llm_type="reasoning",
                system_prompt="You are an expert in RNA-seq experimental design and DESeq2 analysis."
            )
            
            # Parse JSON response
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                suggestion = json.loads(json_match.group())
            else:
                # Fallback to simple design
                logger.warning("Could not parse LLM response, using default")
                suggestion = {
                    "design_formula": "~ condition",
                    "explanation": "Simple design with single condition factor",
                    "contrasts": [["condition", "treated", "control"]]
                }
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error in LLM suggestion: {e}")
            # Return default
            return {
                "design_formula": "~ condition",
                "explanation": f"Default design (error: {str(e)})",
                "contrasts": [["condition", "group1", "group2"]]
            }
