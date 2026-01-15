"""
Enhanced Web Interface with Progress Bars, Contrast Selection, and Result Visualization
This file contains the enhanced methods to be integrated into web_interface.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def get_possible_contrasts(metadata_file: str) -> List[Tuple[str, str, str]]:
    """
    Analyze metadata to find all possible contrasts
    Returns list of (factor, level1, level2) tuples
    """
    try:
        metadata = pd.read_csv(metadata_file, index_col=0)
        contrasts = []
        
        for col in metadata.columns:
            # Check if column has categorical data
            unique_vals = metadata[col].unique()
            if 2 <= len(unique_vals) <= 10:  # Reasonable number for contrasts
                # Generate all pairwise contrasts
                for i, val1 in enumerate(unique_vals):
                    for val2 in unique_vals[i+1:]:
                        contrasts.append((col, str(val1), str(val2)))
        
        return contrasts
    except Exception as e:
        logger.error(f"Error finding contrasts: {e}")
        return []


def format_contrast_label(contrast: Tuple[str, str, str]) -> str:
    """Format contrast for display"""
    factor, level1, level2 = contrast
    return f"{factor}: {level1} vs {level2}"


def create_volcano_plot_html(results_df: pd.DataFrame, contrast_name: str, fdr_threshold: float, lfc_threshold: float) -> str:
    """Create interactive volcano plot using plotly"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Add significance column
        results_df['significant'] = (
            (results_df['padj'] < fdr_threshold) & 
            (results_df['log2FoldChange'].abs() > lfc_threshold)
        )
        results_df['direction'] = 'Not Significant'
        results_df.loc[(results_df['significant']) & (results_df['log2FoldChange'] > 0), 'direction'] = 'Upregulated'
        results_df.loc[(results_df['significant']) & (results_df['log2FoldChange'] < 0), 'direction'] = 'Downregulated'
        
        # Create plot
        fig = go.Figure()
        
        colors = {'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'gray'}
        
        for direction in ['Not Significant', 'Downregulated', 'Upregulated']:
            df_subset = results_df[results_df['direction'] == direction]
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
            title=f'Volcano Plot - {contrast_name}',
            xaxis_title='log2 Fold Change',
            yaxis_title='-log10(p-value)',
            hovermode='closest',
            height=500
        )
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray", annotation_text="p=0.05")
        fig.add_vline(x=lfc_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=-lfc_threshold, line_dash="dash", line_color="gray")
        
        return fig.to_html(include_plotlyjs='cdn', div_id='volcano_plot')
    
    except Exception as e:
        logger.error(f"Error creating volcano plot: {e}")
        return f"<p>Error creating plot: {str(e)}</p>"


def create_ma_plot_html(results_df: pd.DataFrame, contrast_name: str, fdr_threshold: float, lfc_threshold: float) -> str:
    """Create interactive MA plot using plotly"""
    try:
        import plotly.graph_objects as go
        
        # Add significance column
        results_df['significant'] = (
            (results_df['padj'] < fdr_threshold) & 
            (results_df['log2FoldChange'].abs() > lfc_threshold)
        )
        
        fig = go.Figure()
        
        # Non-significant genes
        df_ns = results_df[~results_df['significant']]
        fig.add_trace(go.Scatter(
            x=np.log10(df_ns['baseMean'].clip(lower=1)),
            y=df_ns['log2FoldChange'],
            mode='markers',
            name='Not Significant',
            marker=dict(color='gray', size=3, opacity=0.4),
            text=df_ns.index,
            hovertemplate='<b>%{text}</b><br>baseMean: %{x:.1f}<br>log2FC: %{y:.2f}<extra></extra>'
        ))
        
        # Significant genes
        df_sig = results_df[results_df['significant']]
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
            title=f'MA Plot - {contrast_name}',
            xaxis_title='log10(baseMean)',
            yaxis_title='log2 Fold Change',
            hovermode='closest',
            height=500
        )
        
        # Add threshold lines
        fig.add_hline(y=lfc_threshold, line_dash="dash", line_color="gray")
        fig.add_hline(y=-lfc_threshold, line_dash="dash", line_color="gray")
        fig.add_hline(y=0, line_color="black", line_width=0.5)
        
        return fig.to_html(include_plotlyjs='cdn', div_id='ma_plot')
    
    except Exception as e:
        logger.error(f"Error creating MA plot: {e}")
        return f"<p>Error creating plot: {str(e)}</p>"


def format_results_table(results_df: pd.DataFrame, top_n: int = 50) -> str:
    """Format top DE results as HTML table"""
    try:
        # Get top genes by adjusted p-value
        top_results = results_df.nsmallest(top_n, 'padj')
        
        # Format for display
        display_df = top_results[['baseMean', 'log2FoldChange', 'pvalue', 'padj']].copy()
        display_df['baseMean'] = display_df['baseMean'].apply(lambda x: f"{x:.1f}")
        display_df['log2FoldChange'] = display_df['log2FoldChange'].apply(lambda x: f"{x:.3f}")
        display_df['pvalue'] = display_df['pvalue'].apply(lambda x: f"{x:.2e}")
        display_df['padj'] = display_df['padj'].apply(lambda x: f"{x:.2e}")
        
        # Create HTML table
        html = display_df.to_html(classes='dataframe', border=0)
        return html
    
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return "<p>Error formatting results</p>"
