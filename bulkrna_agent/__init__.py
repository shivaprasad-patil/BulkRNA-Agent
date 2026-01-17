"""
BulkRNA Agent - Transcriptomics AI Agent for Bulk RNA-seq Analysis

This agent provides comprehensive bulk RNA-seq analysis including:
- Quality Control (QC)
- Differential Expression Analysis (DESeq2)
- Enrichment Analysis
- Design Matrix Suggestions
"""

__version__ = "0.1.0"
__author__ = "Shivaprasad Patil"

from .config import BulkRNAConfig
from .agent import BulkRNAAgent

__all__ = ["BulkRNAConfig", "BulkRNAAgent"]
