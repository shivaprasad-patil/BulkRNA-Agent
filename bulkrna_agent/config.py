"""
Configuration for BulkRNA Agent
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging


@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    reasoning_model: str = "gpt-oss:20b"  # General reasoning
    biomedical_model: str = "cniongolo/biomistral"  # Biomedical expertise
    base_url: str = "http://localhost:11434"  # Ollama default
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 300


@dataclass
class DataConfig:
    """Configuration for data handling"""
    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"
    cache_dir: str = "./data/cache"
    max_file_size_mb: int = 500


@dataclass
class AnalysisConfig:
    """Configuration for RNA-seq analysis"""
    min_count_threshold: int = 10
    fdr_threshold: float = 0.05
    log2fc_threshold: float = 1.0
    normalization_method: str = "DESeq2"  # DESeq2, TMM, CPM
    
    # DESeq2 specific
    deseq2_design_formula: Optional[str] = None
    deseq2_contrast: Optional[list] = None
    
    # Enrichment analysis
    enrichment_databases: list = field(default_factory=lambda: [
        "GO_Biological_Process_2021",
        "GO_Molecular_Function_2021", 
        "KEGG_2021_Human",
        "Reactome_2022"
    ])
    enrichment_cutoff: float = 0.05


@dataclass
class BulkRNAConfig:
    """Main configuration for BulkRNA Agent"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/bulkrna_agent.log"
    
    # MCP Server
    use_mcp_server: bool = True
    mcp_servers: Dict[str, Any] = field(default_factory=lambda: {
        "r_transcriptomics": {
            "command": "Rscript",
            "args": ["./mcp_servers/r_transcriptomics_server.R"]
        }
    })
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data.upload_dir, exist_ok=True)
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
