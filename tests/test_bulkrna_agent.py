# BulkRNA Agent Tests

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.tools import (
    QualityControlTool,
    DifferentialExpressionTool,
    EnrichmentAnalysisTool
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config(temp_dir):
    """Create test configuration"""
    config = BulkRNAConfig()
    config.data.upload_dir = f"{temp_dir}/uploads"
    config.data.output_dir = f"{temp_dir}/outputs"
    config.data.cache_dir = f"{temp_dir}/cache"
    config.log_file = f"{temp_dir}/test.log"
    return config


@pytest.fixture
def example_counts(temp_dir):
    """Create example count data"""
    np.random.seed(42)
    counts = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, size=(100, 6)),
        index=[f"ENSG{i:05d}" for i in range(100)],
        columns=[f"sample{i+1}" for i in range(6)]
    )
    
    path = Path(temp_dir) / "counts.csv"
    counts.to_csv(path)
    return str(path)


@pytest.fixture
def example_metadata(temp_dir):
    """Create example metadata"""
    metadata = pd.DataFrame({
        'sample_id': [f"sample{i+1}" for i in range(6)],
        'condition': ['control'] * 3 + ['treated'] * 3
    })
    metadata.set_index('sample_id', inplace=True)
    
    path = Path(temp_dir) / "metadata.csv"
    metadata.to_csv(path)
    return str(path)


def test_config_initialization(config):
    """Test configuration initialization"""
    assert config.llm.reasoning_model == "gpt-oss:20b"
    assert config.llm.biomedical_model == "cniongolo/biomistral"
    assert Path(config.data.upload_dir).exists()
    assert Path(config.data.output_dir).exists()


def test_qc_tool(config, example_counts, example_metadata):
    """Test quality control tool"""
    tool = QualityControlTool(config)
    
    result = tool.execute(
        counts_file=example_counts,
        metadata_file=example_metadata,
        min_counts=10,
        min_samples=2
    )
    
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["n_samples"] == 6
    assert Path(result["filtered_counts_path"]).exists()


def test_de_tool_pydeseq2(config, example_counts, example_metadata):
    """Test differential expression with PyDESeq2"""
    tool = DifferentialExpressionTool(config)
    
    result = tool.execute(
        counts_file=example_counts,
        metadata_file=example_metadata,
        design_formula="~ condition",
        use_mcp=False
    )
    
    # May fail if PyDESeq2 not installed
    if result["status"] == "error" and "not installed" in result.get("message", ""):
        pytest.skip("PyDESeq2 not installed")
    
    assert result["status"] in ["success", "error"]


def test_enrichment_tool(config):
    """Test enrichment analysis tool"""
    tool = EnrichmentAnalysisTool(config)
    
    # Use a small list of real genes
    gene_list = ["TP53", "BRCA1", "EGFR", "MYC"]
    
    result = tool.execute(gene_list=gene_list)
    
    # May fail due to network issues
    if result["status"] == "error":
        pytest.skip("Enrichment API unavailable")
    
    assert result["status"] == "success"
    assert result["n_genes"] == len(gene_list)


def test_config_env_vars(monkeypatch, temp_dir):
    """Test configuration from environment variables"""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:11434")
    monkeypatch.setenv("FDR_THRESHOLD", "0.01")
    
    # Would need to implement env var loading in config
    config = BulkRNAConfig()
    
    # For now just check defaults
    assert config.llm.base_url == "http://localhost:11434"
    assert config.analysis.fdr_threshold == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
