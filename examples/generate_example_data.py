"""
Generate example RNA-seq data for testing
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_example_data():
    """Generate synthetic RNA-seq count data and metadata"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_genes = 1000
    n_control = 3
    n_treated = 3
    n_samples = n_control + n_treated
    
    # Generate gene names
    gene_names = [f"ENSG{i:011d}" for i in range(n_genes)]
    
    # Generate sample names
    control_samples = [f"control_{i+1}" for i in range(n_control)]
    treated_samples = [f"treated_{i+1}" for i in range(n_treated)]
    sample_names = control_samples + treated_samples
    
    # Generate count data
    # Most genes: similar between conditions
    # Some genes: differentially expressed
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_genes, n_samples))
    
    # Add some differentially expressed genes (first 100 genes)
    # Upregulated in treated
    counts[:50, n_control:] = counts[:50, n_control:] * 3
    
    # Downregulated in treated
    counts[50:100, n_control:] = counts[50:100, n_control:] // 3
    
    # Create count matrix dataframe
    counts_df = pd.DataFrame(
        counts,
        index=gene_names,
        columns=sample_names
    )
    
    # Create metadata
    metadata = pd.DataFrame({
        'sample_id': sample_names,
        'condition': ['control'] * n_control + ['treated'] * n_treated,
        'batch': ['batch1'] * 2 + ['batch2'] * 2 + ['batch1'] * 2,
        'replicate': [1, 2, 3, 1, 2, 3]
    })
    metadata.set_index('sample_id', inplace=True)
    
    # Save files
    data_dir = Path('./data/examples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    counts_df.to_csv(data_dir / 'example_counts.csv')
    metadata.to_csv(data_dir / 'example_metadata.csv')
    
    print("✅ Example data generated!")
    print(f"📁 Count matrix: {data_dir / 'example_counts.csv'}")
    print(f"   - {n_genes} genes × {n_samples} samples")
    print(f"📁 Metadata: {data_dir / 'example_metadata.csv'}")
    print(f"   - Conditions: control (n={n_control}), treated (n={n_treated})")
    print(f"   - Expected DE genes: ~100 (50 up, 50 down)")
    
    # Generate a README for the example data
    readme = f"""# Example Data

This directory contains synthetic RNA-seq data for testing BulkRNA Agent.

## Files

- `example_counts.csv`: Count matrix ({n_genes} genes × {n_samples} samples)
- `example_metadata.csv`: Sample metadata

## Experimental Design

- **Control group**: {n_control} samples
- **Treated group**: {n_treated} samples
- **Batches**: 2 batches (batch1, batch2)

## Expected Results

This synthetic data contains:
- 50 upregulated genes in treated vs control (3-fold change)
- 50 downregulated genes in treated vs control (3-fold change)
- ~900 non-differentially expressed genes

## Usage

Upload these files to BulkRNA Agent:
1. Go to "Data Upload" tab
2. Upload `example_counts.csv` as count matrix
3. Upload `example_metadata.csv` as metadata
4. Run QC, then DE analysis with design: `~ condition`

## Analysis Parameters

Recommended settings:
- Design formula: `~ condition`
- FDR threshold: 0.05
- Log2FC threshold: 1.0
"""
    
    with open(data_dir / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"📄 README: {data_dir / 'README.md'}")

if __name__ == "__main__":
    generate_example_data()
