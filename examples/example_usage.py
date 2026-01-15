"""
Example usage of BulkRNA Agent programmatically
"""
from pathlib import Path
from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.llm import DualLLMManager
from bulkrna_agent.agent import BulkRNAAgent
from bulkrna_agent.tools import (
    QualityControlTool,
    DifferentialExpressionTool,
    EnrichmentAnalysisTool,
    DesignMatrixSuggestionTool
)

def run_analysis_programmatically():
    """
    Example of running analysis without web interface
    """
    # Initialize configuration
    config = BulkRNAConfig()
    
    # Initialize LLM manager
    llm_manager = DualLLMManager(config)
    
    # Initialize tools
    tools = {
        "quality_control": QualityControlTool(config),
        "differential_expression": DifferentialExpressionTool(config),
        "enrichment_analysis": EnrichmentAnalysisTool(config),
        "design_matrix_suggestion": DesignMatrixSuggestionTool(config, llm_manager)
    }
    
    # Initialize agent
    agent = BulkRNAAgent(config, llm_manager, tools)
    
    # Example file paths (replace with your actual files)
    counts_file = "./data/uploads/counts.csv"
    metadata_file = "./data/uploads/metadata.csv"
    
    # Step 1: Run QC
    print("Step 1: Running Quality Control...")
    qc_tool = tools["quality_control"]
    qc_results = qc_tool.execute(
        counts_file=counts_file,
        metadata_file=metadata_file,
        min_counts=10,
        min_samples=2
    )
    print(f"QC Status: {qc_results['status']}")
    print(f"Filtered genes: {qc_results.get('n_genes_after', 0)}")
    
    # Step 2: Get design suggestion
    print("\nStep 2: Getting design matrix suggestion...")
    design_tool = tools["design_matrix_suggestion"]
    design_results = design_tool.execute(metadata_file=metadata_file)
    print(f"Suggested design: {design_results.get('suggested_design', 'N/A')}")
    
    # Step 3: Run differential expression
    print("\nStep 3: Running Differential Expression...")
    de_tool = tools["differential_expression"]
    de_results = de_tool.execute(
        counts_file=qc_results.get("filtered_counts_path", counts_file),
        metadata_file=metadata_file,
        design_formula="~ condition",
        use_mcp=False  # Use PyDESeq2
    )
    print(f"DE Status: {de_results['status']}")
    print(f"Significant genes: {de_results.get('n_significant', 0)}")
    
    # Step 4: Run enrichment analysis
    if de_results.get('status') == 'success' and de_results.get('n_significant', 0) > 0:
        print("\nStep 4: Running Enrichment Analysis...")
        
        # Load significant genes
        import pandas as pd
        sig_genes_df = pd.read_csv(de_results['significant_genes_path'], index_col=0)
        gene_list = sig_genes_df.index.tolist()
        
        enrich_tool = tools["enrichment_analysis"]
        enrich_results = enrich_tool.execute(gene_list=gene_list)
        print(f"Enrichment Status: {enrich_results['status']}")
        print(f"Analyzed {len(gene_list)} genes")
    
    # Step 5: Chat with agent
    print("\nStep 5: Chatting with agent...")
    response = agent.chat("What are the most significant differentially expressed genes?")
    print(f"Agent: {response}")
    
    print("\n✅ Analysis complete!")

def chat_example():
    """
    Example of using the chat interface
    """
    config = BulkRNAConfig()
    llm_manager = DualLLMManager(config)
    tools = {
        "quality_control": QualityControlTool(config),
        "differential_expression": DifferentialExpressionTool(config),
        "enrichment_analysis": EnrichmentAnalysisTool(config),
        "design_matrix_suggestion": DesignMatrixSuggestionTool(config, llm_manager)
    }
    agent = BulkRNAAgent(config, llm_manager, tools)
    
    # Example questions
    questions = [
        "What is differential expression analysis?",
        "Explain what DESeq2 does",
        "What is a design matrix in RNA-seq?",
    ]
    
    for q in questions:
        print(f"\n🙋 User: {q}")
        response = agent.chat(q)
        print(f"🤖 Agent: {response}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chat_example()
    else:
        run_analysis_programmatically()
