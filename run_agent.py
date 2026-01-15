"""
Example script to run BulkRNA Agent
"""
from bulkrna_agent import BulkRNAConfig
from bulkrna_agent.web_interface import BulkRNAWebInterface

def main():
    # Create configuration
    config = BulkRNAConfig()
    
    # Customize if needed
    config.llm.reasoning_model = "gpt-oss:20b"
    config.llm.biomedical_model = "cniongolo/biomistral"
    config.analysis.fdr_threshold = 0.05
    config.analysis.log2fc_threshold = 1.0
    
    print("🧬 Starting BulkRNA Agent...")
    print(f"📊 FDR threshold: {config.analysis.fdr_threshold}")
    print(f"📈 Log2FC threshold: {config.analysis.log2fc_threshold}")
    print(f"🤖 Reasoning LLM: {config.llm.reasoning_model}")
    print(f"🧪 Biomedical LLM: {config.llm.biomedical_model}")
    print("\n✨ Launching web interface...\n")
    
    # Create and launch interface
    app = BulkRNAWebInterface(config)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True
    )

if __name__ == "__main__":
    main()
