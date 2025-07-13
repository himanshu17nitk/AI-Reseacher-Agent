from graph.langgraph_workflow import graph
import pprint
from utils.logger import api_logger

def run_app():
    query = "What are the latest advancements in LLM-based reasoning systems?"
    
    api_logger.info("Starting Research Assistant with LangChain chunking")
    
    initial_state = {
        "query": query,
        "papers": [],
        "chunks": [],
        "chunk_batches": [],
        "current_batch": None,
        "current_summary": None,
        "summaries": [],
        "feedback": "approve",  # Initialize with approve to start processing
        "retry_count": 0,
        "report": None,
    }

    try:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 5})
        
        if final_state.get("report"):
            print("\n" + "="*80)
            print("RESEARCH REPORT")
            print("="*80)
            pprint.pprint(final_state["report"])
            print("="*80)
        else:
            print("No report generated. Check logs for errors.")
            
    except Exception as e:
        api_logger.error(f"Error in main workflow: {e}")
        
        # If it's a recursion limit error, try to get the final state
        if "recursion_limit" in str(e).lower():
            api_logger.warning("Recursion limit reached. Attempting to get final state...")
            try:
                # Try to invoke with a higher limit to get the final state
                final_state = graph.invoke(initial_state, config={"recursion_limit": 10})
                if final_state.get("report"):
                    print("\n" + "="*80)
                    print("RESEARCH REPORT (Recovery Mode)")
                    print("="*80)
                    pprint.pprint(final_state["report"])
                    print("="*80)
                else:
                    print("Recursion limit reached but no report generated.")
            except Exception as recovery_error:
                api_logger.error(f"Recovery attempt failed: {recovery_error}")
                print(f"Error: {e}")
        else:
            print(f"Error: {e}")

def run_with_custom_chunking():
    """Run with custom chunking parameters."""
    from agents.research_agent import ResearchAgent
    
    # Custom chunking parameters
    chunk_size = 500
    chunk_overlap = 100
    
    print(f"Running with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    agent = ResearchAgent(
        max_results=5,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    query = "Recent developments in transformer architectures"
    papers = agent.run(query)
    
    print(f"Processed {len(papers)} papers successfully")

if __name__ == "__main__":
    # Run the standard workflow
    run_app()
    
    # Uncomment to test custom chunking
    # run_with_custom_chunking()
