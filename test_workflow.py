#!/usr/bin/env python3
"""
Test script to verify the workflow is working correctly.
"""

from graph.langgraph_workflow import graph
from utils.logger import api_logger

def test_workflow():
    """Test the complete workflow with a simple query."""
    
    print("=== Testing Research Assistant Workflow ===\n")
    
    # Simple query for testing
    query = "What are transformer architectures?"
    
    print(f"Query: {query}")
    
    # Initialize state
    initial_state = {
        "query": query,
        "papers": [],
        "chunks": [],
        "chunk_batches": [],
        "current_batch": None,
        "current_summary": None,
        "summaries": [],
        "feedback": "reject",
        "retry_count": 0,
        "report": None,
    }
    
    try:
        print("Starting workflow execution...")
        
        # Run the workflow
        final_state = graph.invoke(initial_state)
        
        print("\n=== Workflow Results ===")
        print(f"Papers found: {len(final_state.get('papers', []))}")
        print(f"Chunks retrieved: {len(final_state.get('chunks', []))}")
        print(f"Summaries generated: {len(final_state.get('summaries', []))}")
        print(f"Final report length: {len(final_state.get('report', ''))}")
        
        if final_state.get('report'):
            print("\n=== Final Report ===")
            print(final_state['report'])
        else:
            print("\nNo report generated.")
            
        return final_state
        
    except Exception as e:
        print(f"Error in workflow: {e}")
        api_logger.error(f"Workflow test failed: {e}")
        return None

def test_individual_components():
    """Test individual components to isolate issues."""
    
    print("\n=== Testing Individual Components ===\n")
    
    # Test RetrieverAgent
    print("Testing RetrieverAgent...")
    try:
        from agents.retriever_agent import RetrieverAgent
        retriever = RetrieverAgent()
        chunks = retriever.retrieve("transformer architectures")
        print(f"✓ RetrieverAgent: Found {len(chunks)} chunks")
        
        if chunks:
            print(f"First chunk preview: {chunks[0][:100]}...")
    except Exception as e:
        print(f"✗ RetrieverAgent failed: {e}")
    
    # Test SummarizerAgent
    print("\nTesting SummarizerAgent...")
    try:
        from agents.summarizer_agent import SummarizerAgent
        summarizer = SummarizerAgent()
        
        # Test with sample chunks
        test_chunks = [
            "Transformer architectures use attention mechanisms to process sequential data.",
            "The attention mechanism allows models to focus on different parts of the input.",
            "BERT and GPT are examples of transformer-based models."
        ]
        
        summaries = summarizer.run("transformer architectures", test_chunks)
        print(f"✓ SummarizerAgent: Generated {len(summaries)} summaries")
        
        if summaries:
            print(f"First summary: {summaries[0][:100]}...")
    except Exception as e:
        print(f"✗ SummarizerAgent failed: {e}")

if __name__ == "__main__":
    api_logger.info("Starting workflow test")
    
    # Test individual components first
    test_individual_components()
    
    # Test complete workflow
    result = test_workflow()
    
    if result:
        print("\n✓ Workflow test completed successfully!")
    else:
        print("\n✗ Workflow test failed!")
    
    print("\n=== Test Complete ===")