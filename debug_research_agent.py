#!/usr/bin/env python3
"""
Debug script for ResearchAgent to identify specific issues.
Run this to get detailed error information.
"""

import sys
import traceback
from agents.research_agent import ResearchAgent
from utils.logger import api_logger

def debug_research_agent():
    """Debug the ResearchAgent with detailed error reporting."""
    
    print("=== ResearchAgent Debug Session ===")
    
    # Initialize the agent
    try:
        agent = ResearchAgent(max_results=5)
        print("✓ ResearchAgent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize ResearchAgent: {e}")
        traceback.print_exc()
        return
    
    # Test search functionality
    query = "What are the latest advancements in LLM-based reasoning systems?"
    print(f"\n--- Testing search with query: {query} ---")
    
    try:
        papers = agent.search(query)
        print(f"✓ Search completed. Found {len(papers)} papers")
        
        if papers:
            print("\nPapers found:")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper['title']}")
                print(f"   URL: {paper['pdf_url']}")
                print(f"   Authors: {', '.join(paper['authors'])}")
                print()
        
    except Exception as e:
        print(f"✗ Search failed: {e}")
        traceback.print_exc()
        return
    
    # Test individual paper processing
    if papers:
        test_paper = papers[0]
        print(f"--- Testing paper processing: {test_paper['title']} ---")
        
        try:
            success = agent.process_paper(test_paper)
            if success:
                print("✓ Paper processed successfully")
            else:
                print("✗ Paper processing failed")
        except Exception as e:
            print(f"✗ Paper processing error: {e}")
            traceback.print_exc()
    
    # Test full workflow
    print(f"\n--- Testing full workflow ---")
    try:
        results = agent.run(query)
        print(f"✓ Full workflow completed. Processed {len(results)} papers successfully")
    except Exception as e:
        print(f"✗ Full workflow failed: {e}")
        traceback.print_exc()

def test_specific_paper():
    """Test processing the specific paper that's causing issues."""
    
    print("\n=== Testing Specific Paper ===")
    
    # The paper that was causing issues
    problem_paper = {
        "title": "MultiGen: Using Multimodal Generation in Simulation to Learn Multimodal Policies in Real",
        "authors": ["Test Author"],
        "summary": "Test summary",
        "pdf_url": "https://arxiv.org/pdf/test.pdf",  # This will fail, but we can see the error
        "published": "2024-01-01",
        "categories": ["cs.AI"]
    }
    
    agent = ResearchAgent(max_results=1)
    
    try:
        success = agent.process_paper(problem_paper)
        print(f"Processing result: {success}")
    except Exception as e:
        print(f"Error processing specific paper: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Set up logging to console for debugging
    api_logger.info("Starting ResearchAgent debug session")
    
    try:
        debug_research_agent()
        test_specific_paper()
    except Exception as e:
        print(f"Critical error in debug session: {e}")
        traceback.print_exc()
    
    print("\n=== Debug Session Complete ===") 