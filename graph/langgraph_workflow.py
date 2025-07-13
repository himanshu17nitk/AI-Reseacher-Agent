from agents.research_agent import ResearchAgent
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent import CriticAgent
from agents.writer_agent import WriterAgent
from langgraph.graph import StateGraph, END
from utils.logger import api_logger

from typing import List, TypedDict, Literal, Optional

class GraphState(TypedDict):
    query: str
    papers: List[dict]                # from ResearchAgent
    chunks: List[str]                 # full list of text chunks
    chunk_batches: List[List[str]]   # list of chunk batches (each batch is a list)
    current_batch: Optional[List[str]]    # current batch being processed
    current_summary: Optional[str]        # summary for current batch
    summaries: List[str]             # all accepted summaries
    feedback: Literal["approve", "reject"]
    retry_count: int
    report: Optional[str]

# === Node functions ===

def research_node(state):
    api_logger.log_workflow_step("research", {"query": state["query"]})
    papers = ResearchAgent().run(state["query"])
    state["papers"] = papers
    api_logger.info(f"Research completed. Found {len(papers)} papers")
    return state

def retrieval_node(state):
    api_logger.log_workflow_step("retrieval", {"query": state["query"]})
    # extract questions from query
    chunks = RetrieverAgent().retrieve(state["query"])
    state["chunks"] = chunks
    state["chunk_batches"] = [chunks[i:i+5] for i in range(0, len(chunks), 5)]
    api_logger.info(f"Retrieval completed. Found {len(chunks)} chunks, created {len(state['chunk_batches'])} batches")
    return state

def summarisation_node(state):
    api_logger.log_workflow_step("summarisation", {"batches_remaining": len(state["chunk_batches"])})
    
    if not state["chunk_batches"]:
        api_logger.warning("No chunk batches available for summarization")
        state["current_summary"] = "No content available for summarization."
        # Force move to writer when no batches are available
        state["retry_count"] = 5  # Set to max to force termination
        return state
    
    # Get the current batch
    batch = state["chunk_batches"].pop(0)
    state["current_batch"] = batch
    
    api_logger.debug(f"Processing batch with {len(batch)} chunks")
    
    if not batch:
        api_logger.warning("Current batch is empty")
        state["current_summary"] = "Empty batch - no content to summarize."
        return state
    
    try:
        # Use the single batch summarizer method
        summary = SummarizerAgent().run_single_batch(state["query"], batch, state.get("feedback", ""))
        state["current_summary"] = summary
            
        api_logger.info(f"Summarization completed. Summary length: {len(state['current_summary'])}")
        
    except Exception as e:
        api_logger.log_error_with_recovery(
            e,
            "Error in summarization node",
            "Setting default summary"
        )
        state["current_summary"] = f"Error during summarization: {str(e)}"
    
    return state

def critic_node(state):
    api_logger.log_workflow_step("critic", {"summary_length": len(state.get("current_summary", ""))})
    
    try:
        # Pass the current batch as original_chunks for the critic
        feedback = CriticAgent().critique_summary_agent(
            state["current_summary"], 
            state["current_batch"], 
            state["query"]
        )
        state["feedback"] = feedback
        api_logger.info(f"Critic feedback: {feedback}")
    except Exception as e:
        api_logger.log_error_with_recovery(
            e,
            "Error in critic node",
            "Defaulting to approve"
        )
        state["feedback"] = "approve"
    
    return state

def writer_node(state):
    api_logger.log_workflow_step("writer", {"summaries_count": len(state["summaries"])})
    
    try:
        final_report = WriterAgent().run(state["summaries"], state["query"])
        state["report"] = final_report
        api_logger.info(f"Final report generated. Length: {len(final_report)}")
    except Exception as e:
        api_logger.log_error_with_recovery(
            e,
            "Error in writer node",
            "Setting default report"
        )
        state["report"] = f"Error generating final report: {str(e)}"
    
    return state

# === Conditional routing ===

def router_feedback(state):
    api_logger.debug(f"Router feedback: {state.get('feedback')}, retry_count: {state.get('retry_count', 0)}, batches_remaining: {len(state.get('chunk_batches', []))}")
    
    # Add current summary to summaries list if it exists and is not empty
    if state.get("current_summary") and state["current_summary"].strip() and state["current_summary"] not in state["summaries"]:
        state["summaries"].append(state["current_summary"])
        api_logger.info(f"Added summary to list. Total summaries: {len(state['summaries'])}")
    
    # Check if we should continue processing or move to writer
    batches_remaining = len(state.get("chunk_batches", []))
    retry_count = state.get("retry_count", 0)
    
    if retry_count >= 5:
        api_logger.info(f"Moving to writer node - max retries reached ({retry_count})")
        return "writer"
    elif batches_remaining == 0:
        api_logger.info("Moving to writer node - no more batches to process")
        return "writer"
    else:
        state["retry_count"] = retry_count + 1
        api_logger.info(f"Continuing to summarisation - retry count: {state['retry_count']}, batches remaining: {batches_remaining}")
        return "summarisation"


# === LangGraph Workflow ===

builder = StateGraph(GraphState)

builder.add_node("research", research_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("summarisation", summarisation_node)
builder.add_node("critic", critic_node)
builder.add_node("writer", writer_node)

builder.set_entry_point("research")

builder.add_edge("research", "retrieval")
builder.add_edge("retrieval", "summarisation")
builder.add_edge("summarisation", "critic")
builder.add_conditional_edges("critic", router_feedback, ["summarisation", "writer"])
builder.add_edge("writer", END)

# Set recursion limit to 5
graph = builder.compile()
