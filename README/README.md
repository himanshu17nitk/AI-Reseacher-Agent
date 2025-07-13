# Research Assistant

An intelligent research assistant that automatically finds, analyzes, and summarizes academic papers to answer your research questions. Built with LangGraph, it uses a multi-agent workflow to provide comprehensive research insights.

## Features

- **Automated Paper Discovery**: Searches arXiv for relevant research papers
- **Smart Document Processing**: Extracts and chunks PDF content intelligently
- **Multi-Agent Workflow**: Research → Retrieval → Summarization → Critique → Writing
- **Vector Search**: Combines BM25 and Qdrant for optimal retrieval
- **Quality Assurance**: Critic agent ensures summary quality
- **Comprehensive Reports**: Generates detailed research summaries
- **Robust Error Handling**: Built-in retry mechanisms and fallback strategies
- **Performance Monitoring**: Detailed logging and error tracking

## Architecture

The system uses a LangGraph workflow with specialized agents:

```
Research Agent → Retrieval Agent → Summarizer Agent → Critic Agent → Writer Agent
     ↓              ↓                  ↓              ↓            ↓
  Find Papers   Extract Chunks   Create Summaries   Validate   Generate Report
```

## Prerequisites

- Python 3.8+
- API key for LLM service (configured in `config.py`)
- Internet connection for arXiv access

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Research-Assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Edit `config.py` with your API credentials:
```python
API_KEY = "your-api-key-here"
BASE_LLM_URL = "your-llm-endpoint"
```

## Quick Start

### Basic Usage

```python
from graph.langgraph_workflow import graph

# Initialize the workflow
workflow = graph

# Define your research query
query = "What are the latest developments in transformer architectures?"

# Create initial state
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
    "report": None
}

# Run the workflow
result = workflow.invoke(initial_state)

# Get the final report
final_report = result["report"]
print(final_report)
```

### Using Individual Agents

```python
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent
from agents.writer_agent import WriterAgent

# Research papers
research_agent = ResearchAgent()
papers = research_agent.run("machine learning applications in healthcare")

# Summarize content
summarizer = SummarizerAgent()
summaries = summarizer.run("ML in healthcare", chunks, feedback="")

# Generate final report
writer = WriterAgent()
report = writer.run(summaries, "ML in healthcare")
```

## API Usage

### Research Workflow

```python
# Complete research workflow
from graph.langgraph_workflow import graph

def research_topic(query: str):
    """Complete research workflow for a given topic."""
    
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
        "report": None
    }
    
    try:
        result = graph.invoke(initial_state)
        return {
            "success": True,
            "report": result["report"],
            "papers_found": len(result["papers"]),
            "summaries_generated": len(result["summaries"])
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
result = research_topic("What are the latest advances in quantum computing?")
if result["success"]:
    print(f"Research completed! Found {result['papers_found']} papers.")
    print(result["report"])
else:
    print(f"Research failed: {result['error']}")
```

### Custom Research Queries

```python
# Example research queries
queries = [
    "Recent developments in natural language processing",
    "Machine learning applications in climate change",
    "Advances in computer vision for autonomous vehicles",
    "Quantum machine learning algorithms",
    "Federated learning privacy and security"
]

for query in queries:
    print(f"Researching: {query}")
    result = research_topic(query)
    if result["success"]:
        print(f"Found {result['papers_found']} papers")
        print(f"Generated {result['summaries_generated']} summaries")
    else:
        print(f"Failed: {result['error']}")
```

## Configuration

### Agent Configuration

```python
# Customize agent behavior
from agents.research_agent import ResearchAgent
from agents.summarizer_agent import SummarizerAgent

# Research agent with custom settings
research_agent = ResearchAgent(
    max_results=20,        # Get more papers
    chunk_size=500,        # Text chunk size in characters
    chunk_overlap=100      # Overlap between chunks
)

# Summarizer with custom batch size
summarizer = SummarizerAgent(
    batch_size=3,      # Smaller batches for detailed processing
    max_attempts=5     # More retry attempts
)
```

### Retrieval Configuration

```python
from retrievers.ensemble_retriever import EnsembleRetriever

# Configure retrieval
retriever = EnsembleRetriever()
retriever.configure(
    bm25_weight=0.6,    # Weight for BM25 retrieval
    vector_weight=0.4   # Weight for vector search
)
```

## Monitoring and Logging

The system includes comprehensive logging:

```python
from utils.logger import api_logger

# Check system status
api_logger.info("Research Assistant started")

# Monitor performance
api_logger.log_performance_metric("research_workflow", 45.2)

# Track errors
api_logger.log_error_with_recovery(
    error, 
    "research_agent", 
    "Retrying with different parameters"
)
```

### Log Files
- `logs/debug.log`: Detailed debug information
- `logs/errors.log`: Error tracking and stack traces
- `logs/api_requests.log`: API call monitoring

## Error Handling

The system includes robust error handling:

```python
from utils.error_handler import (
    ErrorRecoveryStrategy, 
    CircuitBreaker, 
    safe_execute
)

# Retry strategy
recovery = ErrorRecoveryStrategy(max_retries=3, base_delay=2.0)
result = recovery.execute(research_function, query)

# Circuit breaker for external services
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
result = breaker.call(external_api_call)

# Safe execution with fallback
result = safe_execute(risky_function, default_value="fallback")
```

## Project Structure

```
Research Assistant/
├── agents/                 # Multi-agent system
│   ├── research_agent.py   # Paper discovery and processing
│   ├── retriever_agent.py  # Content retrieval
│   ├── summarizer_agent.py # Content summarization
│   ├── critic_agent.py     # Quality validation
│   └── writer_agent.py     # Report generation
├── retrievers/             # Retrieval systems
│   ├── bm25_retriever.py   # BM25 text search
│   ├── qdrant_retriever.py # Vector search
│   └── ensemble_retriever.py # Combined retrieval
├── services/               # External service clients
│   ├── llm_client.py       # LLM API client
│   ├── embedding_client.py # Embedding service
│   └── reranker_client.py  # Reranking service
├── graph/                  # LangGraph workflow
│   └── langgraph_workflow.py
├── utils/                  # Utilities
│   ├── logger.py           # Logging system
│   └── error_handler.py    # Error handling
├── data/                   # Sample data
├── logs/                   # Log files
└── config.py              # Configuration
```

## Use Cases

### Academic Research
```python
# Literature review automation
result = research_topic("Systematic review of deep learning in medical imaging")
```

### Market Research
```python
# Industry trend analysis
result = research_topic("Latest developments in renewable energy technologies")
```

### Technology Assessment
```python
# Technology evaluation
result = research_topic("Comparison of blockchain consensus mechanisms")
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```python
   # Check config.py
   API_KEY = "your-valid-api-key"
   ```

2. **Network Issues**
   ```python
   # Use error handling
   from utils.error_handler import retry_on_exception
   
   @retry_on_exception(max_retries=3)
   def network_call():
       # Your network operation
       pass
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size
   summarizer = SummarizerAgent(batch_size=2)
   ```

### Performance Optimization

```python
# Optimize for speed
research_agent = ResearchAgent(max_results=5)  # Fewer papers
summarizer = SummarizerAgent(batch_size=10)    # Larger batches

# Optimize for quality
research_agent = ResearchAgent(max_results=20) # More papers
summarizer = SummarizerAgent(batch_size=3)     # Smaller batches
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the logs in `logs/` directory
- Review error handling documentation
- Open an issue on GitHub

---

Happy Researching! 