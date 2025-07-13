# Error Handling and Logging System

This document describes the comprehensive error handling and logging system implemented in the Research Assistant project.

## Overview

The system now includes:
- **Enhanced Logging**: Multi-level logging with rotation and different handlers
- **Error Recovery**: Retry mechanisms with exponential backoff
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Error Monitoring**: Centralized error tracking and analysis
- **Fallback Mechanisms**: Graceful degradation when services fail

## Components

### 1. Enhanced Logger (`utils/logger.py`)

#### Features:
- **Multi-level logging**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Rotating file handlers**: Prevents log files from growing too large
- **Separate log files**: Different files for different purposes
  - `debug.log`: All debug information
  - `errors.log`: Error details with stack traces
  - `api_requests.log`: API call tracking

#### Usage:
```python
from utils.logger import api_logger

# Basic logging
api_logger.debug("Debug message")
api_logger.info("Info message")
api_logger.warning("Warning message")
api_logger.error("Error message", exc_info=True)

# Specialized logging
api_logger.log_agent_activity("AgentName", "action", details="info")
api_logger.log_retrieval_activity("BM25", "query", 10)
api_logger.log_workflow_step("step_name", {"state": "info"})
api_logger.log_performance_metric("operation", 1.5)
```

#### Decorators:
```python
from utils.logger import log_execution_time, log_retry_attempt

@log_execution_time(api_logger, "Operation Name")
@log_retry_attempt(api_logger, max_retries=3, delay=1.0)
def my_function():
    # Function implementation
    pass
```

### 2. Error Handler (`utils/error_handler.py`)

#### Components:

##### ErrorRecoveryStrategy
- Implements retry logic with exponential backoff
- Configurable retry attempts and delays

##### CircuitBreaker
- Prevents cascading failures
- Three states: CLOSED, OPEN, HALF_OPEN
- Automatic recovery after timeout

##### ErrorMonitor
- Tracks error counts and history
- Provides error summaries for analysis
- Maintains error history for debugging

#### Usage:
```python
from utils.error_handler import (
    ErrorRecoveryStrategy, CircuitBreaker, ErrorMonitor,
    handle_errors, retry_on_exception, safe_execute, ErrorContext
)

# Retry strategy
recovery = ErrorRecoveryStrategy(max_retries=3, base_delay=1.0)
result = recovery.execute(my_function, arg1, arg2)

# Circuit breaker
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
result = breaker.call(my_function, arg1, arg2)

# Decorators
@handle_errors(fallback_value="default")
def risky_function():
    pass

@retry_on_exception(exceptions=(ValueError,), max_retries=3)
def function_with_retry():
    pass

# Safe execution
result = safe_execute(my_function, default_value="fallback")

# Context manager
with ErrorContext("operation_name", cleanup_func=cleanup):
    # Risky operation
    pass
```

### 3. Enhanced LLM Client (`services/llm_client.py`)

#### Improvements:
- **Comprehensive error handling**: Different exception types for different errors
- **Request/response logging**: Detailed API call tracking
- **Response validation**: Ensures response format is correct
- **Performance monitoring**: Tracks response times
- **Retry mechanisms**: Automatic retry with exponential backoff

#### Error Types:
- `LLMError`: Base exception class
- `LLMAPIError`: API-related errors (status codes, response text)
- `LLMTimeoutError`: Timeout errors

#### Usage:
```python
from services.llm_client import LLMClient

client = LLMClient()

try:
    response = client.chat("Your prompt here")
    result = client.get_response(response)
except LLMAPIError as e:
    print(f"API Error: {e.status_code} - {e.message}")
except LLMTimeoutError as e:
    print(f"Timeout: {e}")
```

### 4. Enhanced Summarizer Agent (`agents/summarizer_agent.py`)

#### Features:
- **Input validation**: Validates query and chunks before processing
- **Batch processing**: Processes chunks in configurable batches
- **Fallback summaries**: Creates summaries when LLM fails
- **Comprehensive logging**: Tracks all agent activities
- **Error recovery**: Retries failed operations

#### Error Handling:
- Validates input before processing
- Retries failed LLM calls
- Creates fallback summaries when LLM fails
- Logs all activities and errors

## Configuration

### Logging Configuration
Logs are stored in the `logs/` directory:
- `debug.log`: All debug information (rotates at 10MB, keeps 5 backups)
- `errors.log`: Error details (rotates at 5MB, keeps 3 backups)
- `api_requests.log`: API call tracking (rotates at 10MB, keeps 5 backups)

### Error Monitoring
- Error history is maintained in memory (max 1000 entries)
- Error counts are tracked by type
- Error summaries available for analysis

## Best Practices

### 1. Logging
- Use appropriate log levels
- Include context in log messages
- Use structured logging for complex data
- Avoid logging sensitive information

### 2. Error Handling
- Always catch specific exceptions when possible
- Provide meaningful error messages
- Implement fallback mechanisms
- Use retry logic for transient failures

### 3. Performance
- Use decorators for automatic timing and retry logic
- Monitor performance metrics
- Implement circuit breakers for external services
- Use exponential backoff for retries

### 4. Monitoring
- Regularly check error logs
- Monitor error rates and types
- Set up alerts for critical errors
- Review performance metrics

## Updated Requirements

The `requirements.txt` file has been updated to include all necessary dependencies:

```txt
# Core AI/ML Libraries
openai>=1.0.0
langgraph>=0.2.0
langchain>=0.2.0
langchain-community>=0.2.0
tiktoken>=0.5.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0

# Vector Database
qdrant-client>=1.7.0
rank-bm25>=0.2.2

# Document Processing
arxiv>=2.1.0
PyPDF2>=3.0.0
PyMuPDF>=1.23.0  # fitz

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0

# Database
pymongo>=4.6.0

# HTTP Client
requests>=2.31.0

# Data Validation
pydantic>=2.5.0

# Utilities
python-dotenv>=1.0.0
```

## Usage Examples

### Basic Error Handling
```python
from utils.logger import api_logger
from utils.error_handler import safe_execute

# Safe function execution
result = safe_execute(risky_function, default_value="fallback")

# Logging with context
api_logger.log_agent_activity("ResearchAgent", "paper_search", query="AI research")
```

### Advanced Error Handling
```python
from utils.error_handler import ErrorContext, CircuitBreaker

# Circuit breaker for external service
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

# Context manager with cleanup
with ErrorContext("database_operation", cleanup_func=cleanup_resources):
    result = breaker.call(database_operation)
```

### Performance Monitoring
```python
from utils.logger import log_execution_time

@log_execution_time(api_logger, "Data Processing")
def process_data():
    # Data processing logic
    pass
```

## Monitoring and Alerting

### Error Monitoring
- Check `logs/errors.log` for error details
- Monitor error rates using `ErrorMonitor.get_error_summary()`
- Set up alerts for high error rates

### Performance Monitoring
- Monitor response times in `logs/debug.log`
- Track API call performance
- Monitor memory usage for components

### Health Checks
- Circuit breaker status
- Error rate thresholds
- Response time thresholds
- Service availability

This comprehensive error handling and logging system ensures the Research Assistant is robust, monitorable, and maintainable. 