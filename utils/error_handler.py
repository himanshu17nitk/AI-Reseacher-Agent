"""
Error handling and recovery utilities for the Research Assistant system.
Provides centralized error handling, recovery strategies, and monitoring.
"""

import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from utils.logger import api_logger


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_count = 0
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if the error should trigger a retry."""
        return self.retry_count < self.max_retries
    
    def get_delay(self) -> float:
        """Calculate delay for next retry (exponential backoff)."""
        return self.base_delay * (2 ** self.retry_count)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        while self.retry_count < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.retry_count += 1
                api_logger.warning(f"Attempt {self.retry_count} failed: {str(e)}")
                
                if self.should_retry(e):
                    delay = self.get_delay()
                    api_logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    api_logger.error(f"All {self.max_retries} attempts failed")
                    raise


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                api_logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                api_logger.info("Circuit breaker reset to CLOSED")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                api_logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


class ErrorMonitor:
    """Monitor and track errors for analysis and alerting."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def record_error(self, error: Exception, context: str, **kwargs):
        """Record an error for monitoring."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            **kwargs
        }
        
        self.error_history.append(error_record)
        
        # Keep history size manageable
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        api_logger.error(f"Error recorded: {error_type} in {context}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_counts.clear()
        self.error_history.clear()
        api_logger.info("Error history cleared")


# Global instances
error_monitor = ErrorMonitor()


def handle_errors(
    recovery_strategy: Optional[ErrorRecoveryStrategy] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    fallback_value: Any = None,
    log_errors: bool = True
):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                elif recovery_strategy:
                    return recovery_strategy.execute(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    error_monitor.record_error(e, f"{func.__module__}.{func.__name__}")
                
                if fallback_value is not None:
                    api_logger.warning(f"Using fallback value due to error: {str(e)}")
                    return fallback_value
                else:
                    raise
        return wrapper
    return decorator


def retry_on_exception(
    exceptions: Union[Type[Exception], tuple] = Exception,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Decorator for retrying on specific exceptions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    api_logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff_factor ** attempt)
                        api_logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
            
            api_logger.error(f"All {max_retries} attempts failed")
            if last_exception:
                raise last_exception
            else:
                raise Exception(f"All {max_retries} attempts failed")
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_value: Any = None, **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_monitor.record_error(e, f"{func.__module__}.{func.__name__}")
        api_logger.error(f"Safe execution failed: {str(e)}")
        return default_value


class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(self, context_name: str, cleanup_func: Optional[Callable] = None):
        self.context_name = context_name
        self.cleanup_func = cleanup_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        api_logger.debug(f"Entering error context: {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - (self.start_time or 0)
        
        if exc_type is not None:
            error_monitor.record_error(
                exc_val, 
                self.context_name,
                duration=duration
            )
            api_logger.error(f"Error in context {self.context_name}: {str(exc_val)}")
        
        if self.cleanup_func:
            try:
                self.cleanup_func()
                api_logger.debug(f"Cleanup completed for {self.context_name}")
            except Exception as e:
                api_logger.error(f"Cleanup failed for {self.context_name}: {str(e)}")
        
        api_logger.debug(f"Exiting error context: {self.context_name} (duration: {duration:.3f}s)")
        return False  # Re-raise the exception


def validate_input(data: Any, validation_func: Callable, error_message: str = "Invalid input"):
    """Validate input data with error handling."""
    try:
        if not validation_func(data):
            raise ValueError(error_message)
        return True
    except Exception as e:
        error_monitor.record_error(e, "input_validation")
        api_logger.error(f"Input validation failed: {str(e)}")
        return False 