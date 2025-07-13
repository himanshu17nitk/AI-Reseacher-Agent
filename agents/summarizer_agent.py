from typing import Dict, List, Optional
from services.llm_client import LLMClient, LLMError, LLMAPIError, LLMTimeoutError
from agent_prompt.summarisation_prompt import SUMMARISATION_PROMPT
from agents.critic_agent import CriticAgent
from utils.logger import api_logger, log_execution_time, log_retry_attempt


class SummarizerAgent:
    def __init__(self, batch_size: int = 5, max_attempts: int = 3):
        self.llm = LLMClient()
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        api_logger.info(f"SummarizerAgent initialized with batch_size={batch_size}, max_attempts={max_attempts}")

    @log_execution_time(api_logger, "Summarize Batch")
    def _summarise_batch(self, chunks: List[str], query: str, feedback: str) -> str:
        """Summarize a batch of text chunks with error handling and logging."""
        try:
            chunks_prompt = f"Chunks: {chr(10).join([f'- {chunk}' for chunk in chunks])}"
            prompt = SUMMARISATION_PROMPT.format(chunks=chunks_prompt, query=query, feedback=feedback)

            api_logger.log_agent_activity(
                "SummarizerAgent",
                "summarize_batch",
                chunks_count=len(chunks),
                query_length=len(query),
                feedback_length=len(feedback)
            )

            response_json = self.llm.chat(prompt=prompt)
            
            # Validate response before processing
            if not self.llm.validate_response(response_json):
                raise LLMError("Invalid response format from LLM")
            
            result = self.llm.get_response(response_json)
            summary = result['response']
            
            api_logger.debug(f"Batch summarization successful. Summary length: {len(summary)}")
            return summary
            
        except (LLMAPIError, LLMTimeoutError) as e:
            api_logger.log_error_with_recovery(
                e,
                f"LLM error in batch summarization",
                "Will retry with exponential backoff"
            )
            raise
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                f"Unexpected error in batch summarization",
                "Check prompt format and LLM configuration"
            )
            raise LLMError(f"Summarization failed: {str(e)}")

    @log_execution_time(api_logger, "Summarizer Agent Run")
    def run(self, query: str, chunks: List[str], feedback: Optional[str] = "") -> List[str]:
        """Run the summarization process with comprehensive error handling."""
        try:
            api_logger.log_agent_activity(
                "SummarizerAgent",
                "run",
                query_length=len(query),
                total_chunks=len(chunks),
                batch_size=self.batch_size
            )
            
            if not chunks:
                api_logger.warning("No chunks provided for summarization")
                return []
            
            if not query.strip():
                api_logger.warning("Empty query provided for summarization")
                return []
            
            summaries = []
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(chunks), self.batch_size):
                batch_num = (i // self.batch_size) + 1
                batch = chunks[i:i + self.batch_size]
                
                api_logger.debug(f"Processing batch {batch_num}/{total_batches} with {len(batch)} chunks")
                
                summary = ""
                feedback_str = feedback if feedback is not None else ""
                
                # Retry logic for each batch
                for attempt in range(self.max_attempts):
                    try:
                        summary = self._summarise_batch(batch, query, feedback_str)
                        api_logger.debug(f"Batch {batch_num} summarized successfully on attempt {attempt + 1}")
                        break
                    except (LLMAPIError, LLMTimeoutError) as e:
                        if attempt == self.max_attempts - 1:
                            api_logger.error(f"Failed to summarize batch {batch_num} after {self.max_attempts} attempts")
                            # Create a fallback summary
                            summary = self._create_fallback_summary(batch, query)
                        else:
                            api_logger.warning(f"Attempt {attempt + 1} failed for batch {batch_num}, retrying...")
                            continue
                    except Exception as e:
                        api_logger.log_error_with_recovery(
                            e,
                            f"Unexpected error in batch {batch_num}",
                            "Creating fallback summary"
                        )
                        summary = self._create_fallback_summary(batch, query)
                        break
                
                summaries.append(summary)
                api_logger.debug(f"Batch {batch_num} completed. Summary length: {len(summary)}")
            
            api_logger.info(f"Summarization completed. Generated {len(summaries)} summaries")
            return summaries
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Critical error in summarizer agent run",
                "Returning empty summaries list"
            )
            return []

    def _create_fallback_summary(self, chunks: List[str], query: str) -> str:
        """Create a fallback summary when LLM fails."""
        try:
            api_logger.warning("Creating fallback summary due to LLM failure")
            
            # Simple concatenation of first sentences from each chunk
            fallback_parts = []
            for chunk in chunks:
                sentences = chunk.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if first_sentence:
                        fallback_parts.append(first_sentence)
            
            if fallback_parts:
                fallback_summary = ". ".join(fallback_parts[:3]) + "."
                api_logger.debug(f"Fallback summary created with length: {len(fallback_summary)}")
                return fallback_summary
            else:
                api_logger.warning("Could not create fallback summary - no valid content found")
                return "Unable to generate summary due to processing error."
                
        except Exception as e:
            api_logger.error(f"Error creating fallback summary: {e}")
            return "Summary generation failed."

    def validate_input(self, query: str, chunks: List[str]) -> bool:
        """Validate input parameters before processing."""
        try:
            if not query or not query.strip():
                api_logger.warning("Query is empty or contains only whitespace")
                return False
            
            if not chunks:
                api_logger.warning("No chunks provided for summarization")
                return False
            
            # Check for valid chunk content
            valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
            if len(valid_chunks) != len(chunks):
                api_logger.warning(f"Found {len(chunks) - len(valid_chunks)} empty chunks")
            
            api_logger.debug(f"Input validation successful. Query length: {len(query)}, Valid chunks: {len(valid_chunks)}")
            return True
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Error in input validation",
                "Input validation failed"
            )
            return False

    def run_single_batch(self, query: str, chunks: List[str], feedback: Optional[str] = "") -> str:
        """Run summarization for a single batch and return a single summary string."""
        try:
            api_logger.log_agent_activity(
                "SummarizerAgent",
                "run_single_batch",
                query_length=len(query),
                chunks_count=len(chunks)
            )
            
            if not chunks:
                api_logger.warning("No chunks provided for summarization")
                return "No content available for summarization."
            
            if not query.strip():
                api_logger.warning("Empty query provided for summarization")
                return "Query is required for summarization."
            
            feedback_str = feedback if feedback is not None else ""
            
            # Retry logic for the batch
            for attempt in range(self.max_attempts):
                try:
                    summary = self._summarise_batch(chunks, query, feedback_str)
                    api_logger.debug(f"Single batch summarized successfully on attempt {attempt + 1}")
                    return summary
                except (LLMAPIError, LLMTimeoutError) as e:
                    if attempt == self.max_attempts - 1:
                        api_logger.error(f"Failed to summarize batch after {self.max_attempts} attempts")
                        return self._create_fallback_summary(chunks, query)
                    else:
                        api_logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                        continue
                except Exception as e:
                    api_logger.log_error_with_recovery(
                        e,
                        "Unexpected error in single batch summarization",
                        "Creating fallback summary"
                    )
                    return self._create_fallback_summary(chunks, query)
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Critical error in single batch summarizer agent run",
                "Returning error message"
            )
            return f"Error during summarization: {str(e)}"
        
        # Fallback return in case all retries fail
        return "Failed to generate summary after all attempts."
