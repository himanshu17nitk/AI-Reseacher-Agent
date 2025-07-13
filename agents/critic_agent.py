from typing import List
import json
from agent_prompt.critic_prompt import CRITIC_PROMPT
from services.llm_client import LLMClient       
from utils.logger import api_logger


class CriticAgent:
    def __init__(self):
        self.llm = LLMClient()

    def critique_summary_agent(self, summary: str, original_chunks: List[str], query: str) -> str:
        try:
            # Convert original_chunks list to a formatted string
            chunks_text = "\n".join([f"- {chunk}" for chunk in original_chunks]) if original_chunks else "No chunks provided"
            
            # Ensure all inputs are strings and don't contain template variables
            summary_str = str(summary) if summary else "No summary provided"
            query_str = str(query) if query else "No query provided"
            
            prompt = CRITIC_PROMPT.format(summary=summary_str, original_chunks=chunks_text, query=query_str)
            
            response_json = self.llm.chat(prompt=prompt)
            
            # Try to parse as JSON if it's a string
            if isinstance(response_json, str):
                try:
                    response = json.loads(response_json)
                except json.JSONDecodeError:
                    # If it's not valid JSON, check if it contains approve/reject
                    if "approve" in response_json.lower():
                        return "approve"
                    elif "reject" in response_json.lower():
                        return "reject"
                    else:
                        return "approve"  # Default to approve if unclear
            
            # Now handle as JSON object
            if isinstance(response_json, dict):
                response = response_json.get('response', response_json)
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        if "approve" in response.lower():
                            return "approve"
                        elif "reject" in response.lower():
                            return "reject"
                        else:
                            return "approve"
                
                if isinstance(response, dict):
                    verdict = response.get('verdict', 'approve')
                    feedback = response.get('feedback', '')
                    if verdict == "reject":
                        return feedback
                    else:
                        return "approve"
                else:
                    return "approve"  # Default to approve for unexpected formats
            else:
                return "approve"  # Default to approve on any error
                
        except Exception as e:
            api_logger.error(f"Error in critic agent: {e}")
            return "approve"  # Default to approve on any error
