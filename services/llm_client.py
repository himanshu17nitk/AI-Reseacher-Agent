import requests
from typing import List, Dict, Any
from config import API_KEY, BASE_LLM_URL

class RAGError(Exception):
    """Base exception for RAG Chat API errors"""
    pass

class RAGAPIError(RAGError):
    """Exception raised for API-related errors"""
    def __init__(self, status_code: int, message: str, response_text: str = ""):
        self.status_code = status_code
        self.message = message
        self.response_text = response_text
        super().__init__(f"API Error {status_code}: {message}")

class RAGClient:
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_LLM_URL.rstrip("/")  # remove trailing slash if any
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7,
             web_search: bool = False, stream: bool = False, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI Chat API.
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        messages = [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(endpoint, headers=self.headers, json=payload, timeout=100)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                raise RAGAPIError(
                    status_code=e.response.status_code,
                    message="Request failed",
                    response_text=e.response.text
                )
            raise RAGError(f"Network error: {str(e)}")

    def get_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured response data from the API response."""
        try:
            return {
                "model": response.get("model", ""),
                "response": response['choices'][0]['message']['content'],
                "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": response.get("usage", {}).get("completion_tokens", 0)
            }
        except (KeyError, IndexError) as e:
            raise RAGError(f"Invalid response format: {e}")
            