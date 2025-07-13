# agents/writer_agent.py

from typing import List, Optional
from services.llm_client import LLMClient
from agent_prompt.writer_prompt import WRITER_PROMPT


class WriterAgent:
    def __init__(self):
        self.llm = LLMClient()

    def run(
        self,
        summaries: List[str],
        topic: str,
        report_structure: Optional[List[str]] = None
    ) -> str:
        prompt = WRITER_PROMPT.format(summaries=summaries, topic=topic)

        if report_structure:
            sections = "\n".join([f"- {section}" for section in report_structure])
            prompt += f""" Structure the report using the following sections: 
            {sections}

            Write it in a formal academic tone, suitable for publication or presentation.
            """
            
        response_json = self.llm.chat(prompt=prompt)
        return self.llm.get_response(response_json)['response']
