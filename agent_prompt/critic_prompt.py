from langchain.prompts import PromptTemplate

CRITIC_PROMPT_TEMPLATE = """
You are a rigorous research critic.

Your task is to evaluate whether the following summary faithfully captures all relevant and important information from the given text chunks in relation to the query: "{query}".

Return a structured JSON object:
{"verdict": "approve" | "reject", "feedback": "<only required if verdict is reject>"}

Summary:
{summary}

Chunks:
{original_chunks}
"""

input_variables = ["summary", "original_chunks", "query"]

CRITIC_PROMPT = PromptTemplate(
    template=CRITIC_PROMPT_TEMPLATE,
    input_variables=input_variables
)