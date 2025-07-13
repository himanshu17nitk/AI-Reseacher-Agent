from langchain.prompts import PromptTemplate


SUMMARISATION_PROMPT_TEMPLATE = """
You are a scientific summarization assistant.

Given the following text chunks from academic papers related to the query: "{query}", your job is to summarize the key insights, technical contributions, and relevant results.

{feedback}

Chunks:
{chunks}

Return a concise, coherent, contextually accurate summary for the above.
"""


input_variables = ["chunks", "query", "feedback"]


SUMMARISATION_PROMPT = PromptTemplate(
    template=SUMMARISATION_PROMPT_TEMPLATE,
    input_variables=input_variables
)