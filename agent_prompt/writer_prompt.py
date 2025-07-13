from langchain.prompts import PromptTemplate

WRITER_PROMPT_TEMPLATE = """
You are a professional academic writer.

Your task is to write a comprehensive research report based on the summaries provided below, for the topic: "{topic}".

Summaries:
{summaries}
"""     


input_variables = ["summaries", "topic"]

WRITER_PROMPT = PromptTemplate(
    template=WRITER_PROMPT_TEMPLATE,
    input_variables=input_variables
)