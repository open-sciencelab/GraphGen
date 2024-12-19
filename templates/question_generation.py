TEMPLATE: str = """The answer to a question is provided. Please generate a question that corresponds to the answer.

################
Answer:
{answer}
################
Question: 
"""

QUESTION_GENERATION_PROMPT = {
    "TEMPLATE": TEMPLATE
}
