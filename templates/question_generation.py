TEMPLATE_EN: str = """The answer to a question is provided. Please generate a question that corresponds to the answer.

################
Answer:
{answer}
################
Question: 
"""

TEMPLATE_ZH: str = """下面提供了一个问题的答案，请生成一个与答案对应的问题。

################
答案：
{answer}
################
问题：
"""


QUESTION_GENERATION_PROMPT = {
    "English": {
        "TEMPLATE": TEMPLATE_EN
    },
    "Chinese": {
        "TEMPLATE": TEMPLATE_ZH
    }
}
