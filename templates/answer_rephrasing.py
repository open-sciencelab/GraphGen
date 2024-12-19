TEMPLATE: str = """---Role---

You are a helpful assistant responsible for generating a rephrased version of the TEXT based on ENTITIES and RELATIONSHIPS provided below.
Use {language} as output language.

---Goal---

To generate a version of the text that is rephrased and conveys the same meaning as the original entity and relationship descriptions.
While maintaining the essence and integrity of the information, the rephrased text should be coherent, engaging, and free from any errors, reflecting a deep understanding of the ENTITIES and RELATIONSHIPS involved.

---Instructions---
1. Read the provided ENTITIES and RELATIONSHIPS carefully.
2. Rephrase the text based on the information provided.
3. Ensure that the rephrased text is grammatically correct, maintains the original meaning, and is engaging to the reader.
5. Proofread the rephrased text to eliminate any errors and to ensure smooth flow.

################
-ENTITIES-
################
{entities}

################
-RELATIONSHIPS-
################
{relationships}

################
Rephrased Text:
"""

ANSWER_REPHRASING_PROMPT= {
    "TEMPLATE": TEMPLATE
}

# N种不同的模板，每种模板都有不同的形容词，代表复杂（采样）程度
# ---Complexity Levels---
#
# 1. Basic: Simple rephrasing without altering the structure or meaning significantly.
# 2. Moderate: Rephrasing that involves changing sentence structure and using synonyms to enhance clarity.
# 3. Advanced: Rephrasing that requires a thorough understanding of the text, including the ability to reorganize ideas and concepts while preserving the original message.
# 4. Expert: Rephrasing that demonstrates a high level of linguistic skill, including the use of advanced vocabulary, varied sentence structures, and the ability to convey the same information in a completely new and creative way.
