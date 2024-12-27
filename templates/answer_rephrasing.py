TEMPLATE: str = """---Role---

You are a helpful assistant responsible for generating a logically structured and coherent rephrased version of the TEXT based on ENTITIES and RELATIONSHIPS provided below.
Use {language} as output language.

---Goal---

To generate a version of the text that is rephrased and conveys the same meaning as the original entity and relationship descriptions, while:
1. Following a clear logical flow and structure
2. Establishing proper cause-and-effect relationships
3. Ensuring temporal and sequential consistency
4. Creating smooth transitions between ideas using conjunctions and appropriate linking words like "firstly," "however," "therefore," etc.

---Instructions---
1. Analyze the provided ENTITIES and RELATIONSHIPS carefully to identify:
   - Key concepts and their hierarchies
   - Temporal sequences and chronological order
   - Cause-and-effect relationships
   - Dependencies between different elements

2. Organize the information in a logical sequence by:
   - Starting with foundational concepts
   - Building up to more complex relationships
   - Grouping related ideas together
   - Creating clear transitions between sections

3. Rephrase the text while maintaining:
   - Logical flow and progression
   - Clear connections between ideas
   - Proper context and background
   - Coherent narrative structure

4. Review and refine the text to ensure:
   - Logical consistency throughout
   - Natural progression of ideas
   - Clear cause-and-effect relationships
   - Smooth transitions between concepts

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
