from typing import List
from models import Chunk
from models import OpenAIModel
from templates import CONFERENCE_RESOLUTION_TEMPLATE
from utils import detect_main_language

async def resolute_conference(
        llm_client: OpenAIModel,
        chunks: List[Chunk]) -> List[Chunk]:
    """
    Resolute conference

    :param llm_client: LLM model
    :param chunks: List of chunks
    :return: List of chunks
    """

    if len(chunks) == 0:
        return chunks

    results = [chunks[0]]

    for _, chunk in enumerate(chunks[1:]):
        language = detect_main_language(chunk.content)
        result = await llm_client.generate_answer(
            CONFERENCE_RESOLUTION_TEMPLATE[language].format(
                reference = results[0].content,
                input_sentence = chunk.content
            )
        )
        results.append(Chunk(id=chunk.id, content=result))

    return results
