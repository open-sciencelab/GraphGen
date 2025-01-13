import asyncio

from tqdm.asyncio import tqdm as tqdm_async
from models import JsonKVStorage, OpenAIModel, NetworkXStorage
from utils import logger, detect_main_language
from templates import DESCRIPTION_REPHRASING_PROMPT


async def quiz_relations(
        teacher_llm_client: OpenAIModel,
        graph_storage: NetworkXStorage,
        rephrase_storage: JsonKVStorage,
        max_samples: int = 1,
        max_concurrent: int = 1000) -> JsonKVStorage:
    """
    Get all edges and quiz them

    :param teacher_llm_client: generate statements
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param max_samples: max samples for each edge
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _quiz_single_relation(
        edge: tuple,
    ):
        async with semaphore:
            source_id = edge[0]
            target_id = edge[1]
            edge_data = edge[2]

            description = edge_data["description"]
            language = "English" if detect_main_language(description) == "en" else "Chinese"

            try:
                # 如果在rephrase_storage中已经存在，直接取出
                descriptions = await rephrase_storage.get_by_id(description)
                if not descriptions:
                    # 多次采样，取平均
                    descriptions = [(description, 'yes')]

                    new_description_tasks = []
                    new_anti_description_tasks = []
                    for i in range(max_samples):
                        if i > 0:
                            new_description_tasks.append(
                                teacher_llm_client.generate_answer(
                                    DESCRIPTION_REPHRASING_PROMPT[language]['TEMPLATE'].format(
                                        input_sentence=description),
                                    temperature=1
                                )
                            )
                        new_anti_description_tasks.append(
                            teacher_llm_client.generate_answer(
                                DESCRIPTION_REPHRASING_PROMPT[language]['ANTI_TEMPLATE'].format(
                                    input_sentence=description),
                                temperature=1
                            )
                        )

                    new_descriptions = await asyncio.gather(*new_description_tasks)
                    new_anti_descriptions = await asyncio.gather(*new_anti_description_tasks)

                    for new_description in new_descriptions:
                        descriptions.append((new_description, 'yes'))
                    for new_anti_description in new_anti_descriptions:
                        descriptions.append((new_anti_description, 'no'))

                    descriptions = list(set(descriptions))
            except Exception as e: # pylint: disable=broad-except
                logger.error("Error when quizzing edge %s -> %s: %s", source_id, target_id, e)
                descriptions = [(description, 'yes')]

            await rephrase_storage.upsert({description: descriptions})

            return {description: descriptions}


    edges = await graph_storage.get_all_edges()

    results = []
    for result in tqdm_async(
            asyncio.as_completed([_quiz_single_relation(edge) for edge in edges]),
            total=len(edges),
            desc="Quizzing relations"
    ):
        results.append(await result)

    return rephrase_storage
