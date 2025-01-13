import math
import asyncio
from tqdm.asyncio import tqdm as tqdm_async
from models import NetworkXStorage, OpenAIModel, JsonKVStorage
from utils import logger, yes_no_loss_entropy, detect_main_language
from templates import DESCRIPTION_REPHRASING_PROMPT, STATEMENT_JUDGEMENT_PROMPT


async def judge_relations(
        teacher_llm_client: OpenAIModel,
        student_llm_client: OpenAIModel,
        graph_storage: NetworkXStorage,
        rephrase_storage: JsonKVStorage,
        re_judge: bool = False,
        max_samples: int = 1,
        max_concurrent: int = 1000) -> NetworkXStorage:
    """
    Get all edges and judge them

    :param teacher_llm_client: generate statements
    :param student_llm_client: judge the statements to get comprehension loss
    :param graph_storage: graph storage instance
    :param rephrase_storage: rephrase storage instance
    :param re_judge: re-judge the relations
    :param max_samples: max samples for each edge
    :param max_concurrent: max concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_single_relation(
        edge: tuple,
    ):
        async with semaphore:
            source_id = edge[0]
            target_id = edge[1]
            edge_data = edge[2]

            if (not re_judge) and "loss" in edge_data and edge_data["loss"] is not None:
                logger.info(f"Edge {source_id} -> {target_id} already judged, loss: {edge_data['loss']}, skip")
                return source_id, target_id, edge_data

            description = edge_data["description"]
            language = "English" if detect_main_language(description) == "en" else "Chinese"

            try:
                # 如果在rephrase_storage中已经存在，直接取出
                descriptions = await rephrase_storage.get_by_id(description)
                if not descriptions:
                    # 多次采样，取平均
                    descriptions = [(description, 'yes')]
                    for i in range(max_samples):
                        if i > 0:
                            new_description = await teacher_llm_client.generate_answer(
                                DESCRIPTION_REPHRASING_PROMPT[language]['TEMPLATE'].format(input_sentence=description),
                                temperature=1
                            )
                            descriptions.append((new_description, 'yes'))
                        new_anti_description = await teacher_llm_client.generate_answer(
                            DESCRIPTION_REPHRASING_PROMPT[language]['ANTI_TEMPLATE'].format(input_sentence=description),
                            temperature=1
                        )
                        descriptions.append((new_anti_description, 'no'))

                    descriptions = list(set(descriptions))

                    await rephrase_storage.upsert({description: descriptions})

                judgements = []
                gts = [gt for _, gt in descriptions]
                for description, gt in descriptions:
                    judgement = await student_llm_client.generate_topk_per_token(
                        STATEMENT_JUDGEMENT_PROMPT['TEMPLATE'].format(statement=description)
                    )
                    judgements.append(judgement[0].top_candidates)

                loss = yes_no_loss_entropy(judgements, gts)

                logger.info("Edge %s -> %s description: %s loss: %s", source_id, target_id, description, loss)

                edge_data["loss"] = loss
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Error in judging relation {source_id} -> {target_id}: {e}")
                logger.info("Use default loss 0.1")
                edge_data["loss"] = -math.log(0.1)

            await graph_storage.update_edge(source_id, target_id, edge_data)
            return source_id, target_id, edge_data

    edges = await graph_storage.get_all_edges()

    results = []
    for result in tqdm_async(
            asyncio.as_completed([_judge_single_relation(edge) for edge in edges]),
            total=len(edges),
            desc="Judging relations"
    ):
        results.append(await result)

    return graph_storage
