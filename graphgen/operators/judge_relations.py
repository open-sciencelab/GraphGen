import asyncio
from models import NetworkXStorage
from utils import logger, yes_no_loss, detect_main_language
from templates import ANTI_DESCRIPTION_REPHRASING_PROMPT, STATEMENT_JUDGEMENT_PROMPT
from models import OpenAIModel
from tqdm.asyncio import tqdm as tqdm_async


async def judge_relations(
        teacher_llm_client: OpenAIModel,
        student_llm_client: OpenAIModel,
        graph_storage: NetworkXStorage,
        re_judge: bool = False,
        max_concurrent: int = 1000) -> NetworkXStorage:
    """
    Get all edges and judge them

    :param teacher_llm_client: generate statements
    :param student_llm_client: judge the statements to get comprehension loss
    :param graph_storage: graph storage instance
    :param re_judge: re-judge the relations
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
                anti_description = await teacher_llm_client.generate_answer(
                    ANTI_DESCRIPTION_REPHRASING_PROMPT[language]['TEMPLATE'].format(input_sentence=description)
                )

                judgement = await student_llm_client.generate_topk_per_token(
                    STATEMENT_JUDGEMENT_PROMPT['TEMPLATE'].format(statement=description)
                )
                anti_judgement = await student_llm_client.generate_topk_per_token(
                    STATEMENT_JUDGEMENT_PROMPT['TEMPLATE'].format(statement=anti_description)
                )

                loss = yes_no_loss(
                    [judgement[0].top_candidates, anti_judgement[0].top_candidates],
                    ['yes', 'no']
                )

                logger.info(f"Edge {source_id} -> {target_id} description: {description} loss: {loss}")

                edge_data["loss"] = loss
            except Exception as e:
                logger.error(f"Error in judging relation {source_id} -> {target_id}: {e}")
                logger.info("Use default loss 0.1")
                edge_data["loss"] = 0.1

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
