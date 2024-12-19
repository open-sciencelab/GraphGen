import asyncio
from models import NetworkXStorage
from utils import logger, yes_no_loss
from templates import ANTI_DESCRIPTION_REPHRASING_PROMPT, STATEMENT_JUDGEMENT_PROMPT
from models import OpenAIModel
from tqdm.asyncio import tqdm as tqdm_async


async def judge_relations(llm_client: OpenAIModel, graph_storage: NetworkXStorage) -> NetworkXStorage:
    """
    Get all edges and judge them

    :param llm_client: llm client
    :param graph_storage: graph storage instance
    :return:
    """

    async def _judge_single_relation(
        edge: tuple,
    ):
        source_id = edge[0]
        target_id = edge[1]
        edge_data = edge[2]
        description = edge_data["description"]

        anti_description = await llm_client.generate_answer(
            ANTI_DESCRIPTION_REPHRASING_PROMPT['TEMPLATE'].format(input_sentence=description)
        )

        judgement = await llm_client.generate_topk_per_token(
            STATEMENT_JUDGEMENT_PROMPT['TEMPLATE'].format(statement=description)
        )
        anti_judgement = await llm_client.generate_topk_per_token(
            STATEMENT_JUDGEMENT_PROMPT['TEMPLATE'].format(statement=anti_description)
        )

        loss = yes_no_loss(
            [judgement[0].top_candidates, anti_judgement[0].top_candidates],
            ['yes', 'no']
        )

        logger.info(f"Edge {source_id} -> {target_id} description: {description} loss: {loss}")

        # 将loss加入到边的属性中
        edge_data["loss"] = loss

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
