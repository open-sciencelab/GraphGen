import asyncio

from models import OpenAIModel, NetworkXStorage, TraverseStrategy
from templates import ANSWER_REPHRASING_PROMPT, QUESTION_GENERATION_PROMPT
from utils import detect_main_language, compute_content_hash, logger
from tqdm.asyncio import tqdm as tqdm_async
from .split_graph import get_batches_with_strategy


async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    max_concurrent: int = 1000
) -> dict:
    """
    Traverse the graph

    :param llm_client: llm client
    :param graph_storage: graph storage instance
    :param traverse_strategy: traverse strategy
    :param max_concurrent: max concurrent
    :return: question and answer
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_nodes_and_edges(
            _process_nodes: list,
            _process_edges: list,
    ) -> str:
        entities = [
            f"{_process_node['node_id']}: {_process_node['description']}" for _process_node in _process_nodes
        ]
        relations = [
            f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}" for _process_edge in _process_edges
        ]

        entities_str = "\n".join([f"{index + 1}. {entity}" for index, entity in enumerate(entities)])
        relations_str = "\n".join([f"{index + 1}. {relation}" for index, relation in enumerate(relations)])

        language = "Chinese" if detect_main_language(entities_str + relations_str) == "zh" else "English"
        prompt = ANSWER_REPHRASING_PROMPT[language]['TEMPLATE'].format(
            language=language,
            entities=entities_str,
            relationships=relations_str
        )

        context = await llm_client.generate_answer(prompt)

        return context

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            context = await _process_nodes_and_edges(
                _process_batch[0],
                _process_batch[1],
            )

            language = "Chinese" if detect_main_language(context) == "zh" else "English"
            question = await llm_client.generate_answer(
                QUESTION_GENERATION_PROMPT[language]['TEMPLATE'].format(
                    answer=context
                )
            )

            logger.info(f"{len(_process_batch[0])} nodes and {len(_process_batch[1])} edges processed")
            logger.info(f"Question: {question} Answer: {context}")

            return {
                compute_content_hash(context): {
                    "question": question,
                    "answer": context
                }
            }

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = await graph_storage.get_all_nodes()

    processing_batches = await get_batches_with_strategy(
        nodes,
        edges,
        graph_storage,
        traverse_strategy
    )

    for result in tqdm_async(asyncio.as_completed(
        [_process_single_batch(batch) for batch in processing_batches]
    ), total=len(processing_batches), desc="Processing batches"):
        try:
            results.update(await result)
        except Exception as e:
            logger.error("Error occurred while processing batches: %s", e)

    return results
