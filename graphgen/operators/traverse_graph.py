import asyncio

from models import OpenAIModel, NetworkXStorage
from templates import ANSWER_REPHRASING_PROMPT, QUESTION_GENERATION_PROMPT
from utils import detect_main_language, compute_content_hash, logger
from tqdm.asyncio import tqdm as tqdm_async


async def _get_node_info(
    node_id: str,
    graph_storage: NetworkXStorage,
)-> dict:
    """
    Get node info

    :param node_id: node id
    :param graph_storage: graph storage instance
    :return: node info
    """
    node_data = await graph_storage.get_node(node_id)
    return {
        "node_id": node_id,
        **node_data
    }


def _get_level_2_edges(
    edges: list,
    node_id: str,
    top_extra_edges: int
) -> list:
    """
    Get level 2 edges

    :param edges: find edges
    :param node_id: source node id
    :param top_extra_edges: top extra edges
    :return: level 2 edges
    """
    level_2_edges = []
    for edge in edges:
        if "visited" in edge[2] and edge[2]["visited"]:
            continue
        if edge[0] == node_id:
            level_2_edges.append(edge)
            edge[2]["visited"] = True
        elif edge[1] == node_id:
            level_2_edges.append((edge[1], edge[0], edge[2]))
            edge[2]["visited"] = True
        if len(level_2_edges) >= top_extra_edges:
            break
    return level_2_edges


async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    graph_storage: NetworkXStorage,
    top_extra_edges: int = 5,
    max_concurrent: int = 1000
) -> dict:
    """
    Traverse the graph

    :param llm_client: llm client
    :param graph_storage: graph storage instance
    :param top_extra_edges: top extra edges
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
            f"{_process_edge[0]} -> {_process_edge[1]}: {_process_edge[2]['description']}" for _process_edge in _process_edges
        ]

        entities_str = "\n".join([f"{i + 1}. {entity}" for i, entity in enumerate(entities)])
        relations_str = "\n".join([f"{i + 1}. {relation}" for i, relation in enumerate(relations)])

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

    # 按照loss从大到小排序
    edges = sorted(edges, key=lambda x: x[2]["loss"], reverse=True)

    processing_batches = []
    node_cache = {}

    async def get_cached_node_info(node_id):
        if node_id not in node_cache:
            node_cache[node_id] = await _get_node_info(node_id, graph_storage)
        return node_cache[node_id]

    for edge in tqdm_async(edges, desc="Preparing batches"):
        if "visited" in edge[2] and edge[2]["visited"]:
            continue

        edge[2]["visited"] = True

        _process_nodes = []
        _process_edges = []

        src_id = edge[0]
        tgt_id = edge[1]

        _process_nodes.extend([await get_cached_node_info(src_id), await get_cached_node_info(tgt_id)])
        _process_edges.append(edge)

        level_2_edges = _get_level_2_edges(edges, tgt_id, top_extra_edges)
        assert len(level_2_edges) <= top_extra_edges

        for _edge in level_2_edges:
            _process_nodes.append(await get_cached_node_info(_edge[1]))
            _process_edges.append(_edge)

        processing_batches.append((_process_nodes, _process_edges))




    # isolate nodes
    visited_nodes = set()
    for _process_nodes, _process_edges in processing_batches:
        for node in _process_nodes:
            visited_nodes.add(node["node_id"])
    for node in nodes:
        if node[0] not in visited_nodes:
            _process_nodes = [await _get_node_info(node[0], graph_storage)]
            processing_batches.append((_process_nodes, []))

    for result in tqdm_async(asyncio.as_completed(
        [_process_single_batch(batch) for batch in processing_batches]
    ), total=len(processing_batches), desc="Processing batches"):
        try:
            results.update(await result)
        except Exception as e:
            logger.error("Error occurred while processing batches: %s", e)

    return results
