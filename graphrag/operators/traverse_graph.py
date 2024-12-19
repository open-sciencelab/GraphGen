from models import OpenAIModel, NetworkXStorage
from templates import ANSWER_REPHRASING_PROMPT, QUESTION_GENERATION_PROMPT
from utils import detect_main_language, compute_content_hash, logger


async def _process_nodes_and_edges(
        nodes: list,
        edges: list,
        llm_client: OpenAIModel
) -> str:
    """
    Process a single link

    :param nodes: nodes
    :param edges: edges
    :param llm_client: llm client
    :return: context str
    """

    entities = [
        f"{node['node_id']}: {node['description']}" for node in nodes
    ]
    relations = [
        f"{edge[0]} -> {edge[1]}: {edge[2]['description']}" for edge in edges
    ]

    entities_str = "\n".join([f"{i+1}. {entity}" for i, entity in enumerate(entities)])
    relations_str = "\n".join([f"{i+1}. {relation}" for i, relation in enumerate(relations)])

    prompt = ANSWER_REPHRASING_PROMPT['TEMPLATE'].format(
        language = "Chinese" if detect_main_language(entities_str + relations_str) == "zh" else "English",
        entities=entities_str,
        relationships=relations_str
    )

    context = await llm_client.generate_answer(prompt)

    return context

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
    edges = sorted(edges, key=lambda x: x[2]["loss"], reverse=True)
    level_2_edges = []
    for edge in edges:
        if edge[0] == node_id:
            level_2_edges.append(edge)
            edges.remove(edge)
        if edge[1] == node_id:
            level_2_edges.append((edge[1], edge[0], edge[2]))
            edges.remove(edge)
        if len(level_2_edges) >= top_extra_edges:
            break
    return level_2_edges


async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    graph_storage: NetworkXStorage,
    top_extra_edges: int = 3
) -> dict:
    """
    Traverse the graph

    :param llm_client: llm client
    :param graph_storage: graph storage instance
    :param top_extra_edges: top extra edges
    :return: question and answer
    """

    results = {}
    edges = list(await graph_storage.get_all_edges())
    while len(edges) > 0:
        # 按照loss从大到小排序
        edges = sorted(edges, key=lambda x: x[2]["loss"], reverse=True)

        _process_nodes = []
        _process_edges = []

        max_loss_edge = edges.pop(0)
        src_id = max_loss_edge[0]
        tgt_id = max_loss_edge[1]

        _process_nodes.append(await _get_node_info(src_id, graph_storage))
        _process_nodes.append(await _get_node_info(tgt_id, graph_storage))
        _process_edges.append(max_loss_edge)

        level_2_edges = _get_level_2_edges(edges, tgt_id, top_extra_edges)
        assert len(level_2_edges) <= top_extra_edges

        for edge in level_2_edges:
            _process_nodes.append(await _get_node_info(edge[1], graph_storage))
            _process_edges.append(edge)

        context = await _process_nodes_and_edges(
            _process_nodes,
            _process_edges,
            llm_client
        )

        question = await llm_client.generate_answer(
            QUESTION_GENERATION_PROMPT['TEMPLATE'].format(
                answer=context
            )
        )

        logger.info(f"Question: {question} Answer: {context}")

        results[compute_content_hash(context)] = {
            "question": question,
            "answer": context
        }

    return results
