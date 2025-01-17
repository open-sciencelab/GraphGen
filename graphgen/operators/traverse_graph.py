import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from models import OpenAIModel, NetworkXStorage, TraverseStrategy, Tokenizer
from templates import ANSWER_REPHRASING_PROMPT, QUESTION_GENERATION_PROMPT
from utils import detect_main_language, compute_content_hash, logger
from graphgen.operators.split_graph import get_batches_with_strategy


async def _pre_tokenize(graph_storage: NetworkXStorage,
                        tokenizer: Tokenizer,
                        edges: list,
                        nodes: list) -> tuple:

    sem = asyncio.Semaphore(1000)
    async def handle_edge(edge: tuple) -> tuple:
        async with sem:
            if 'length' not in edge[2]:
                edge[2]['length'] = len(
                    await asyncio.get_event_loop().run_in_executor(None,
                                                                   tokenizer.encode_string,
                                                                   edge[2]['description']))
            return edge

    async def handle_node(node: dict) -> dict:
        async with sem:
            if 'length' not in node[1]:
                node[1]['length'] = len(
                    await asyncio.get_event_loop().run_in_executor(None,
                                                                   tokenizer.encode_string,
                                                                   node[1]['description']))
            return node

    new_edges = []
    new_nodes = []

    for result in tqdm_async(asyncio.as_completed([handle_edge(edge) for edge in edges]),
                             total=len(edges), desc="Pre-tokenizing edges"):
        new_edge = await result
        await graph_storage.update_edge(new_edge[0], new_edge[1], new_edge[2])
        new_edges.append(new_edge)

    for result in tqdm_async(asyncio.as_completed([handle_node(node) for node in nodes]),
                             total=len(nodes), desc="Pre-tokenizing nodes"):
        new_node = await result
        await graph_storage.update_node(new_node[0], new_node[1])
        new_nodes.append(new_node)

    await graph_storage.index_done_callback()
    return new_edges, new_nodes


def get_loss_tercile(losses: list) -> (float, float):
    losses = sorted(losses)
    q1_index = int(len(losses) * (1 / 3))
    q2_index = int(len(losses) * (2 / 3))

    return losses[q1_index], losses[q2_index]

def get_average_loss(batch: tuple) -> float:
    return sum(edge[2]['loss'] for edge in batch[1]) + sum(node['loss'] for node in batch[0]) / \
           (len(batch[0]) + len(batch[1]))

async def traverse_graph_by_edge(
    llm_client: OpenAIModel,
    tokenizer: Tokenizer,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
    max_concurrent: int = 1000
) -> dict:
    """
    Traverse the graph

    :param llm_client
    :param tokenizer
    :param graph_storage
    :param traverse_strategy
    :param max_concurrent
    :return: question and answer
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_nodes_and_edges(
            _process_nodes: list,
            _process_edges: list,
            _difficulty: str
    ) -> str:
        entities = [
            f"{_process_node['node_id']}: {_process_node['description']}" for _process_node in _process_nodes
        ]
        relations = [
            f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
            for _process_edge in _process_edges
        ]

        entities_str = "\n".join([f"{index + 1}. {entity}" for index, entity in enumerate(entities)])
        relations_str = "\n".join([f"{index + 1}. {relation}" for index, relation in enumerate(relations)])

        language = "Chinese" if detect_main_language(entities_str + relations_str) == "zh" else "English"
        prompt = ANSWER_REPHRASING_PROMPT[_difficulty][language]['TEMPLATE'].format(
            language=language,
            entities=entities_str,
            relationships=relations_str
        )

        context = await llm_client.generate_answer(prompt)

        # post-process the context
        if context.startswith("Rephrased Text:"):
            context = context[len("Rephrased Text:"):].strip()
        elif context.startswith("重述文本:"):
            context = context[len("重述文本:"):].strip()

        return context

    async def _process_single_batch(
        _process_batch: tuple
    ) -> dict:
        async with semaphore:
            context = await _process_nodes_and_edges(
                _process_batch[0],
                _process_batch[1],
                _process_batch[2]
            )

            language = "Chinese" if detect_main_language(context) == "zh" else "English"
            question = await llm_client.generate_answer(
                QUESTION_GENERATION_PROMPT[language]['TEMPLATE'].format(
                    answer=context
                )
            )

            if question.startswith("Question:"):
                question = question[len("Question:"):].strip()
            elif question.startswith("问题："):
                question = question[len("问题："):].strip()

            pre_length = sum(node['length'] for node in _process_batch[0]) \
                         + sum(edge[2]['length'] for edge in _process_batch[1])

            logger.info("%d nodes and %d edges processed", len(_process_batch[0]), len(_process_batch[1]))
            logger.info("Pre-length: %s", pre_length)
            logger.info("Question: %s Answer: %s", question, context)

            return {
                compute_content_hash(context): {
                    "question": question,
                    "answer": context,
                    "loss": get_average_loss(_process_batch),
                    "difficulty": _process_batch[2],
                }
            }

    results = {}
    edges = list(await graph_storage.get_all_edges())
    nodes = list(await graph_storage.get_all_nodes())

    edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)

    processing_batches = await get_batches_with_strategy(
        nodes,
        edges,
        graph_storage,
        traverse_strategy
    )

    losses = []
    for batch in processing_batches:
        loss = get_average_loss(batch)
        losses.append(loss)
    q1, q2 = get_loss_tercile(losses)

    difficulty_order = traverse_strategy.difficulty_order
    for i, batch in enumerate(processing_batches):
        loss = get_average_loss(batch)
        if loss < q1:
            # easy
            processing_batches[i] = (batch[0], batch[1], difficulty_order[0])
        elif loss < q2:
            # medium
            processing_batches[i] = (batch[0], batch[1], difficulty_order[1])
        else:
            # hard
            processing_batches[i] = (batch[0], batch[1], difficulty_order[2])

    for result in tqdm_async(asyncio.as_completed(
        [_process_single_batch(batch) for batch in processing_batches]
    ), total=len(processing_batches), desc="Processing batches"):
        try:
            results.update(await result)
        except Exception as e: # pylint: disable=broad-except
            logger.error("Error occurred while processing batches: %s", e)

    return results
