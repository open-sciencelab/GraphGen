import re
import asyncio

from typing import List
from collections import defaultdict
from models import Chunk, TopkTokenModel, OpenAIModel
from models.storage.base_storage import BaseGraphStorage
from templates import ENTITY_EXTRACTION_PROMPT
from tqdm.asyncio import tqdm as tqdm_async
from utils import logger
from utils import (pack_history_conversations, split_string_by_multi_markers,
                   handle_single_entity_extraction, handle_single_relationship_extraction)
from .merge import merge_nodes, merge_edges


async def extract_entities_and_relations(
        llm_client: TopkTokenModel,
        knowledge_graph_instance: BaseGraphStorage,
        chunks: List[Chunk],
        language: str =  None,
        entity_types: List[str] = None,
        example_number: int = 3,
):
    """

    :param llm_client: LLM model to chat with
    :param knowledge_graph_instance: knowledge graph storage instance
    :param chunks:
    :param language: prompt language
    :param entity_types: entity types to extract
    :param example_number: number of examples in prompt
    :return:
    """

    if example_number and example_number < len(ENTITY_EXTRACTION_PROMPT["EXAMPLES"]):
        examples = "\n".join(
            ENTITY_EXTRACTION_PROMPT["EXAMPLES"][: int(example_number)]
        )
    else:
        examples = "\n".join(ENTITY_EXTRACTION_PROMPT["EXAMPLES"])

    if entity_types:
        ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["entity_types"] = ",".join(entity_types)

    if language:
        ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["language"] = language

    # add example's format
    examples = examples.format(**ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"])

    async def _process_single_content(chunk: Chunk, example: str, max_loop: int = 3):
        """

        :param chunk: chunk to process
        :param example: examples to extract entities
        :param max_loop: max loop to extract entities
        :return:
        """
        chunk_id = chunk.id
        content = chunk.content
        hint_prompt = ENTITY_EXTRACTION_PROMPT["TEMPLATE"].format(
            **ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"], examples=example, input_text=content
        )

        final_result = await llm_client.generate_answer(hint_prompt)
        logger.info('First result: %s', final_result)

        history = pack_history_conversations(hint_prompt, final_result)
        for loop_index in range(max_loop):
            glean_result = await llm_client.generate_answer(text=ENTITY_EXTRACTION_PROMPT["CONTINUE"], history=history)
            logger.info(f"Loop {loop_index} glean: {glean_result}")

            history += pack_history_conversations(ENTITY_EXTRACTION_PROMPT["CONTINUE"], glean_result)
            final_result += glean_result
            if loop_index == max_loop - 1:
                break

            # 是否结束循环
            if_loop_result = await llm_client.generate_answer(text=ENTITY_EXTRACTION_PROMPT["IF_LOOP"], history=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [
            ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["record_delimiter"],
            ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["completion_delimiter"]],
        )

        nodes = defaultdict(list)
        edges = defaultdict(list)

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1) # 提取括号内的内容
            record_attributes = split_string_by_multi_markers(
                record, [ENTITY_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["tuple_delimiter"]]
            )

            entity = await handle_single_entity_extraction(record_attributes, chunk_id)
            if entity is not None:
                nodes[entity["entity_name"]].append(entity)
                continue
            relation = await handle_single_relationship_extraction(record_attributes, chunk_id)
            if relation is not None:
                edges[(relation["src_id"], relation["tgt_id"])].append(relation)
        return dict(nodes), dict(edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c, examples) for c in chunks]),
        total=len(chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    logger.info("Inserting entities into storage...")
    entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [merge_nodes(k, v, knowledge_graph_instance, llm_client) for k, v in nodes.items()]
        ),
        total=len(nodes),
        desc="Inserting entities into storage",
        unit="entity",
    ):
        entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    relationships_data = []

    for result in tqdm_async(
        asyncio.as_completed(
            [merge_edges(src_id, tgt_id, v, knowledge_graph_instance, llm_client) for
             (src_id, tgt_id), v in edges.items()]
        ),
        total=len(edges),
        desc="Inserting relationships into storage",
        unit="relationship",
    ):
        relationships_data.append(await result)

    return knowledge_graph_instance
