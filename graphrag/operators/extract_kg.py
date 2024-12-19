import re
import asyncio

from typing import List
from collections import defaultdict
from models import Chunk, TopkTokenModel, OpenAIModel
from models.storage.base_storage import BaseGraphStorage
from templates import KG_EXTRACTION_PROMPT
from tqdm.asyncio import tqdm as tqdm_async
from utils import logger
from utils import (pack_history_conversations, split_string_by_multi_markers,
                   handle_single_entity_extraction, handle_single_relationship_extraction)
from .merge import merge_nodes, merge_edges


async def extract_kg(
        llm_client: TopkTokenModel,
        kg_instance: BaseGraphStorage,
        chunks: List[Chunk],
        language: str =  None,
        entity_types: List[str] = None
):
    """

    :param llm_client: teacher LLM model to extract entities and relationships
    :param kg_instance: knowledge graph storage instance
    :param chunks
    :param language
    :param entity_types
    :return:
    """

    examples = "\n".join(KG_EXTRACTION_PROMPT["EXAMPLES"])

    if entity_types:
        KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["entity_types"] = ",".join(entity_types)

    if language:
        KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["language"] = language

    # add example's format
    examples = examples.format(**KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"])

    async def _process_single_content(chunk: Chunk, example: str, max_loop: int = 3):
        chunk_id = chunk.id
        content = chunk.content
        hint_prompt = KG_EXTRACTION_PROMPT["TEMPLATE"].format(
            **KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"], examples=example, input_text=content
        )

        final_result = await llm_client.generate_answer(hint_prompt)
        logger.info('First result: %s', final_result)

        history = pack_history_conversations(hint_prompt, final_result)
        for loop_index in range(max_loop):
            # 是否结束循环
            if_loop_result = await llm_client.generate_answer(text=KG_EXTRACTION_PROMPT["IF_LOOP"], history=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

            glean_result = await llm_client.generate_answer(text=KG_EXTRACTION_PROMPT["CONTINUE"], history=history)
            logger.info(f"Loop {loop_index} glean: {glean_result}")

            history += pack_history_conversations(KG_EXTRACTION_PROMPT["CONTINUE"], glean_result)
            final_result += glean_result
            if loop_index == max_loop - 1:
                break

        records = split_string_by_multi_markers(
            final_result,
            [
            KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["record_delimiter"],
            KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["completion_delimiter"]],
        )

        nodes = defaultdict(list)
        edges = defaultdict(list)

        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1) # 提取括号内的内容
            record_attributes = split_string_by_multi_markers(
                record, [KG_EXTRACTION_PROMPT["EXAMPLES_FORMAT"]["tuple_delimiter"]]
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
            [merge_nodes(k, v, kg_instance, llm_client) for k, v in nodes.items()]
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
            [merge_edges(src_id, tgt_id, v, kg_instance, llm_client) for
             (src_id, tgt_id), v in edges.items()]
        ),
        total=len(edges),
        desc="Inserting relationships into storage",
        unit="relationship",
    ):
        relationships_data.append(await result)

    return kg_instance
