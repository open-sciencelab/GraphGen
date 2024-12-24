from collections import Counter
from utils.format import split_string_by_multi_markers
from models import TopkTokenModel, Tokenizer
from models.storage.base_storage import BaseGraphStorage
from templates import ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT
from utils.log import logger

async def _handle_kg_summary(
    entity_or_relation_name: str,
    description: str,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
    max_summary_tokens: int = 200
) -> str:
    """
    处理实体或关系的描述信息

    :param entity_or_relation_name
    :param description
    :param llm_client
    :param tokenizer_instance
    :param max_summary_tokens
    :return: new description
    """
    tokens = tokenizer_instance.encode_string(description)
    if len(tokens) <  max_summary_tokens:
        return description

    use_description = tokenizer_instance.decode_tokens(tokens[:max_summary_tokens])
    prompt = ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT["TEMPLATE"].format(
        entity_name=entity_or_relation_name,
        description_list=use_description.split('<SEP>'),
        **ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT["EXAMPLES_FORMAT"]
    )
    new_description = await llm_client.generate_answer(prompt)
    logger.info(f"Entity or relation {entity_or_relation_name} summary: {new_description}")
    return new_description


async def merge_nodes(
    entity_name: str,
    nodes_data: list[dict],
    kg_instance: BaseGraphStorage,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
):
    """
    Merge nodes

    :param entity_name
    :param nodes_data
    :param kg_instance
    :param llm_client
    :param tokenizer_instance
    :return: None
    """
    entity_types = []
    source_ids = []
    descriptions = []

    node = await kg_instance.get_node(entity_name)
    if node is not None:
        entity_types.append(node["entity_type"])
        source_ids.extend(
            split_string_by_multi_markers(node["source_id"], ['<SEP>'])
        )
        descriptions.append(node["description"])

    # 统计当前节点数据和已有节点数据的entity_type出现次数，取出现次数最多的entity_type
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = '<SEP>'.join(
        sorted(set([dp["description"] for dp in nodes_data] + descriptions))
    )
    description = await _handle_kg_summary(
        entity_name, description, llm_client, tokenizer_instance
    )

    source_id = '<SEP>'.join(
        set([dp["source_id"] for dp in nodes_data] + source_ids)
    )

    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id
    )
    await kg_instance.upsert_node(
        entity_name,
        node_data=node_data
    )
    node_data["entity_name"] = entity_name
    return node_data

async def merge_edges(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_instance: BaseGraphStorage,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
):
    """
    Merge edges

    :param src_id
    :param tgt_id
    :param edges_data
    :param knowledge_graph_instance
    :param llm_client
    :param tokenizer_instance
    :return: None
    """

    source_ids = []
    descriptions = []

    edge = await knowledge_graph_instance.get_edge(src_id, tgt_id)
    if edge is not None:
        source_ids.extend(
            split_string_by_multi_markers(edge["source_id"], ['<SEP>'])
        )
        descriptions.append(edge["description"])

    description = '<SEP>'.join(
        sorted(set([dp["description"] for dp in edges_data] + descriptions))
    )
    source_id = '<SEP>'.join(
        set([dp["source_id"] for dp in edges_data] + source_ids)
    )

    for insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_instance.has_node(insert_id)):
            await knowledge_graph_instance.upsert_node(
                insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKNOWN"
                }
            )

    description = await _handle_kg_summary(
        f"({src_id}, {tgt_id})", description, llm_client, tokenizer_instance
    )

    await knowledge_graph_instance.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            description=description,
            source_id=source_id
        )
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
    )
    return edge_data
