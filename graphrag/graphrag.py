import os
import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from .operators import *
from models import Chunk, JsonKVStorage, OpenAIModel, NetworkXStorage, WikiSearch
from typing import List, cast

from dataclasses import dataclass
from utils import create_event_loop, logger, set_logger, compute_content_hash, chunk_by_token_size
from models.storage.base_storage import StorageNameSpace
from dotenv import load_dotenv

load_dotenv()

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
set_logger(os.path.join(sys_path, "cache", "graphrag.log"))

@dataclass
class GraphRag:
    working_dir: str = os.path.join(sys_path, "cache")
    full_docs_storage: JsonKVStorage = JsonKVStorage(
        working_dir, namespace="full_docs"
    )
    text_chunks_storage: JsonKVStorage = JsonKVStorage(
        working_dir, namespace="text_chunks"
    )
    wiki_storage: JsonKVStorage = JsonKVStorage(
        working_dir, namespace="wiki"
    )
    graph_storage: NetworkXStorage = NetworkXStorage(
        working_dir, namespace="graph"
    )

    # text chunking
    chunk_size: int = 1200
    chunk_overlap_size: int = 100

    # llm
    llm_client: OpenAIModel = OpenAIModel(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    # web search
    wiki_client: WikiSearch = WikiSearch()

    def insert(self, chunks: List[Chunk]):
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert(chunks))

    async def async_insert(self, chunks: List[Chunk]):
        """
        insert chunks into the graph
        """

        # compute hash for each chunk
        new_docs = {
            compute_content_hash(chunk.content, prefix="doc-"): {'content': chunk.content} for chunk in chunks
        }
        _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            logger.warning("All docs are already in the storage")
            return

        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        inserting_chunks = {}
        for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
            chunks = {
                compute_content_hash(dp["content"], prefix="chunk-"): {
                    **dp,
                    'full_doc_id': doc_key
                } for dp in chunk_by_token_size(doc["content"], self.chunk_overlap_size, self.chunk_size)
            }
            inserting_chunks.update(chunks)
        _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
        inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        if not len(inserting_chunks):
            logger.warning("All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.llm_client,
            kg_instance=self.graph_storage,
            chunks=[Chunk(id=k, content=v['content']) for k, v in inserting_chunks.items()],
            language="Chinese"
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        logger.info(f"[Wiki Search]...")
        _add_wiki_data = await search_wikipedia(
            llm_client= self.llm_client,
            wiki_search_client=self.wiki_client,
            knowledge_graph_instance=_add_entities_and_relations
        )

        # 将文档、块、wiki搜索结果分别存入各自的数据库中
        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)
        await self.wiki_storage.upsert(_add_wiki_data)

        await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_instance in [self.full_docs_storage, self.text_chunks_storage, self.graph_storage, self.wiki_storage]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    def judge(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_judge())

    async def async_judge(self):
        _update_relations = await judge_relations(self.llm_client, self.graph_storage)
        await _update_relations.index_done_callback()

    def traverse(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_traverse())

    async def async_traverse(self):
        await traverse_graph_by_edge(self.llm_client, self.graph_storage)
