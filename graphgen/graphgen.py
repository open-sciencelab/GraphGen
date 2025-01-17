# https://github.com/HKUDS/LightRAG

import os
import asyncio
import time
from typing import List, cast, Union
from dataclasses import dataclass
from tqdm.asyncio import tqdm as tqdm_async

from models import Chunk, JsonKVStorage, OpenAIModel, NetworkXStorage, WikiSearch, Tokenizer, TraverseStrategy
from models.storage.base_storage import StorageNameSpace
from utils import create_event_loop, logger, compute_content_hash
from .operators import extract_kg, search_wikipedia, quiz_relations, judge_relations, traverse_graph_by_edge


sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class GraphGen:
    unique_id: int = int(time.time())
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
    rephrase_storage: JsonKVStorage = JsonKVStorage(
        working_dir, namespace="rephrase"
    )
    qa_storage: JsonKVStorage = JsonKVStorage(
        os.path.join(working_dir, "data", "graphgen"), namespace=f"qa-{unique_id}"
    )

    # text chunking
    chunk_size: int = 1024
    chunk_overlap_size: int = 100

    # llm
    teacher_llm_client: OpenAIModel = None
    student_llm_client: OpenAIModel = None
    tokenizer_instance: Tokenizer = None

    # web search
    if_web_search: bool = False
    wiki_client: WikiSearch = WikiSearch()

    # traverse strategy
    traverse_strategy: TraverseStrategy = TraverseStrategy()

    async def async_split_chunks(self, data: Union[List[list], List[dict]], data_type: str) -> dict:
        # TODO： 是否进行指代消解
        if len(data) == 0:
            return {}

        new_docs = {}
        inserting_chunks = {}
        if data_type == "raw":
            assert isinstance(data, list) and isinstance(data[0], dict)
            # compute hash for each document
            new_docs = {
                compute_content_hash(doc['content'], prefix="doc-"): {'content': doc['content']} for doc in data
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")
            for doc_key, doc in tqdm_async(
                    new_docs.items(), desc="Chunking documents", unit="doc"
                ):
                chunks = {
                    compute_content_hash(dp["content"], prefix="chunk-"): {
                        **dp,
                        'full_doc_id': doc_key
                    } for dp in self.tokenizer_instance.chunk_by_token_size(doc["content"],
                                                                            self.chunk_overlap_size, self.chunk_size)
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
        elif data_type == "chunked":
            assert isinstance(data, list) and isinstance(data[0], list)
            new_docs = {
                compute_content_hash("".join(chunk['content']), prefix="doc-"): {'content': "".join(chunk['content'])}
                for doc in data for chunk in doc
            }
            _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if len(new_docs) == 0:
                logger.warning("All docs are already in the storage")
                return {}
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")
            for doc in tqdm_async(data, desc="Chunking documents", unit="doc"):
                doc_str = "".join([chunk['content'] for chunk in doc])
                for chunk in doc:
                    chunk_key = compute_content_hash(chunk['content'], prefix="chunk-")
                    inserting_chunks[chunk_key] = {
                        **chunk,
                        'full_doc_id': compute_content_hash(doc_str, prefix="doc-")
                    }
            _add_chunk_keys = await self.text_chunks_storage.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}

        await self.full_docs_storage.upsert(new_docs)
        await self.text_chunks_storage.upsert(inserting_chunks)

        return inserting_chunks

    def insert(self, data: Union[List[list], List[dict]], data_type: str):
        loop = create_event_loop()
        loop.run_until_complete(self.async_insert(data, data_type))

    async def async_insert(self, data: Union[List[list], List[dict]], data_type: str):
        """

        insert chunks into the graph
        """

        inserting_chunks = await self.async_split_chunks(data, data_type)

        if not len(inserting_chunks):
            logger.warning("All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        logger.info("[Entity and Relation Extraction]...")
        _add_entities_and_relations = await extract_kg(
            llm_client=self.teacher_llm_client,
            kg_instance=self.graph_storage,
            tokenizer_instance=self.tokenizer_instance,
            chunks=[Chunk(id=k, content=v['content']) for k, v in inserting_chunks.items()]
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted")
            return

        logger.info(f"[Wiki Search] is {'enabled' if self.if_web_search else 'disabled'}")
        if self.if_web_search:
            logger.info("[Wiki Search]...")
            _add_wiki_data = await search_wikipedia(
                llm_client= self.teacher_llm_client,
                wiki_search_client=self.wiki_client,
                knowledge_graph_instance=_add_entities_and_relations
            )
            await self.wiki_storage.upsert(_add_wiki_data)

        await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_instance in [self.full_docs_storage, self.text_chunks_storage,
                                 self.graph_storage, self.wiki_storage]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    def quiz(self, max_samples=1):
        loop = create_event_loop()
        loop.run_until_complete(self.async_quiz(max_samples))

    async def async_quiz(self, max_samples=1):
        await quiz_relations(self.teacher_llm_client, self.graph_storage, self.rephrase_storage, max_samples)
        await self.rephrase_storage.index_done_callback()

    def judge(self, re_judge=False):
        loop = create_event_loop()
        loop.run_until_complete(self.async_judge(re_judge))

    async def async_judge(self, re_judge=False):
        _update_relations = await judge_relations(self.student_llm_client, self.graph_storage,
                                                  self.rephrase_storage, re_judge)
        await _update_relations.index_done_callback()

    def traverse(self):
        loop = create_event_loop()
        loop.run_until_complete(self.async_traverse())

    async def async_traverse(self):
        results = await traverse_graph_by_edge(self.teacher_llm_client, self.tokenizer_instance,
                                               self.graph_storage, self.traverse_strategy)
        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()
