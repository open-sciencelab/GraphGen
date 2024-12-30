# Rewrite from https://github.com/ZitongYang/Synthetic_Continued_Pretraining/blob/main/tasks/quality.py
import json
from hashlib import md5

from .task_abc import Document, Task
from baselines.EntiGraph.entigraph_utils.prompt_utils import (
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS,
                                QUALITY_FEW_SHOT_COT_PROMPT)

class BaselineTask(Task):
    openai_system_generate_entities = OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES
    openai_system_generate_two_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS
    openai_system_generate_three_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS
    llama_cot_prompt = QUALITY_FEW_SHOT_COT_PROMPT

    @staticmethod
    def _load_split():
        file_path = '../../resources/examples/chunked_demo.json'
        data = json.load(open(file_path, 'r'))

        documents = []
        for doc in data:
            for chunk in doc:
                documents.append(chunk)
        return documents

    def _create_documents(self):
        documents = []
        for adict in self._data:
            document = Document(text=adict['content'], questions=[])
            documents.append(document)
        super().__init__('baseline', documents)

    def _dedup(self):
        deuped_documents = {}
        for document in self.documents:
            key = compute_content_hash(document.text)
            if key not in deuped_documents:
                deuped_documents[key] = document

        self.documents = list(deuped_documents.values())


    def __init__(self):
        self._data = self._load_split()
        self._create_documents()
        self._dedup()

    def performance_stats(self):
        pass

    def load_attempts_json(self, file_path: str):
        pass


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
