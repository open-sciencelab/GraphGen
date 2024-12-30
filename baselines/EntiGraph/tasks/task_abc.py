from typing import List, Dict
from abc import abstractmethod


class Question:
    def __init__(self, statement: Dict, answer: str, attempts: List[Dict], formatted_prompt: str = ""):
        self.statement = statement
        self.answer = answer
        self.attempts = attempts
        self.formatted_prompt = formatted_prompt

    @abstractmethod
    def prompt(self):
        pass

    @abstractmethod
    def iscorrect(self, attempt_index: int = 0):
        pass

    @abstractmethod
    def asdict(self):
        pass

    @abstractmethod
    def llama_parse_answer(self):
        pass


class Document:
    def __init__(self, text: str, questions: List[Dict]):
        self.text = text
        self.questions = questions

    @property
    @abstractmethod
    def uid(self):
        pass

    @property
    @abstractmethod
    def content(self):
        pass

    @abstractmethod
    def question_prompts(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        pass

    @abstractmethod
    def asdict(self):
        pass

    def majority_vote(self, n_samples):
        for question in self.questions:
            question.majority_vote(n_samples)


class Task:
    openai_system_generate_entities: str
    openai_system_generate_two_entity_relations: str
    openai_system_generate_three_entity_relations: str
    llama_cot_prompt: str

    def __init__(self, name, documents: List[Document]):
        self.name = name
        self.documents = documents

    @abstractmethod
    def load_attempts_json(self, file_path: str):
        pass

    @abstractmethod
    def performance_stats(self):
        pass

    def all_questions(self, add_document_context: bool, add_thought_process: bool, sep_after_question: str):
        prompts = []
        for document in self.documents:
            prompts += document.question_prompts(add_document_context, add_thought_process, sep_after_question)

        return prompts

    @property
    def all_document_contents(self):
        return '\n'.join([document.content for document in self.documents])

    def asdict(self):
        return [document.asdict() for document in self.documents]

    def majority_vote(self, n_samples: int = 1):
        for document in self.documents:
            document.majority_vote(n_samples)
