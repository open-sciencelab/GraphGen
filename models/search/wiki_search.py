import wikipedia
from wikipedia import set_lang

from typing import List
from utils import detect_main_language
from dataclasses import dataclass


@dataclass
class WikiSearch:
    @staticmethod
    def set_language(language: str):
        assert language in ["en", "zh"], "Only support English and Chinese"
        set_lang(language)

    def search(self, query: str) -> List[str]:
        self.set_language(detect_main_language(query))
        return wikipedia.search(query)

    def summary(self, query: str) -> str:
        self.set_language(detect_main_language(query))
        return wikipedia.summary(query)

    def page(self, query: str) -> str:
        self.set_language(detect_main_language(query))
        return wikipedia.page(query).content
