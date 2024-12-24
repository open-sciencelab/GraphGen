from dataclasses import dataclass
from base_evaluator import BaseEvaluator
from utils import detect_main_language, NLTKHelper

nltk_helper = NLTKHelper()

@dataclass
class LengthEvaluator(BaseEvaluator):
    """
    衡量文本长度的指标
    """
    def evaluate(self, text: str) -> float:
        lang = "chinese" if detect_main_language(text) == "zh" else "english"
        tokens = nltk_helper.word_tokenize(text, lang)

        return len(tokens)
