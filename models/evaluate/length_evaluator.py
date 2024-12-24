from dataclasses import dataclass
from .base_evaluator import BaseEvaluator
from models.llm.tokenizer import Tokenizer
from models.text.text_pair import TextPair



@dataclass
class LengthEvaluator(BaseEvaluator):
    tokenizer_name: str = "cl100k_base"
    def __post_init__(self):
        self.tokenizer = Tokenizer(
            model_name=self.tokenizer_name
        )

    async def evaluate_single(self, pair: TextPair) -> float:
        return len(self.tokenizer.encode_string(pair.answer))
