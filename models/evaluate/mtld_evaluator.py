from dataclasses import  dataclass
from .base_evaluator import BaseEvaluator

@dataclass
class MLTDEvaluator(BaseEvaluator):
    """

    """
    def evaluate(self, text: str) -> float:
