from dataclasses import dataclass

@dataclass
class RewardEvaluator:
    def evaluate(self, text: str) -> float:
        pass