from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.text.text_pair import TextPair


@dataclass
class RewardEvaluator:
    # TODO: list the available reward models, like "BAAI/IndustryCorpus2_Datarater"
    reward_name: str = "OpenAssistant/reward-model-deberta-v3-large"

    def __post_init__(self):
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.reward_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)

    async def evaluate_single(self, pair: TextPair) -> float:
        question, answer = pair.question, pair.answer
        # concatenate the question and answer
        inputs = self.tokenizer(question, answer, return_tensors="pt")
        score = self.rank_model(**inputs).logits[0].item()
        return score
