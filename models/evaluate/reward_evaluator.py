from dataclasses import dataclass
from .base_evaluator import BaseEvaluator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.text.text_pair import TextPair
from utils import logger, create_event_loop


@dataclass
class RewardEvaluator(BaseEvaluator):
    
    # TODO: list the available reward models, like "BAAI/IndustryCorpus2_Datarater"
    reward_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    def __post_init__(self):
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.reward_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)

        self.rank_model.eval()
        self.rank_model.to("cuda")

        logger.info(f"Loaded reward model: {self.reward_name}")

    async def evaluate_single(self, pair: TextPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._tokenize_and_rank, pair)
    
    def _tokenize_and_rank(self, pair: TextPair) -> float: 
        question, answer = pair.question, pair.answer

        # concatenate the question and answer
        inputs = self.tokenizer(question, answer, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        score = self.rank_model(**inputs).logits[0].item()
        return score
    