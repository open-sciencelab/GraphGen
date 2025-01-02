import torch
from dataclasses import dataclass
from .base_evaluator import BaseEvaluator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.text.text_pair import TextPair
from utils import logger, create_event_loop


@dataclass
class RewardEvaluator(BaseEvaluator):
    reward_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    current_device_id: int = 0

    def __post_init__(self):
        self.device_ids = range(torch.cuda.device_count())
        self.num_gpus = len(self.device_ids)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")

        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
        self.rank_models = []
        for device_id in self.device_ids:
            self.rank_models.append(AutoModelForSequenceClassification.from_pretrained(self.reward_name).cuda(device_id))
            self.rank_models[-1].eval()

        logger.info(f"Loaded reward model: {self.reward_name}")

    async def evaluate_single(self, pair: TextPair) -> float:
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._tokenize_and_rank, pair)
    
    def _tokenize_and_rank(self, pair: TextPair) -> float: 
        question, answer = pair.question, pair.answer

        current_device_id = self.current_device_id
        self.current_device_id = (self.current_device_id + 1) % self.num_gpus

        rank_model = self.rank_models[current_device_id]

        # concatenate the question and answer
        inputs = self.tokenizer(question, answer, return_tensors="pt")
        inputs = {k: v.to(rank_model.device) for k, v in inputs.items()}

        score = rank_model(**inputs).logits[0].item()
        return score
