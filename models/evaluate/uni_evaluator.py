# https://github.com/maszhongming/UniEval/tree/main
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from .base_evaluator import BaseEvaluator
from dataclasses import dataclass, field
from utils import create_event_loop
from models.text.text_pair import TextPair
import asyncio
from tqdm.asyncio import tqdm as tqdm_async


@dataclass
class UniEvaluator(BaseEvaluator):
    model_name: str = "MingZhong/unieval-sum"
    dimensions: list = field(default_factory=lambda: ['naturalness', 'coherence', 'understandability'])

    def __post_init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model.eval()
        self.model.to("cuda")

        self.softmax = nn.Softmax(dim=1)

        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]
    
    def evaluate(self, pairs: list[TextPair], dimension: str) -> float:
        """
        Evaluate the text and return a score.
        """
        return create_event_loop().run_until_complete(self.async_evaluate(pairs, dimension))

    async def async_evaluate(self, pairs: list[TextPair], dimension: str) -> float:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_with_semaphore(pair):
            async with semaphore:  # 获取Semaphore
                return await self.evaluate_single(pair, dimension)
        
        results = []
        for result in tqdm_async(
            asyncio.as_completed([evaluate_with_semaphore(pair) for pair in pairs]),
            total=len(pairs),
        ):
            results.append(await result)
        return results

    async def evaluate_single(self, pair: TextPair, dimension: str) -> float:
        text = self._add_questions(dimension, pair.question, pair.answer)
        loop = create_event_loop()
        return await loop.run_in_executor(None, self._score, text)


    def get_average_score(self, pairs: list[TextPair], dimension: str) -> float:
        """
        Get the average score of a batch of texts.
        """
        return sum(self.evaluate(pairs, dimension)) / len(pairs)

    def _score(self, text: str) -> float:
        """
            Get scores for the given samples.
            final_score = postive_score / (postive_score + negative_score)
        """

        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # Therefore, we construct a random one-word target sequence.
        # The content of the target has no effect on the final scores.

        tgt = "No"

        with torch.no_grad():
            encoded_src = self.tokenizer(
                text,
                # max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            encoded_tgt = self.tokenizer(
                tgt,
                # max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            src_tokens = encoded_src['input_ids'].to("cuda")
            src_mask = encoded_src['attention_mask'].to("cuda")

            tgt_tokens = encoded_tgt['input_ids'].to("cuda")[:, 0].unsqueeze(-1)

            output = self.model(
                input_ids=src_tokens,
                attention_mask=src_mask,
                labels=tgt_tokens
            )

            logits = output.logits.view(-1, self.model.config.vocab_size)

            pos_score = self.softmax(logits)[:, self.pos_id]  # Yes
            neg_score = self.softmax(logits)[:, self.neg_id]

            score = pos_score / (pos_score + neg_score)

        return score.item()
    
    def _add_questions(self, dimension: str, question: str, answer: str):
        if dimension == "naturalness":
            cur_input = 'question: Is this a natural response in the dialogue? </s> response: ' + answer
        elif dimension == "coherence":
            cur_input = 'question: Is this a coherent response given the dialogue history? </s> response: '\
                            + answer + ' </s> dialogue history: ' + question
        elif dimension == "understandability":
            cur_input = 'question: Is this an understandable response in the dialogue? </s> response: ' + answer
        else:
            raise NotImplementedError('The input format for this dimension is still undefined. Please customize it first.')
        return cur_input
