import asyncio

from dataclasses import dataclass
from utils import create_event_loop
from tqdm.asyncio import tqdm as tqdm_async

@dataclass
class BaseEvaluator:
    def evaluate(self, texts: list[str]) -> float:
        """
        Evaluate the text and return a score.
        """
        return create_event_loop().run_until_complete(self.async_evaluate(texts))

    async def async_evaluate(self, texts: list[str]) -> float:
        results = []
        for result in tqdm_async(
            asyncio.as_completed([self.evaluate_single(text) for text in texts]),
            total=len(texts),
        ):
            results.append(await result)
        return results

    async def evaluate_single(self, text: str) -> float:
        raise NotImplementedError()

    def get_average_score(self, texts: list[str]) -> float:
        """
        Get the average score of a batch of texts.
        """
        return sum(self.evaluate(texts)) / len(texts)
