import asyncio

from dataclasses import dataclass
from utils import create_event_loop
from tqdm.asyncio import tqdm as tqdm_async
from models.text.text_pair import TextPair

@dataclass
class BaseEvaluator:
    max_concurrent: int = 1000

    def evaluate(self, pairs: list[TextPair]) -> float:
        """
        Evaluate the text and return a score.
        """
        return create_event_loop().run_until_complete(self.async_evaluate(pairs))

    async def async_evaluate(self, pairs: list[TextPair]) -> float:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_with_semaphore(pair):
            async with semaphore:  # 获取Semaphore
                return await self.evaluate_single(pair)
        
        results = []
        for result in tqdm_async(
            asyncio.as_completed([evaluate_with_semaphore(pair) for pair in pairs]),
            total=len(pairs),
        ):
            results.append(await result)
        return results

    async def evaluate_single(self, pair: TextPair) -> float:
        raise NotImplementedError()

    def get_average_score(self, pairs: list[TextPair]) -> float:
        """
        Get the average score of a batch of texts.
        """
        return sum(self.evaluate(pairs)) / len(pairs)
