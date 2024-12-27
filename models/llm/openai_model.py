import math
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError, ChatCompletion
from models import TopkTokenModel, Token
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

def get_top_response_tokens(response: ChatCompletion) -> List[Token]:
    token_logprobs = response.choices[0].logprobs.content
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob.logprob)
        candidate_tokens = [
            Token(t.token, math.exp(t.logprob))
            for t in token_prob.top_logprobs
        ]
        token = Token(token_prob.token, prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens

@dataclass
class OpenAIModel(TopkTokenModel):
    model_name: str = "gpt-4o-mini"
    api_key: str = None
    base_url: str = None

    system_prompt: str = ""
    json_mode: bool = False
    seed: int = None

    def __post_init__(self):
        assert self.api_key is not None, "Please provide api key to access openai api."
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _pre_generate(self, text: str, history: List[str]) -> Dict:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            assert len(history) % 2 == 0, "History should have even number of elements."
            messages = history + messages

        kwargs['messages']= messages
        return kwargs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    async def generate_topk_per_token(self, text: str, history: Optional[List[str]] = None) -> List[Token]:
        kwargs = self._pre_generate(text, history)
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            **kwargs
        )

        tokens = get_top_response_tokens(completion)

        return tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    async def generate_answer(self, text: str, history: Optional[List[str]] = None) -> str:
        kwargs = self._pre_generate(text, history)

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            **kwargs
        )

        return completion.choices[0].message.content
