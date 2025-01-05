# https://arxiv.org/abs/2401.16380

import os
import json
import argparse
from dotenv import load_dotenv
from models import OpenAIModel

from dataclasses import dataclass
from typing import List
from utils import create_event_loop, compute_content_hash
from tqdm.asyncio import tqdm as tqdm_async


PROMPT_TEMPLATE = '''A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:":{doc}.

Examples and format:
---
Question: What was the stock’s closing price on Friday? Answer: $21.51.
---
Question: How much did the stock rise on Friday? Answer: $2.11 or about 11 percent.
---
Question: What was the revenue drop in the first quarter compared to the same period last year? Answer: The revenue dropped 15 percent.
---
'''

def _post_process(content: str) -> list:
    print(content)
    raw_qas = content.split('---')
    qas = []
    for item in raw_qas:
        try:
            if "Question:" in item and "Answer:" in item:
                question = item.split('Question:')[1].split('Answer:')[0].strip()
                answer = item.split('Answer:')[1].strip()
                qas.append((question, answer))
        except Exception as e:
            print(f"Error: {e}")
            continue
    return qas


@dataclass
class Wrap:
    llm_client: OpenAIModel = None

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> List[dict]:
        results = []
        for doc in tqdm_async(docs, desc="Generating using Wrap"):
            for chunk in doc:
                content = chunk['content']
                prompt = PROMPT_TEMPLATE.format(doc=content)
                try:
                    result = await self.llm_client.generate_answer(prompt)
                    qas = _post_process(result)
                    for qa in qas:
                        results.append({
                            compute_content_hash(qa[0]): {
                                'question': qa[0],
                                'answer': qa[1]
                            }
                        })
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Raw context jsonl path.',
                        default='resources/examples/chunked_demo.json',
                        type=str)
    parser.add_argument('--data_type',
                        help='Data type of input file. (Raw context or chunked context)',
                        choices=['raw', 'chunked'],
                        default='raw',
                        type=str)
    parser.add_argument('--output_file',
                        help='Output file path.',
                        default='cache/data/wrap.json',
                        type=str)

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIModel(
        model_name=os.getenv("TEACHER_MODEL"),
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )

    wrap = Wrap(llm_client=llm_client)

    if args.data_type == 'raw':
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
            data = [[chunk] for chunk in data]
    elif args.data_type == 'chunked':
        with open(args.input_file, "r") as f:
            data = json.load(f)

    results = wrap.generate(data)

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
