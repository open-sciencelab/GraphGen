# https://arxiv.org/pdf/2401.14367
import os
import json
from dotenv import load_dotenv
import argparse

from dataclasses import dataclass
from models import OpenAIModel
from typing import List
from utils import create_event_loop, compute_content_hash
from tqdm.asyncio import tqdm as tqdm_async

PROMPT_TEMPLATE = '''Instruction: Given the next [document], create a [question] and [answer] pair that are grounded in the main point of the document, don't add any additional information that is not in the document. The [question] is by an information-seeking user and the [answer] is provided by a helping AI Agent.

[document]: Scrumptious Sweet Co. factory ...

### Response:
[question]: Where was the movie Chitty Chitty Bang Bang filmed?
[answer]: Chitty Chitty Bang Bang was filmed in a variety of locations in England and Germany. Some ...

[document]: The series stars Eugene Levy as Johnny Rose ...

### Response:
[question]: What is the plot of the show Schitt's Creek?
[answer]: The show Schitt's Creek is about a wealthy family who loses their fortune and is forced to rebuild their lives in a small town. The show follows the family as they adjust to their new life in the town and learn to appreciate the simple things in life.

[document]: 2016's countdown broke several Hottest 100 records ...

### Response:
[question]: What was the most popular song on the 2016 Hottest 100?
[answer]: The most popular song on the 2016 Hottest 100 was "Never Be Like You" by Flume. This was the first time that an electronic dance music producer topped the countdown.

[document]: In Greek mythology, Persephone ...

### Response:
[question]: Who is Persephone in Greek mythology and what is her significance?
[answer]: In Greek mythology, Persephone, also known as Kore, is the daughter of Zeus and Demeter. She ...

[document]: {doc}

### Response:'''


def _post_process(content: str) -> tuple:
    lines = content.split('\n')
    question = lines[0].split('[question]: ')[1]
    answer = lines[1].split('[answer]: ')[1]
    return question, answer


@dataclass
class Genie:
    llm_client: OpenAIModel = None

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> List[dict]:
        results = []
        for doc in tqdm_async(docs, desc="Generating using Genie"):
            for chunk in doc:
                content = chunk['content']
                prompt = PROMPT_TEMPLATE.format(doc=content)
                try:
                    result = await self.llm_client.generate_answer(prompt)
                    question, answer = _post_process(result)
                    results.append({
                        compute_content_hash(question): {
                            'question': question,
                            'answer': content
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
                        default='cache/data/genie.json',
                        type=str)

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIModel(
        model_name=os.getenv("TEACHER_MODEL"),
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )

    genie = Genie(llm_client=llm_client)

    if args.data_type == 'raw':
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
            data = [[chunk] for chunk in data]
    elif args.data_type == 'chunked':
        with open(args.input_file, "r") as f:
            data = json.load(f)

    results = genie.generate(data)

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
