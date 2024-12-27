# https://arxiv.org/pdf/2304.08460
# https://github.com/akoksal/LongForm/tree/main
from dataclasses import dataclass
from models import OpenAIModel
from typing import List
from utils import create_event_loop, compute_content_hash
from tqdm.asyncio import tqdm as tqdm_async

PROMPT_TEMPLATE = '''Instruction: X
Output:{doc}

What kind of instruction could this be the answer to?
X:'''

@dataclass
class LongForm:
    llm_client: OpenAIModel = None

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> List[dict]:
        results = []
        for doc in tqdm_async(docs, desc="Generating using LongForm"):
            for chunk in doc:
                content = chunk['content']
                prompt = PROMPT_TEMPLATE.format(doc=content)
                try:
                    question = await self.llm_client.generate_answer(prompt)
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
    import os
    import json
    from dotenv import load_dotenv
    from models import OpenAIModel

    load_dotenv()

    llm_client = OpenAIModel(
        model_name=os.getenv("TEACHER_MODEL"),
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )

    longform = LongForm(llm_client=llm_client)

    with open("../../resources/examples/chunked_demo.json", "r") as f:
        data = json.load(f)

    results = longform.generate(data)

    # Save results
    with open("../../cache/data/longform.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
