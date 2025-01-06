# https://arxiv.org/abs/2409.07431
# https://github.com/zitongyang/synthetic_continued_pretraining

import os
import json
import asyncio
import argparse
from hashlib import md5

from .inference.devapi import gptqa
from .tasks.baseline_task import BaselineTask
import random
from tqdm.asyncio import tqdm as tqdm_async


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


async def generate_entities(document_content: str,
                      system_message: str,
                      openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None
    while not can_read_entities:
        try:
            completion = await gptqa(prompt,
                               openai_model,
                               system_message,
                               json_format=False)
            completion = completion[completion.find("{"): completion.rfind("}") + 1]
            response = json.loads(completion)
            can_read_entities = response['entities']
            return response
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}")

async def generate_two_entity_relations(document_content: str,
                                  entity1: str,
                                  entity2: str,
                                  system_message: str,
                                  openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = await gptqa(prompt,
                       openai_model,
                       system_message)
    return completion

async def generate_three_entity_relations(document_content: str,
                                    entity1: str,
                                    entity2: str,
                                    entity3: str,
                                    system_message: str,
                                    openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = await gptqa(prompt,
                       openai_model,
                       system_message)
    return completion

def _post_process_synthetic_data(data):
    block = data.split("\n\n")
    qas = {}
    for line in block:
        if "Question: " in line and "Answer: " in line:
            question = line.split("Question: ")[1].split("Answer: ")[0]
            answer = line.split("Answer: ")[1]
            qas[compute_content_hash(question)] = {
                "question": question,
                "answer": answer
            }
    return qas


async def generate_synthetic_data_for_document(input_file, data_type):
    random.seed(42)

    model_name = os.getenv("TEACHER_MODEL")

    task = BaselineTask(input_file, data_type)

    async def process_single_document(doc):
        output = [[]]

        entities = await generate_entities(
            doc.text,
            task.openai_system_generate_entities,
            model_name)
        if not entities:
            return []
        output[0] = entities['entities']
        output.append(entities['summary'])
        entities = entities['entities']

        pair_list = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair = (entities[i], entities[j])
                pair_list.append(pair)

        for response in tqdm_async(
            asyncio.as_completed([generate_two_entity_relations(doc.text, entity1, entity2, task.openai_system_generate_two_entity_relations, model_name) for entity1, entity2 in pair_list]),
            total=len(pair_list),
            desc="Generating synthetic data"
        ):
            try:
                response = await response
                if response:
                    output.append(response)
            except Exception as e:
                print(f"Error: {e}")

        # # iterate over triples of entities and generate relations
        # triple_list = []
        # for i in range(len(entities)):
        #     for j in range(i + 1, len(entities)):
        #         for k in range(j + 1, len(entities)):
        #             triple = (entities[i], entities[j], entities[k])
        #             triple_list.append(triple)
        # random.shuffle(triple_list)
        # for entity1, entity2, entity3 in tqdm(triple_list):
        #     response = await generate_three_entity_relations(
        #         doc.text, entity1, entity2, entity3,
        #         task.openai_system_generate_three_entity_relations,
        #         model_name)
        #     if response:
        #         output.append(response)

        corpus = output[1:]
        return corpus

    results = []
    for result in tqdm_async(
            asyncio.as_completed([process_single_document(doc) for doc in task.documents]),
            total=len(task.documents),
            desc="Generating synthetic data"
    ):
        results.extend(await result)


    async def generate_qa_sft(content):
        completion = await gptqa(content, model_name, task.openai_system_quality_qa_sft)
        return completion

    qa_sft_results = {}
    tasks = []
    for corpus in results:
        tasks.append(generate_qa_sft(corpus))

    for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating QA SFT"
    ):
        try:
            result = await result
            qa_sft_results.update(_post_process_synthetic_data(result))
        except Exception as e:
            print(f"Error: {e}")

    return qa_sft_results



if __name__ == '__main__':
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
                        default='cache/data/entigraph.json',
                        type=str)

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(generate_synthetic_data_for_document(args.input_file, args.data_type))

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
