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

    max_concurrent = 1000
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_document_entities(doc):
        async with semaphore:
            entities = await generate_entities(
                doc.text,
                task.openai_system_generate_entities,
                model_name)
            if not entities:
                return None
            return {
                'document': doc.text,
                'entities': entities['entities'],
                'summary': entities['summary']
            }

    entities_list = []
    for result in tqdm_async(
            asyncio.as_completed([generate_document_entities(doc) for doc in task.documents]),
            total=len(task.documents),
            desc="Generating entities"
    ):
        result = await result
        if result:
            entities_list.append(result)

    # iterate over triples of entities and generate relations
    pair_list = []
    for doc in entities_list:
        entities = doc['entities']
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair = (doc['document'], entities[i], entities[j])
                pair_list.append(pair)

    async def process_two_entity_relations(pair):
        async with semaphore:
            document, entity1, entity2 = pair
            response = await generate_two_entity_relations(
                document, entity1, entity2,
                task.openai_system_generate_two_entity_relations,
                model_name)
            return response

    corpus= []
    for result in tqdm_async(
            asyncio.as_completed([process_two_entity_relations(pair) for pair in pair_list]),
            total=len(pair_list),
            desc="Generating two entity relations"
    ):
        corpus.append(await result)

    # triple_list = []
    # for doc in entities_list:
    #     entities = doc['entities']
    #     for i in range(len(entities)):
    #         for j in range(i + 1, len(entities)):
    #             for k in range(j + 1, len(entities)):
    #                 triple = (doc['document'], entities[i], entities[j], entities[k])
    #                 triple_list.append(triple)
    #
    # async def process_three_entity_relations(triple):
    #     async with semaphore:
    #         document, entity1, entity2, entity3 = triple
    #         response = await generate_three_entity_relations(
    #             document, entity1, entity2, entity3,
    #             task.openai_system_generate_three_entity_relations,
    #             model_name)
    #         return response
    #
    # for result in tqdm_async(
    #         asyncio.as_completed([process_three_entity_relations(triple) for triple in triple_list]),
    #         total=len(triple_list),
    #         desc="Generating three entity relations"
    # ):
    #     corpus.append(await result)

    corpus = [doc['summary'] for doc in entities_list] + corpus

    qa_sft_results = {}

    async def generate_qa_sft(content):
        async with semaphore:
            completion = await gptqa(content, model_name, task.openai_system_quality_qa_sft)
            return completion


    for result in tqdm_async(
            asyncio.as_completed([generate_qa_sft(content) for content in corpus]),
            total=len(corpus),
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
