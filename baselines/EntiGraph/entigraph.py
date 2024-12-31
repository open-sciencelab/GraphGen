# https://arxiv.org/abs/2409.07431
# https://github.com/zitongyang/synthetic_continued_pretraining

import json
import asyncio

from inference.devapi import gptqa
from tasks.baseline_task import BaselineTask
import random
from tqdm.asyncio import tqdm as tqdm_async


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
                               json_format=True)
            response = json.loads(completion)
            can_read_entities = response['entities']
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}")
    return response

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


async def generate_synthetic_data_for_document(model_name: str):
    random.seed(42)

    task = BaselineTask()

    async def process_single_document(doc):
        output = [[]]

        entities = await generate_entities(
            doc.text,
            task.openai_system_generate_entities,
            model_name)
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
        results.append(await result)

    with open("../../cache/data/entigraph.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    model_name = "gpt-4o-mini"

    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_synthetic_data_for_document(model_name))
