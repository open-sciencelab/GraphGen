# https://arxiv.org/abs/2409.07431
# https://github.com/zitongyang/synthetic_continued_pretraining

import os
import json

from tqdm import tqdm

from inference.devapi import gptqa
from entigraph_utils.io_utils import jload, jdump, set_openai_key
from tasks.baseline_task import BaselineTask
import random


def generate_entities(document_content: str,
                      system_message: str,
                      openai_model: str):
    prompt = f"""
    ### Document Content:
    {document_content}
    """
    can_read_entities = None
    while not can_read_entities:
        try:
            completion = gptqa(prompt,
                               openai_model,
                               system_message,
                               json_format=True)
            response = json.loads(completion)
            can_read_entities = response['entities']
        except Exception as e:
            print(f"Failed to generate entities: {str(e)}")
    return response

def generate_two_entity_relations(document_content: str,
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
    completion = gptqa(prompt,
                       openai_model,
                       system_message)
    return completion

def generate_three_entity_relations(document_content: str,
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
    completion = gptqa(prompt,
                       openai_model,
                       system_message)
    return completion


def generate_synthetic_data_for_document(document_index: str, model_name: str, ):
    random.seed(42)
    # set_openai_key()
    # task = QuALITY('all')

    task = BaselineTask()
    document = task.documents[document_index]
    print(f"Generating synthetic data for article {document_index}")
    output_path = f'data/dataset/raw/quality_entigraph_{model_name}/{document.uid}.json'

    if os.path.exists(output_path):
        output = jload(output_path)
    else:
        output = [[]]

    # first check if entities are already generated
    if isinstance(output[0], list) and len(output[0]) > 0:
        entities = output[0]
    else:
        entities = generate_entities(
            document.text,
            task.openai_system_generate_entities,
            model_name)
        output[0] = entities['entities']
        output.append(entities['summary'])
        jdump(output, output_path)
        entities = entities['entities']

    pair_list = []
    # iterate over pairs of entities and generate relations
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            pair = (entities[i], entities[j])
            pair_list.append(pair)
    for entity1, entity2 in tqdm(pair_list):
        # if _pair_already_generated(entity1, entity2, output):
        #     continue
        response = generate_two_entity_relations(
            document.text, entity1, entity2,
            task.openai_system_generate_two_entity_relations,
            model_name)
        if response:
            output.append(response)
        jdump(output, output_path)

    # iterate over triples of entities and generate relations
    triple_list = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            for k in range(j + 1, len(entities)):
                triple = (entities[i], entities[j], entities[k])
                triple_list.append(triple)
    random.shuffle(triple_list)
    for entity1, entity2, entity3 in tqdm(triple_list):
        response = generate_three_entity_relations(
            document.text, entity1, entity2, entity3,
            task.openai_system_generate_three_entity_relations,
            model_name)
        if response:
            output.append(response)
        jdump(output, output_path)

if __name__ == '__main__':
    # seq 0 264 | xargs -P 265 -I {} sh -c 'python data/entigraph.py {} > data/dataset/log/log_gpt4turbo_{}.txt 2>&1'
    model_name = "gpt-4o-mini"
    # document_index = int(sys.argv[1])
    document_index = 0
    generate_synthetic_data_for_document(document_index, model_name)
