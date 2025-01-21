import os
import json
import time
import argparse
import yaml
from dotenv import load_dotenv

from graphgen.graphgen import GraphGen
from models import OpenAIModel, Tokenizer, TraverseStrategy
from utils import set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
unique_id = int(time.time())
set_logger(os.path.join(sys_path, "cache", "logs", f"graphgen_{unique_id}.log"), if_stream=False)
config_path = os.path.join(sys_path, "cache", "configs", f"graphgen_{unique_id}.yaml")

load_dotenv()

def save_config(global_config):
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding='utf-8') as config_file:
        yaml.dump(global_config, config_file, default_flow_style=False, allow_unicode=True)

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
    parser.add_argument('--web_search',
                        help='Search node info from wiki.',
                        action='store_true',
                        default=False)
    parser.add_argument('--tokenizer',
                        help='Tokenizer name.',
                        default='cl100k_base',
                        type=str)

    args = parser.parse_args()
    input_file = args.input_file

    if args.data_type == 'raw':
        with open(input_file, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif args.data_type == 'chunked':
        with open(input_file, "r", encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    synthesizer_llm_client = OpenAIModel(
        model_name=os.getenv("TEACHER_MODEL"),
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )
    training_llm_client = OpenAIModel(
        model_name=os.getenv("STUDENT_MODEL"),
        api_key=os.getenv("STUDENT_API_KEY"),
        base_url=os.getenv("STUDENT_BASE_URL")
    )

    traverse_strategy = TraverseStrategy()

    graph_gen = GraphGen(
        unique_id=unique_id,
        teacher_llm_client=synthesizer_llm_client,
        student_llm_client=training_llm_client,
        if_web_search=args.web_search,
        tokenizer_instance=Tokenizer(
            model_name=args.tokenizer
        ),
        traverse_strategy=traverse_strategy
    )

    graph_gen.insert(data, args.data_type)

    graph_gen.quiz(max_samples=2)

    graph_gen.judge(re_judge=False)

    graph_gen.traverse()

    config = {
        "unique_id": unique_id,
        "input_file": input_file,
        "data_type": args.data_type,
        "web_search": args.web_search,
        "tokenizer": args.tokenizer,
        "teacher_model": os.getenv("TEACHER_MODEL"),
        "student_model": os.getenv("STUDENT_MODEL"),
    }
    traverse_strategy_yaml = traverse_strategy.to_yaml()
    config.update(traverse_strategy_yaml)
    save_config(config)
