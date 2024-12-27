import os
import json
import argparse
from graphrag.graphrag import GraphRag
from models import OpenAIModel, Tokenizer
from dotenv import load_dotenv
from utils import set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
set_logger(os.path.join(sys_path, "cache", "logs", "graphrag.log"))

load_dotenv()

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
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
    elif args.data_type == 'chunked':
        with open(input_file, "r") as f:
            data = json.load(f)

    teacher_llm_client = OpenAIModel(
        model_name=os.getenv("TEACHER_MODEL"),
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )
    student_llm_client = OpenAIModel(
        model_name=os.getenv("STUDENT_MODEL"),
        api_key=os.getenv("STUDENT_API_KEY"),
        base_url=os.getenv("STUDENT_BASE_URL")
    )

    graph_rag = GraphRag(
        teacher_llm_client=teacher_llm_client,
        student_llm_client=student_llm_client,
        if_web_search=args.web_search,
        tokenizer_instance=Tokenizer(
            model_name=args.tokenizer
        )
    )

    graph_rag.insert(data, args.data_type)

    graph_rag.judge()

    graph_rag.traverse()
