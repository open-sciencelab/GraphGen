import os
import json
import argparse
from graphrag.graphrag import GraphRag
from models import Chunk, OpenAIModel
from dotenv import load_dotenv

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
    parser.add_argument('--teacher_model',
                    default='gpt-4o-mini',
                    type=str)
    parser.add_argument('--student_model',
                    default='gpt-4o-mini',
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
        model_name=args.teacher_model or "gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    student_llm_client = OpenAIModel(
        model_name=args.student_model or "gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    graph_rag = GraphRag(
        teacher_llm_client=teacher_llm_client,
        student_llm_client=student_llm_client,
        if_web_search=args.web_search
    )

    graph_rag.insert(data, args.data_type)

    graph_rag.judge()

    graph_rag.traverse()
