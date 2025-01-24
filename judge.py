import os
import argparse
import asyncio
from dotenv import load_dotenv

from models import NetworkXStorage, JsonKVStorage, OpenAIModel
from graphgen.operators import judge_statement

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='cache/output/new_graph.graphml', help='path to save output')

    args = parser.parse_args()

    llm_client = OpenAIModel(
        model_name=os.getenv("STUDENT_MODEL"),
        api_key=os.getenv("STUDENT_API_KEY"),
        base_url=os.getenv("STUDENT_BASE_URL")
    )

    graph_storage = NetworkXStorage(
        os.path.join(sys_path, "cache"),
        namespace="graph"
    )

    rephrase_storage = JsonKVStorage(
        os.path.join(sys_path, "cache"),
        namespace="rephrase"
    )

    new_graph = asyncio.run(judge_statement(llm_client, graph_storage, rephrase_storage, re_judge=True))

    graph_file = asyncio.run(graph_storage.get_graph())

    new_graph.write_nx_graph(graph_file, args.output)
