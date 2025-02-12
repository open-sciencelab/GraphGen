# import copy
# import os
# import json
# import gradio as gr
#
# from models import TraverseStrategy, NetworkXStorage, Tokenizer
# from webui.charts import plot_pre_length_distribution, plot_post_synth_length_distribution, plot_loss_distribution
# from graphgen.operators.split_graph import get_batches_with_strategy
# from utils import create_event_loop
#
#
# if __name__ == "__main__":
#     networkx_storage = NetworkXStorage(
#         '/cache', namespace="graph"
#     )
#
#     async def get_batches(traverse_strategy: TraverseStrategy):
#         nodes = await networkx_storage.get_all_nodes()
#         edges = await networkx_storage.get_all_edges()
#
#         nodes = list(nodes)
#         edges = list(edges)
#
#         # deepcopy
#         nodes = [(node[0], node[1].copy()) for node in nodes]
#         edges = [(edge[0], edge[1], edge[2].copy()) for edge in edges]
#
#         nodes = copy.deepcopy(nodes)
#         edges = copy.deepcopy(edges)
#         assert all('length' in edge[2] for edge in edges)
#         assert all('length' in node[1] for node in nodes)
#
#         return await get_batches_with_strategy(nodes, edges, networkx_storage, traverse_strategy)
#
#     def traverse_graph(
#         ts_bidirectional: bool,
#         ts_expand_method: str,
#         ts_max_extra_edges: int,
#         ts_max_tokens: int,
#         ts_max_depth: int,
#         ts_edge_sampling: str,
#         ts_isolated_node_strategy: str
#     ) -> str:
#         traverse_strategy = TraverseStrategy(
#             bidirectional=ts_bidirectional,
#             expand_method=ts_expand_method,
#             max_extra_edges=ts_max_extra_edges,
#             max_tokens=ts_max_tokens,
#             max_depth=ts_max_depth,
#             edge_sampling=ts_edge_sampling,
#             isolated_node_strategy=ts_isolated_node_strategy
#         )
#
#         loop = create_event_loop()
#         batches = loop.run_until_complete(get_batches(traverse_strategy))
#         loop.close()
#
#         data = []
#         for _process_batch in batches:
#             pre_length = sum(node['length'] for node in _process_batch[0]) + sum(
#                 edge[2]['length'] for edge in _process_batch[1])
#             data.append({
#                 'pre_length': pre_length
#             })
#         fig = plot_pre_length_distribution(data)
#
#         return fig
#
#
#     def update_sliders(method_name):
#         if method_name == "max_tokens":
#             return gr.update(visible=True), gr.update(visible=False)  # Show max_tokens, hide max_extra_edges
#         return gr.update(visible=False), gr.update(visible=True)  # Hide max_tokens, show max_extra_edges
#
#
#     with gr.Blocks() as app:
#         with gr.Tab("Before Traversal"):
#             with gr.Row():
#                 with gr.Column():
#                     bidirectional = gr.Checkbox(label="Bidirectional", value=False)
#                     expand_method = gr.Dropdown(
#                         choices=["max_width", "max_tokens"],
#                         value="max_tokens",
#                         label="Expand Method",
#                         interactive=True
#                     )
#
#                     # Initialize sliders
#                     max_extra_edges = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Max Extra Edges",
#                                                 visible=False)
#                     max_tokens = gr.Slider(minimum=128, maximum=8 * 1024, value=1024, step=128, label="Max Tokens")
#                     max_depth = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Depth")
#                     edge_sampling = gr.Dropdown(
#                         choices=["max_loss", "random", "min_loss"],
#                         value="max_loss",
#                         label="Edge Sampling Strategy"
#                     )
#                     isolated_node_strategy = gr.Dropdown(
#                         choices=["add", "ignore", "connect"],
#                         value="add",
#                         label="Isolated Node Strategy"
#                     )
#                     submit_btn = gr.Button("Traverse Graph")
#
#             with gr.Row():
#                 output_plot = gr.Plot(label="Graph Visualization")
#
#             # Set up event listener for expand_method dropdown
#             expand_method.change(fn=update_sliders, inputs=expand_method, outputs=[max_tokens, max_extra_edges])
#
#             submit_btn.click(
#                 fn=traverse_graph,
#                 inputs=[
#                     bidirectional,
#                     expand_method,
#                     max_extra_edges,
#                     max_tokens,
#                     max_depth,
#                     edge_sampling,
#                     isolated_node_strategy
#                 ],
#                 outputs=[output_plot]
#             )
#
#         with gr.Tab("After Synthesis"):
#             with gr.Row():
#                 with gr.Column():
#                     file_list = os.listdir("../cache/data/graphgen")
#                     input_file = gr.Dropdown(choices=file_list, label="Input File")
#                     file_button = gr.Button("Submit File")
#
#             with gr.Row():
#                 output_plot = gr.Plot(label="Graph Visualization")
#
#             def synthesize_text(file):
#                 tokenizer = Tokenizer()
#                 with open(f"cache/data/graphgen/{file}", "r", encoding='utf-8') as f:
#                     data = json.load(f)
#                 stats = []
#                 for key in data:
#                     item = data[key]
#                     item['post_length'] = len(tokenizer.encode_string(item['answer']))
#                     stats.append({
#                         'post_length': item['post_length']
#                     })
#                 fig = plot_post_synth_length_distribution(stats)
#                 return fig
#             file_button.click(
#                 fn=synthesize_text,
#                 inputs=[input_file],
#                 outputs=[output_plot]
#             )
#
#         with gr.Tab("After Judgement"):
#             with gr.Row():
#                 with gr.Column():
#                     file_list = os.listdir("../cache/data/graphgen")
#                     input_file = gr.Dropdown(choices=file_list, label="Input File")
#                     file_button = gr.Button("Submit File")
#
#             with gr.Row():
#                 output_plot = gr.Plot(label="Graph Visualization")
#
#             def judge_graph(file):
#                 with open(f"cache/data/graphgen/{file}", "r", encoding='utf-8') as f:
#                     data = json.load(f)
#                 stats = []
#                 for key in data:
#                     item = data[key]
#                     item['average_loss'] = sum(loss[2] for loss in item['losses']) / len(item['losses'])
#                     stats.append({
#                         'average_loss': item['average_loss']
#                     })
#                 fig = plot_loss_distribution(stats)
#                 return fig
#
#             file_button.click(
#                 fn=judge_graph,
#                 inputs=[input_file],
#                 outputs=[output_plot]
#             )
#
#
#     app.launch()

import json
import os
import gradio as gr
import yaml
from graphgen.graphgen import GraphGen
from models import OpenAIModel, Tokenizer, TraverseStrategy


def load_config() -> dict:
    with open("config.yaml", "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    with open("config.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config, f)


def load_env() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding='utf-8') as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env[key] = value
    return env


def save_env(env: dict):
    with open(".env", "w", encoding='utf-8') as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")


def init_graph_gen(config: dict, env: dict) -> GraphGen:
    graph_gen = GraphGen()

    # Set up LLM clients
    graph_gen.synthesizer_llm_client = OpenAIModel(
        model_name=env.get("TEACHER_MODEL", ""),
        base_url=env.get("TEACHER_BASE_URL", ""),
        api_key=env.get("TEACHER_API_KEY", "")
    )

    graph_gen.training_llm_client = OpenAIModel(
        model_name=env.get("STUDENT_MODEL", ""),
        base_url=env.get("STUDENT_BASE_URL", ""),
        api_key=env.get("STUDENT_API_KEY", "")
    )

    # Set up tokenizer
    graph_gen.tokenizer_instance = Tokenizer(config.get("tokenizer", "cl100k_base"))

    # Set up traverse strategy
    strategy_config = config.get("traverse_strategy", {})
    graph_gen.traverse_strategy = TraverseStrategy(
        qa_form=config.get("qa_form", "multi_hop"),
        expand_method=strategy_config.get("expand_method", "max_tokens"),
        bidirectional=strategy_config.get("bidirectional", True),
        max_extra_edges=strategy_config.get("max_extra_edges", 5),
        max_tokens=strategy_config.get("max_tokens", 256),
        max_depth=strategy_config.get("max_depth", 2),
        edge_sampling=strategy_config.get("edge_sampling", "max_loss"),
        isolated_node_strategy=strategy_config.get("isolated_node_strategy", "add"),
        difficulty_order=strategy_config.get("difficulty_order", ["medium", "medium", "medium"])
    )

    graph_gen.if_web_search = config.get("web_search", False)

    return graph_gen

# pylint: disable=redefined-outer-name
def run_graphgen(
        input_file: str,
        data_type: str,
        qa_form: str,
        tokenizer: str,
        web_search: bool,
        expand_method: str,
        bidirectional: bool,
        max_extra_edges: int,
        max_tokens: int,
        max_depth: int,
        edge_sampling: str,
        isolated_node_strategy: str,
        difficulty_level: str,
        teacher_model: str,
        teacher_base_url: str,
        teacher_api_key: str,
        student_model: str,
        student_base_url: str,
        student_api_key: str,
        quiz_samples: int,
        progress=gr.Progress()
) -> str:
    # Save configurations
    config = {
        "data_type": data_type,
        "input_file": input_file,
        "qa_form": qa_form,
        "tokenizer": tokenizer,
        "web_search": web_search,
        "quiz_samples": quiz_samples,
        "traverse_strategy": {
            "expand_method": expand_method,
            "bidirectional": bidirectional,
            "max_extra_edges": max_extra_edges,
            "max_tokens": max_tokens,
            "max_depth": max_depth,
            "edge_sampling": edge_sampling,
            "isolated_node_strategy": isolated_node_strategy,
            "difficulty_order": [difficulty_level] * 3
        }
    }
    save_config(config)

    env = {
        "TEACHER_MODEL": teacher_model,
        "TEACHER_BASE_URL": teacher_base_url,
        "TEACHER_API_KEY": teacher_api_key,
        "STUDENT_MODEL": student_model,
        "STUDENT_BASE_URL": student_base_url,
        "STUDENT_API_KEY": student_api_key
    }
    save_env(env)

    # Initialize GraphGen
    graph_gen = init_graph_gen(config, env)

    try:
        # Load input data
        if data_type == "raw":
            with open(input_file, "r", encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        else:  # chunked
            with open(input_file, "r", encoding='utf-8') as f:
                data = json.load(f)

        progress(0.2, "Data loaded")

        # Process the data
        graph_gen.insert(data, data_type)
        progress(0.4, "Data inserted")

        # Generate quiz
        graph_gen.quiz(max_samples=quiz_samples)
        progress(0.6, "Quiz generated")

        # Judge statements
        graph_gen.judge(re_judge=False)
        progress(0.8, "Statements judged")

        # Traverse graph
        graph_gen.traverse()
        progress(1.0, "Graph traversed")

        return "Task completed successfully!"

    except Exception as e: # pylint: disable=broad-except
        return f"Error occurred: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="GraphGen Configuration") as iface:
    with gr.Row():
        # Input Configuration Column
        with gr.Column(scale=1):
            gr.Markdown("### Input Configuration")
            input_file = gr.Textbox(label="Input File Path", value="resources/examples/raw_demo.jsonl")
            data_type = gr.Radio(choices=["raw", "chunked"], label="Data Type", value="raw")
            qa_form = gr.Radio(choices=["atomic", "multi_hop", "open"], label="QA Form", value="multi_hop")
            tokenizer = gr.Textbox(label="Tokenizer", value="cl100k_base")
            web_search = gr.Checkbox(label="Enable Web Search", value=False)
            quiz_samples = gr.Number(label="Quiz Samples", value=2, minimum=1)

        # Traverse Strategy Column
        with gr.Column(scale=1):
            gr.Markdown("### Traverse Strategy")
            expand_method = gr.Radio(choices=["max_width", "max_tokens"], label="Expand Method", value="max_tokens")
            bidirectional = gr.Checkbox(label="Bidirectional", value=True)
            max_extra_edges = gr.Slider(minimum=1, maximum=10, value=5, label="Max Extra Edges", step=1)
            max_tokens = gr.Slider(minimum=64, maximum=1024, value=256, label="Max Tokens", step=64)
            max_depth = gr.Slider(minimum=1, maximum=5, value=2, label="Max Depth", step=1)
            edge_sampling = gr.Radio(choices=["max_loss", "min_loss", "random"], label="Edge Sampling",
                                     value="max_loss")
            isolated_node_strategy = gr.Radio(choices=["add", "ignore"], label="Isolated Node Strategy", value="ignore")
            difficulty_level = gr.Radio(choices=["easy", "medium", "hard"], label="Difficulty Level", value="medium")

        # Model Configuration Column
        with gr.Column(scale=1):
            gr.Markdown("### Model Configuration")
            teacher_model = gr.Textbox(label="Synthesizer Model")
            teacher_base_url = gr.Textbox(label="Synthesizer Base URL")
            teacher_api_key = gr.Textbox(label="Synthesizer API Key", type="password")
            student_model = gr.Textbox(label="Trainee Model")
            student_base_url = gr.Textbox(label="Trainee Base URL")
            student_api_key = gr.Textbox(label="Trainee API Key", type="password")

    # Submission and Output Rows
    with gr.Row():
        submit_btn = gr.Button("Run GraphGen")
    with gr.Row():
        output = gr.Textbox(label="Output")

    # Event Handling
    submit_btn.click(
        run_graphgen,
        inputs=[
            input_file, data_type, qa_form, tokenizer, web_search,
            expand_method, bidirectional, max_extra_edges, max_tokens,
            max_depth, edge_sampling, isolated_node_strategy, difficulty_level,
            teacher_model, teacher_base_url, teacher_api_key,
            student_model, student_base_url, student_api_key,
            quiz_samples
        ],
        outputs=output
    )

if __name__ == "__main__":
    iface.launch()
