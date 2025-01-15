"""Simulate text length distributions using input data distributions when rephrasing."""

import copy
import os
import json
import gradio as gr

from models import TraverseStrategy, NetworkXStorage, Tokenizer
from charts import plot_pre_length_distribution, plot_post_synth_length_distribution, plot_loss_distribution
from graphgen.operators.split_graph import get_batches_with_strategy
from utils import create_event_loop


if __name__ == "__main__":
    networkx_storage = NetworkXStorage(
        '/home/PJLAB/chenzihong/Project/graphgen/cache', namespace="graph"
    )

    async def get_batches(traverse_strategy: TraverseStrategy):
        nodes = await networkx_storage.get_all_nodes()
        edges = await networkx_storage.get_all_edges()

        nodes = list(nodes)
        edges = list(edges)

        # deepcopy
        nodes = [(node[0], node[1].copy()) for node in nodes]
        edges = [(edge[0], edge[1], edge[2].copy()) for edge in edges]

        nodes = copy.deepcopy(nodes)
        edges = copy.deepcopy(edges)
        assert all('length' in edge[2] for edge in edges)
        assert all('length' in node[1] for node in nodes)

        return await get_batches_with_strategy(nodes, edges, networkx_storage, traverse_strategy)

    def traverse_graph(
        ts_bidirectional: bool,
        ts_expand_method: str,
        ts_max_extra_edges: int,
        ts_max_tokens: int,
        ts_max_depth: int,
        ts_edge_sampling: str,
        ts_isolated_node_strategy: str
    ) -> str:
        traverse_strategy = TraverseStrategy(
            bidirectional=ts_bidirectional,
            expand_method=ts_expand_method,
            max_extra_edges=ts_max_extra_edges,
            max_tokens=ts_max_tokens,
            max_depth=ts_max_depth,
            edge_sampling=ts_edge_sampling,
            isolated_node_strategy=ts_isolated_node_strategy
        )

        loop = create_event_loop()
        batches = loop.run_until_complete(get_batches(traverse_strategy))
        loop.close()

        data = []
        for _process_batch in batches:
            pre_length = sum(node['length'] for node in _process_batch[0]) + sum(
                edge[2]['length'] for edge in _process_batch[1])
            data.append({
                'pre_length': pre_length
            })
        fig = plot_pre_length_distribution(data)

        return fig


    def update_sliders(method_name):
        if method_name == "max_tokens":
            return gr.update(visible=True), gr.update(visible=False)  # Show max_tokens, hide max_extra_edges
        return gr.update(visible=False), gr.update(visible=True)  # Hide max_tokens, show max_extra_edges


    with gr.Blocks() as app:
        with gr.Tab("Before Traversal"):
            with gr.Row():
                with gr.Column():
                    bidirectional = gr.Checkbox(label="Bidirectional", value=False)
                    expand_method = gr.Dropdown(
                        choices=["max_width", "max_tokens"],
                        value="max_tokens",
                        label="Expand Method",
                        interactive=True
                    )

                    # Initialize sliders
                    max_extra_edges = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Max Extra Edges",
                                                visible=False)
                    max_tokens = gr.Slider(minimum=128, maximum=8 * 1024, value=1024, step=128, label="Max Tokens")
                    max_depth = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Depth")
                    edge_sampling = gr.Dropdown(
                        choices=["max_loss", "random", "min_loss"],
                        value="max_loss",
                        label="Edge Sampling Strategy"
                    )
                    isolated_node_strategy = gr.Dropdown(
                        choices=["add", "ignore", "connect"],
                        value="add",
                        label="Isolated Node Strategy"
                    )
                    submit_btn = gr.Button("Traverse Graph")

            with gr.Row():
                output_plot = gr.Plot(label="Graph Visualization")

            # Set up event listener for expand_method dropdown
            expand_method.change(fn=update_sliders, inputs=expand_method, outputs=[max_tokens, max_extra_edges])

            submit_btn.click(
                fn=traverse_graph,
                inputs=[
                    bidirectional,
                    expand_method,
                    max_extra_edges,
                    max_tokens,
                    max_depth,
                    edge_sampling,
                    isolated_node_strategy
                ],
                outputs=[output_plot]
            )

        with gr.Tab("After Synthesis"):
            with gr.Row():
                with gr.Column():
                    file_list = os.listdir("cache/data/graphgen")
                    input_file = gr.Dropdown(choices=file_list, label="Input File")
                    file_button = gr.Button("Submit File")

            with gr.Row():
                output_plot = gr.Plot(label="Graph Visualization")

            def synthesize_text(file):
                tokenizer = Tokenizer()
                with open(f"cache/data/graphgen/{file}", "r", encoding='utf-8') as f:
                    data = json.load(f)
                stats = []
                for key in data:
                    item = data[key]
                    item['post_length'] = len(tokenizer.encode_string(item['answer']))
                    stats.append({
                        'post_length': item['post_length']
                    })
                fig = plot_post_synth_length_distribution(stats)
                return fig
            file_button.click(
                fn=synthesize_text,
                inputs=[input_file],
                outputs=[output_plot]
            )

        with gr.Tab("After Judgement"):
            with gr.Row():
                with gr.Column():
                    file_list = os.listdir("cache/data/graphgen")
                    input_file = gr.Dropdown(choices=file_list, label="Input File")
                    file_button = gr.Button("Submit File")

            with gr.Row():
                output_plot = gr.Plot(label="Graph Visualization")

            def judge_graph(file):
                with open(f"cache/data/graphgen/{file}", "r", encoding='utf-8') as f:
                    data = json.load(f)
                stats = []
                for key in data:
                    item = data[key]
                    item['average_loss'] = sum(loss[2] for loss in item['losses']) / len(item['losses'])
                    stats.append({
                        'average_loss': item['average_loss']
                    })
                fig = plot_loss_distribution(stats)
                return fig

            file_button.click(
                fn=judge_graph,
                inputs=[input_file],
                outputs=[output_plot]
            )


    app.launch()
