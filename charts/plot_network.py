from pyecharts import options as opts
from pyecharts.charts import Graph
from models import NetworkXStorage

async def plot_network(kg_instance: NetworkXStorage):
    """
    Plot network graph

    :param kg_instance
    :return
    """
    entities = await kg_instance.get_all_nodes()
    relations = await kg_instance.get_all_edges()

    nodes = []
    links = []

    for entity in entities:
        node_id = entity[0]
        node_data = entity[1]
        nodes.append({
            "name": node_id,
            "symbolSize": 50,
            "category": node_data["entity_type"],
            "tooltip": {
                "formatter": f"{node_id}: {node_data['description']}"
            }
        })

    for relation in relations:
        source = relation[0]
        target = relation[1]
        links.append({
            "source": source,
            "target": target,
            "value": 1,
            "tooltip": {
                "formatter": f"{source} -> {target}: {relation[2]['description']}"
            }
        })

    categories = list({node['category'] for node in nodes})

    for node in nodes:
        node['category'] = categories.index(node['category'])

    categories = [{"name": category} for category in set(categories)]

    graph = (
        Graph(init_opts=opts.InitOpts(width="1000px", height="800px"))
        .add("",
             nodes,
             links,
             categories=categories,
             layout="force",
             edge_symbol=["circle", "arrow"],
             edge_symbol_size=10,
             linestyle_opts=opts.LineStyleOpts(
                 width=1,
                 opacity=0.7,
                 curve=0.3
             ),
             label_opts=opts.LabelOpts(is_show=True),
             repulsion=1000
             )
        .set_global_opts(title_opts=opts.TitleOpts(title="网络图"))
    )
    graph.render("网络图.html")


if __name__ == "__main__":
    networkx_storage = NetworkXStorage(
        '/home/PJLAB/chenzihong/Project/graphgen/cache', namespace="graph"
    )

    import asyncio
    asyncio.run(plot_network(networkx_storage))
