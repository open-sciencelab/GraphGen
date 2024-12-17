from pyecharts import options as opts
from pyecharts.charts import Graph
from models import NetworkXStorage

async def plot_network(networkx_storage: NetworkXStorage):
    """
    Plot network graph

    :param networkx_storage: NetworkXStorage instance
    :return:
    """
    entities = await networkx_storage.get_all_nodes()
    relations = await networkx_storage.get_all_edges()

    nodes = []
    links = []

    for entity in entities:
        node_id = entity[0]
        node_data = entity[1]
        nodes.append({"name": node_id, "symbolSize": 10, "category": node_data["entity_type"]})

    for relation in relations:
        source = relation[0]
        target = relation[1]
        weight = relation[2]["weight"]
        links.append({
            "source": source,
            "target": target,
            "value": weight
        })

    categories = list(set([node['category'] for node in nodes]))

    for node in nodes:
        node['category'] = categories.index(node['category'])

    categories = [{"name": category} for category in set(categories)]

    graph = (
        Graph(init_opts=opts.InitOpts(width="1000px", height="800px"))
        .add("",
             nodes,
             links,
             categories=categories,
             layout="force"
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

