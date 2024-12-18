from models import NetworkXStorage
from utils import logger

async def traverse_entities_and_relations(graph_storage: NetworkXStorage):
    """
    Get all entities and relations and their edges
    First, get all nodes
    Then, for each node, get all its edges

    :param graph_storage:
    :return:
    """
    nodes = await graph_storage.get_all_nodes()
    edges = await graph_storage.get_all_edges()
    print(edges)
    nodes_and_edges = []
    for node in nodes:
        logger.info(f"Traversing node {node}")
        node_id = node[0]
        node_data = node[1]
        print("node_id", node_id)
        edges = await graph_storage.get_node_edges(node_id)
        print("edges", edges)
        print(len(edges))
    #     node_id = node[0]
    #     node_data = node[1]
    #     print(node_id)
    #     print(node_data)
    #     edges = await graph_storage.get_node_edges(node_id)
    #     print(edges)
    #     nodes_and_edges.append({
    #         "node_id": node_id,
    #         "node_data": node_data,
    #         "edges": edges
    #     })
    # return nodes_and_edges
