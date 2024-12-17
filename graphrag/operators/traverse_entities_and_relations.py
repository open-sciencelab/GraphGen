from models import NetworkXStorage

async def get_nodes_and_edges(graph_storage: NetworkXStorage):
    """
    Get nodes and edges from the graph storage
    First, get all nodes
    Then, for each node, get all its edges

    :param graph_storage:
    :return: [{node_id: str, node_data: dict, edges: list[dict]}]
    """
    nodes = await graph_storage.get_all_nodes()
    nodes_and_edges = []
    for node in nodes:
        print(node)
        node_id = node[0]
        node_data = node[1]
        print(node_id)
        print(node_data)
        edges = await graph_storage.get_node_edges(node_id)
        print(edges)
        nodes_and_edges.append({
            "node_id": node_id,
            "node_data": node_data,
            "edges": edges
        })
    return nodes_and_edges
