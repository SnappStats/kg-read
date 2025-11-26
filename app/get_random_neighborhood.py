import random
from floggit import flog

from utils import fetch_knowledge_graph, get_knowledge_subgraph


@flog
def main(graph_id: str) -> dict:
    """
    Args:
        graph_id (str): The ID of the knowledge graph to query.

    Returns:
        dict: A random entity from the knowledge graph along with its surrounding neighborhood.
    """
    g = fetch_knowledge_graph(graph_id=graph_id)
    entity_id = random.choice(list(g['entities'].keys()))
    entity = g['entities'][entity_id]
    nbhd = get_knowledge_subgraph(
            entity_ids={entity_id}, graph=g, num_hops=1)

    entity_and_nbhd = {
        'entity': entity,
        'entity_neighborhood': nbhd
    }

    return entity_and_nbhd
