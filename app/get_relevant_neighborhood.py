from floggit import flog

from utils import fetch_knowledge_graph, get_relevant_entities, get_knowledge_subgraph


@flog
def main(query: str, graph_id: str) -> dict:
    """
    Args:
        query (str): A user query that might be relevanet to some entities in the knowledge graph.
        graph_id (str): The ID of the knowledge graph to query.

    Returns:
        dict: A relevant subgraph of the knowledge graph, including a surrounding neighborhood of the relevant entities (to help patching in a replacement subgraph).
    """
    g = fetch_knowledge_graph(graph_id=graph_id)

    relevant_entity_ids = get_relevant_entities(
            query=query, entities=g['entities'])
    neighborhood = get_knowledge_subgraph(
            entity_ids=relevant_entity_ids, graph=g, num_hops=1)

    return neighborhood
