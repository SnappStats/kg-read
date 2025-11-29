from floggit import flog

from kg_service import get_relevant_entities_from_db, get_knowledge_subgraph_from_db


@flog
def main(query: str, num_hops: int = 1) -> dict:
    """
    Args:
        query (str): A user query that might be relevant to some entities in the knowledge graph.
        num_hops (int): Number of hops to traverse from relevant entities (default 1).

    Returns:
        dict: A relevant subgraph of the knowledge graph, including a surrounding neighborhood of the relevant entities (to help patching in a replacement subgraph).
    """
    relevant_entities = get_relevant_entities_from_db(query=query)
    relevant_entity_ids = {e['entity_id'] for e in relevant_entities}
    neighborhood = get_knowledge_subgraph_from_db(
            entity_ids=relevant_entity_ids, num_hops=num_hops)

    return neighborhood
