from floggit import flog

from kg_service import get_random_entity_from_db, get_knowledge_subgraph_from_db


@flog
def main(num_hops: int = 1) -> dict:
    """
    Args:
        num_hops (int): Number of hops to traverse from the random entity (default 1).

    Returns:
        dict: A random entity from the knowledge graph along with its surrounding neighborhood.
    """
    entity = get_random_entity_from_db()
    if entity is None:
        return {'entity': None, 'entity_neighborhood': {'entities': {}, 'relationships': []}}

    nbhd = get_knowledge_subgraph_from_db(
            entity_ids={entity['entity_id']}, num_hops=num_hops)

    return {
        'entity': entity,
        'entity_neighborhood': nbhd
    }
