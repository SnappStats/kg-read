import datetime as dt
import json
import os
import logging
from dotenv import load_dotenv
from floggit import flog
from google.cloud import spanner
import vertexai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
INSTANCE_ID = "knowledge-graph"
DATABASE_ID = "kg"

SPANNER_DATABASE = spanner.Client(
        project=PROJECT_ID).instance(INSTANCE_ID).database(DATABASE_ID)

vertexai.init(project=PROJECT_ID)
EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained("gemini-embedding-001")


def fetch_from_database():
    with SPANNER_DATABASE.snapshot() as snapshot:
        entities = snapshot.execute_sql("select * from entity")
    with SPANNER_DATABASE.snapshot() as snapshot:
        relationships = snapshot.execute_sql("select * from relationship")

    return entities, relationships


def embed_entity(entity: dict) -> list[float]:
    """Embeds an entity as a 768-dimensional vector using gemini-embedding-001."""
    entity_for_embedding = {
        k: v for k, v in entity.items()
        if k not in ('entity_id', 'updated_by', 'updated_at')
    }
    text = json.dumps(entity_for_embedding)
    embeddings = EMBEDDING_MODEL.get_embeddings([text], output_dimensionality=768)
    return embeddings[0].values


def embed_query(query: str) -> list[float]:
    """Embeds a query string as a 768-dimensional vector."""
    embeddings = EMBEDDING_MODEL.get_embeddings([query], output_dimensionality=768)
    return embeddings[0].values


@flog
def store_graph_delta(remove_subgraph: dict, add_subgraph: dict):
    entities_to_upsert = [
        [
            e['entity_id'],
            e['entity_names'],
            dt.datetime.strptime(e['updated_at'], "%Y-%m-%dT%H:%M:%S%z"),
            e['updated_by'],
            json.dumps(e.get('properties', {})),
            embed_entity(e)
        ]
        for e in add_subgraph['entities'].values()
    ]

    relationships_to_upsert = [
        [
            r["source_entity_id"],
            r['target_entity_id'],
            r["relationship"]
        ]
        for r in add_subgraph['relationships']
    ]

    entities_to_delete = [
            [entity_id] for entity_id in remove_subgraph['entities']]

    relationships_to_delete = [
        (r['source_entity_id'], r['target_entity_id'], r['relationship'])
        for r in remove_subgraph['relationships']
    ]

    def execute(transaction):
        if relationships_to_delete:
            transaction.delete(
                    'relationship', keyset=spanner.KeySet(keys=relationships_to_delete))

        if entities_to_delete:
            transaction.delete(
                    'entity', keyset=spanner.KeySet(keys=entities_to_delete))

        if entities_to_upsert:
            transaction.insert_or_update(
                'entity',
                columns=['entity_id', 'entity_names', 'updated_at', 'updated_by', 'properties', 'embedding'],
                values=entities_to_upsert
            )

        if relationships_to_upsert:
            transaction.insert_or_update(
                'relationship',
                columns=['source_entity_id', 'target_entity_id', 'relationship'],
                values=relationships_to_upsert
            )

    if (
        entities_to_upsert
        or relationships_to_upsert
        or entities_to_delete
        or relationships_to_delete
    ):
        try:
            results = SPANNER_DATABASE.run_in_transaction(execute)
        except Exception as e:
            print('Transaction failed; rolled back.')
            logging.exception(e)
            return {
                'entities_inserted_or_updated': [],
                'entities_deleted': [],
                'relationships_inserted_or_updated': [],
                'relationships_deleted': []
            }

    return {
        'entities_inserted_or_updated': entities_to_upsert,
        'entities_deleted': entities_to_delete,
        'relationships_inserted_or_updated': relationships_to_upsert,
        'relationships_deleted': relationships_to_delete
    }


@flog
def get_relevant_entities_from_db(
    query: str,
    top_k: int = 10,
    distance_threshold: float = 0.5
) -> list[dict]:
    """
    Finds relevant entities using hybrid search: exact name match + vector similarity.

    Args:
        query: The search query.
        top_k: Maximum number of results to return.
        distance_threshold: Max cosine distance for vector matches (lower = more similar).

    Returns:
        List of entity dicts with entity_id, entity_names, properties, and distance.
    """
    query_embedding = embed_query(query)

    sql = """
        SELECT
            entity_id,
            entity_names,
            properties,
            COSINE_DISTANCE(embedding, @query_embedding) as distance
        FROM entity
        WHERE
            EXISTS (SELECT 1 FROM UNNEST(entity_names) AS name WHERE LOWER(name) LIKE LOWER(@query_pattern))
            OR COSINE_DISTANCE(embedding, @query_embedding) < @threshold
        ORDER BY distance
        LIMIT @top_k
    """

    params = {
        'query_embedding': query_embedding,
        'query_pattern': f'%{query}%',
        'threshold': distance_threshold,
        'top_k': top_k
    }
    param_types = {
        'query_embedding': spanner.param_types.Array(spanner.param_types.FLOAT32),
        'query_pattern': spanner.param_types.STRING,
        'threshold': spanner.param_types.FLOAT64,
        'top_k': spanner.param_types.INT64
    }

    with SPANNER_DATABASE.snapshot() as snapshot:
        results = snapshot.execute_sql(sql, params=params, param_types=param_types)
        entities = [
            {
                'entity_id': row[0],
                'entity_names': list(row[1]),
                'properties': json.loads(row[2]) if row[2] else {},
                'distance': row[3]
            }
            for row in results
        ]

    return entities


@flog
def get_knowledge_subgraph_from_db(
    entity_ids: set[str],
    num_hops: int = 2
) -> dict:
    """
    Gets a subgraph neighborhood around the given entities using Spanner Graph traversal.

    Args:
        entity_ids: Set of entity IDs to start from.
        num_hops: Number of hops to traverse (1 or 2).

    Returns:
        Dict with 'entities' and 'relationships' for the subgraph.
    """
    if not entity_ids:
        return {'entities': {}, 'relationships': []}

    entity_ids_list = list(entity_ids)
    params = {'entity_ids': entity_ids_list}
    param_types = {'entity_ids': spanner.param_types.Array(spanner.param_types.STRING)}

    gql = f"""
        GRAPH knowledge_graph
        MATCH path = (start:Entity)-[r:Related]-{{1,{num_hops}}}(neighbor:Entity)
        WHERE start.entity_id IN UNNEST(@entity_ids)
        RETURN
            start.entity_id AS start_id,
            start.entity_names AS start_names,
            start.properties AS start_props,
            neighbor.entity_id AS neighbor_id,
            neighbor.entity_names AS neighbor_names,
            neighbor.properties AS neighbor_props,
            r.source_entity_id AS rel_source,
            r.target_entity_id AS rel_target,
            r.relationship AS rel_type
    """

    entities = {}
    relationships = []
    seen_rels = set()

    with SPANNER_DATABASE.snapshot() as snapshot:
        results = snapshot.execute_sql(gql, params=params, param_types=param_types)
        for row in results:
            # Add start entity
            if row[0] not in entities:
                entities[row[0]] = {
                    'entity_id': row[0],
                    'entity_names': list(row[1]),
                    'properties': json.loads(row[2]) if row[2] else {}
                }
            # Add neighbor entity
            if row[3] not in entities:
                entities[row[3]] = {
                    'entity_id': row[3],
                    'entity_names': list(row[4]),
                    'properties': json.loads(row[5]) if row[5] else {}
                }
            # Add relationship (deduplicated)
            rel_key = (row[6], row[7], row[8])
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                relationships.append({
                    'source_entity_id': row[6],
                    'target_entity_id': row[7],
                    'relationship': row[8]
                })

    # Mark entities with external neighbors
    all_entity_ids = list(entities.keys())
    ext_params = {'entity_ids': all_entity_ids}
    ext_param_types = {'entity_ids': spanner.param_types.Array(spanner.param_types.STRING)}

    ext_gql = """
        GRAPH knowledge_graph
        MATCH (e:Entity)-[:Related]-(outside:Entity)
        WHERE e.entity_id IN UNNEST(@entity_ids)
          AND outside.entity_id NOT IN UNNEST(@entity_ids)
        RETURN DISTINCT e.entity_id AS entity_id
    """

    with SPANNER_DATABASE.snapshot() as snapshot:
        ext_results = snapshot.execute_sql(ext_gql, params=ext_params, param_types=ext_param_types)
        external_entity_ids = {row[0] for row in ext_results}

    for entity_id in entities:
        entities[entity_id]['has_external_neighbor'] = entity_id in external_entity_ids

    return {
        'entities': entities,
        'relationships': relationships
    }


@flog
def get_random_entity_from_db() -> dict:
    """Gets a random entity from the database."""
    sql = """
        SELECT entity_id, entity_names, properties
        FROM entity
        TABLESAMPLE RESERVOIR (1 ROWS)
    """

    with SPANNER_DATABASE.snapshot() as snapshot:
        results = snapshot.execute_sql(sql)
        for row in results:
            return {
                'entity_id': row[0],
                'entity_names': list(row[1]),
                'properties': json.loads(row[2]) if row[2] else {}
            }

    return None
