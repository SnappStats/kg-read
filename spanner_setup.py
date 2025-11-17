google_sql_ddl_statements = [
    # 0. DROP tables
    '''DROP INDEX IF EXISTS RelationshipEmbeddingIndex''',
    '''DROP TABLE IF EXISTS relationship''',
    '''DROP INDEX IF EXISTS EntityEmbeddingIndex''',
    '''DROP TABLE IF EXISTS entity''',

    # 1. CREATE entity table
    '''
    CREATE TABLE entity (
        entity_id STRING(MAX) NOT NULL,
        entity_names ARRAY<STRING(MAX)> NOT NULL,
        updated_at TIMESTAMP,
        updated_by STRING(MAX),
        properties JSON,
        embedding ARRAY<FLOAT32>(vector_length=>768)
    ) PRIMARY KEY (entity_id)
    ''',
    
    # 2. CREATE relationship table (NON-INTERLEAVED)
    '''
    CREATE TABLE relationship (
        source_entity_id STRING(MAX) NOT NULL,
        target_entity_id STRING(MAX) NOT NULL,
        relationship STRING(MAX) NOT NULL,
        embedding ARRAY<FLOAT32>(vector_length=>768)
    ) PRIMARY KEY (source_entity_id, target_entity_id, relationship)
    ''',
    
    # 3. Add Foreign Key constraints
    '''
    ALTER TABLE relationship 
    ADD CONSTRAINT FK_SourceEntity 
    FOREIGN KEY (source_entity_id) REFERENCES entity (entity_id)
    ''',
    
    '''
    ALTER TABLE relationship 
    ADD CONSTRAINT FK_TargetEntity 
    FOREIGN KEY (target_entity_id) REFERENCES entity (entity_id)
    ''',

    # 4. CREATE VECTOR INDEX for entity
    '''
    CREATE VECTOR INDEX EntityEmbeddingIndex ON entity (embedding)
    WHERE embedding IS NOT NULL
    OPTIONS (distance_type = 'COSINE')
    ''',
    
    # 5. CREATE VECTOR INDEX for relationship
    '''
    CREATE VECTOR INDEX RelationshipEmbeddingIndex ON relationship (embedding)
    WHERE embedding IS NOT NULL
    OPTIONS (distance_type = 'COSINE')
    '''
]


import os
from google.cloud import spanner

# --- Configuration ---
# IMPORTANT: Replace these placeholders with your actual Spanner instance and database IDs.
# Ensure you are authenticated (e.g., using 'gcloud auth application-default login')
PROJECT_ID = os.environ.get("GCLOUD_PROJECT") or "staging-470600"
INSTANCE_ID = "knowledge-graph"
DATABASE_ID = "kg"


def run_dml():
    """Initializes the Spanner client and runs the transaction."""

    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)
    operation = database.update_ddl(google_sql_ddl_statements)
    operation.result()


if __name__ == "__main__":
    run_dml()
