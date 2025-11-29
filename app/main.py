from floggit import flog
from get_relevant_neighborhood import main as get_relevant_neighborhood
from get_random_neighborhood import main as get_random_neighborhood
from knowledge_curation_agent.main import main as _curate_knowledge

from fastapi import FastAPI, BackgroundTasks, Body

app = FastAPI()


from pydantic import BaseModel

class CurateRequest(BaseModel):
    query: str
    user_id: str
    graph_id: str

@app.post('/curate_knowledge')
def curate_knowledge_route(
        data: CurateRequest,
        background_tasks: BackgroundTasks = None) -> dict:
    background_tasks.add_task(
            _curate_knowledge,
            graph_id=data.graph_id,
            user_id=data.user_id,
            query=data.query)
    return {'message': 'All set. Any new or updated knowledge is being curated.'}


@app.get('/random_neighborhood')
@flog
def random_neighborhood_route() -> dict:
    '''Returns a random neighborhood (entity plus neighbors) from the
    knowledge graph.'''
    return get_random_neighborhood()


@app.get("/search")
@flog
def search_route(query: str) -> dict:
    '''Returns a neighborhood (a set of entities plus their neighborhoods),
    relevant to the input query, from the knowledge graph.'''
    return get_relevant_neighborhood(query=query)


@app.get("/expand_query")
@flog
def expand_query_route(query: str) -> str:
    """Returns a paragraph that relates what is contained in the knowledge
    graph, relevant to the input query."""
    nbhd = get_relevant_neighborhood(query=query)

    relevant_entities_str = ""
    for entity in nbhd['entities'].values():
        entity_name = entity['entity_names'][0]
        entity_str = ''
        if entity.get('properties'):
            entity_str += f"{entity_name} has properties: {str(entity['properties'])}. "
        if len(entity['entity_names']) > 1:
            entity_str += f"{entity_name} is also known as: {', '.join(entity['entity_names'][1:])}"

        if entity_str:
            relevant_entities_str += entity_str + "\n"

    relationships_str = ""
    for rel in nbhd['relationships']:
        source_name = nbhd['entities'][rel['source_entity_id']]['entity_names'][0]
        target_name = nbhd['entities'][rel['target_entity_id']]['entity_names'][0]
        relationships_str += f"{source_name} {rel['relationship']} {target_name}\n"

    if relevant_entities_str or relationships_str:
        relevant_subgraph_str = f"(FYI, according to the Knowledge Graph: {relevant_entities_str}\n{relationships_str}.)"
    else:
        relevant_subgraph_str = ''

    return relevant_subgraph_str
