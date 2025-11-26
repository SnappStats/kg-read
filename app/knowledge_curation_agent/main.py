import os
from dotenv import load_dotenv
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.genai import types

from .agent import agent

# Load environment variables from .env file in root directory
load_dotenv()

AGENT_ENGINE_ID = os.environ['SESSION_SERVICE_URI'].split('/')[-1]

session_service = VertexAiSessionService(
    agent_engine_id=AGENT_ENGINE_ID,
)

_agent_runner: Optional[Runner] = None

def get_agent_runner() -> Runner:
    """Lazily initializes and returns the agent_runner."""
    global _agent_runner
    if _agent_runner is None:
        # This code will only run on the FIRST call to main() after startup,
        # and will run within the background task, freeing up the main thread.
        _agent_runner = Runner(
            agent=agent,
            app_name=AGENT_ENGINE_ID,
            session_service=session_service
        )
    return _agent_runner


async def main(graph_id: str, user_id: str, query: str):
    agent_runner = get_agent_runner()

    session = await session_service.create_session(
            app_name=AGENT_ENGINE_ID,
            user_id=user_id,
            state={'graph_id': graph_id})

    user_content = types.Content(role='user', parts=[types.Part(text=query)])
    qwer = agent_runner.run_async(
            user_id=user_id, session_id=session.id, new_message=user_content)

    # Need this line.... Is there a good replacement?
    async for event in qwer:
        pass
