# run_test.py

import asyncio

from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

from google.adk import Context
from google.adk import Workflow
from typing import Any

from google.adk import Agent
from google.adk import Context
from google.adk.workflow import node
from pydantic import BaseModel
from pathlib import Path
import os 
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
load_dotenv(env_path)
cred_path = ROOT / os.getenv("SERVICE_ACCOUNT_KEY")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)


class CityTime(BaseModel):
    time_info: str  # time information
    city: str       # city name

@node(rerun_on_resume=True)
def city_time_function(node_input):
    #print(type(node_input))
    #print(node_input)
    """Simulate returning the current time in a specified city."""
    return CityTime(time_info="10:10 AM", city=node_input["city"])

city_report_agent = Agent(
    name="city_report_agent",
    model="gemini-flash-latest",
    input_schema=CityTime,
    instruction="""output the data provided by the previous node.""",
)

@node(rerun_on_resume=True)
async def city_workflow(ctx: Context):
    city_time = await ctx.run_node(city_time_function, {"city": "Paris"} )
    print(type(city_time),city_time)
    report_text = await ctx.run_node(city_report_agent, city_time)

    return report_text
# Run the workflow
root_agent = Workflow(
    name="root_agent",
    edges=[("START", city_workflow)],
)

async def main():

    app = App(
        name="dynamic_workflow_test",
        root_agent=root_agent,
    )

    runner = InMemoryRunner(app=app)

    await runner.session_service.create_session(
        app_name=app.name,
        user_id="test_user",
        session_id="s1",
    )

    msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="hello dynamic workflow"
            )
        ],
    )

    last_event = None

    async for event in runner.run_async(
        user_id="test_user",
        session_id="s1",
        new_message=msg,
    ):
        last_event = event

if __name__ == "__main__":
    asyncio.run(main())