# %pip install llama-stack-client==0.0.35 > /dev/null 2>&1
# %pip install termcolor

LLAMA_STACK_API_TOGETHER_URL="https://llama-stack.together.ai"
LLAMA31_8B_INSTRUCT = "Llama3.1-8B-Instruct"

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage

async def run_main():
    client = LlamaStackClient(
        base_url=LLAMA_STACK_API_TOGETHER_URL,
        #base_url=LLAMA_STACK_API_LOCAL_URL,
    )

    iterator = client.inference.chat_completion(
        messages=[
            UserMessage(
                content="What is the world's largest living structure, according to the Guinness World Records?",
                role="user",
            ),

            UserMessage(
                content="How large is it?",
                role="user",
            ),
        ],
        model=LLAMA31_8B_INSTRUCT,
        stream=True
    )

    async for log in EventLogger().log(iterator):
        log.print()

#await run_main()


import asyncio
from typing import List, Optional, Dict

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger

from llama_stack_client.types import SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig

class Agent:
    def __init__(self):
        self.client = LlamaStackClient(
            base_url=LLAMA_STACK_API_TOGETHER_URL,
        )

    def create_agent(self, agent_config: AgentConfig):
        agent = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = agent.agent_id
        session = self.client.agents.sessions.create(
            agent_id=agent.agent_id,
            session_name="example_session",
        )
        self.session_id = session.session_id

    async def execute_turn(self, content: str):
        response = self.client.agents.turns.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=True,
        )

        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk

async def run_main():
    agent_config = AgentConfig(
        model=LLAMA31_8B_INSTRUCT,
        instructions="You are a helpful assistant",
        enable_session_persistence=False,
    )

    agent = Agent()
    agent.create_agent(agent_config)

    prompts = [
        "What is the world's largest living structure, according to the Guinness World Records?",
        "How large is it?",
    ]

    for prompt in prompts:
        print(f"User> {prompt}")
        response = agent.execute_turn(content=prompt)
        async for log in EventLogger().log(response):
            if log is not None:
                log.print()

#await run_main()