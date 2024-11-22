# %pip install llama-stack-client==0.0.35 > /dev/null 2>&1
# %pip install termcolor

import os
import requests
import json

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


# Llama Guard for safety
def llamaguard3(prompt, debug=False):
  model = "meta-llama/Meta-Llama-Guard-3-8B"
  url = "https://api.together.xyz/v1/completions"
  payload = {
    "model": model,
    "temperature": 0,
    "prompt": prompt,
    "max_tokens": 4096,
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  if debug:
    print(res)
  return res['choices'][0]['text']


# run vision model
import base64

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import agent_create_params

LLAMA32_11B_INSTRUCT = "Llama3.2-11B-Vision-Instruct"

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

    async def execute_turn(self, prompt: str, image_path: str):
        base64_image = encode_image(image_path)

        messages = [{
            "role": "user",
            "content": [
              {
                "image": {
                  "uri": f"data:image/jpeg;base64,{base64_image}"
                }
              },
              prompt,
            ]
        }]

        response = self.client.agents.turns.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages = messages,
            stream=True,
        )

        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk

async def run_main(image_path, prompt):
    agent_config = AgentConfig(
        model=LLAMA32_11B_INSTRUCT,
        instructions="You are a helpful assistant",
        enable_session_persistence=False,
    )

    agent = Agent()
    agent.create_agent(agent_config)

    print(f"User> {prompt}")
    response = agent.execute_turn(prompt=prompt, image_path=image_path)
    async for log in EventLogger().log(response):
        if log is not None:
            log.print()

#await run_main("images/cat.jpeg",
#         "What cat breed is this? Tell me in detail about the breed.")
