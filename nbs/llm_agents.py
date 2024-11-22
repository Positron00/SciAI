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


