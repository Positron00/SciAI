"""This is where users start to interact with the underlying LLM and agents."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['REPLICATE_API_TOKEN', 'llama3_8b', 'taskRouter', 'vote']

# %% ../nbs/00_core.ipynb 3
# install dependencies
# %pip install replicate
# %pip install gradio
# %pip install langchain

# %% ../nbs/00_core.ipynb 4
# import replicate
# import gradio
# import langchain

import os
from getpass import getpass

# Enter REPLICATE API TOKEN to run inference
REPLICATE_API_TOKEN = getpass(prompt="Enter REPLICATE API TOKEN: ")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# %% ../nbs/00_core.ipynb 5
import replicate

# function to run inference with Meta's Llama 3.8B model
def llama3_8b(prompt):
    output = replicate.run(
      "meta/meta-llama-3-8b-instruct",
      input={"prompt": prompt}
    )
    return ''.join(output)

# %% ../nbs/00_core.ipynb 6
# A chatbot that allows users to interact with an LLM
import gradio as gr

def taskRouter(message, history):
    if message.endswith("?"):
        return "Good question!"
    elif "Hello" in message:
        return "Hello! How can I help?"
    else:
        return "Let me work on that...."

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])

with gr.Blocks() as sciChat:
    chatbot = gr.Chatbot(height=300,placeholder="<strong>Your Personal Science AI Assistant</strong><br>Ask Me Anything")
    chatbot.like(vote, None, None)
    gr.ChatInterface(
        fn=taskRouter,
        type="messages",
        chatbot=chatbot,
        textbox=gr.Textbox(placeholder="How can I help?", container=False, scale=7),
        title="SciAI Assistant",
        description="Ask me any question or carry out a task",
        theme="soft",
        examples=["Hello", "What's new last week in neuroscience?", "I need help analyze some data"],
        cache_examples=True
    )
    
sciChat.launch(share=True)
