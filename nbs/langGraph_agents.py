from ast import Pass
import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

client = OpenAI()

# define the agent class
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
    

# example prompt for the REACT agent
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

ACTION_1:
e.g. ACTION_1:
INSERT ACTION_1 HERE    

ACTION_2:
e.g. ACTION_2:
INSERT ACTION_2 HERE

Example session:

Question: INSERT QUESTION HERE
Thought: INSERT THOUGHT HERE
Action: ACTION_1: INSERT ACTION_1 HERE
PAUSE

You will be called again with this:

Observation: INSERT OBSERVATION HERE

You then output:

Answer: INSERT ANSWER HERE
""".strip()


def ACTION_1(INPUT1):
    pass

def ACTION_2(INPUT2):
    pass

known_actions = {
    "ACTION_1": ACTION_1,
    "ACTION_2": ACTION_2
}

#abot = Agent(prompt)
#result = abot("INSERT QUESTION HERE")
#next_prompt = "Observation: {}".format(ACTION_2("INPUT2"))
#abot(next_prompt)
#abot.messages

action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
        
question = """INSERT QUESTION HERE"""
query(question)


tool = TavilySearchResults(max_results=4) #increased number of results
#print(type(tool))
#print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]