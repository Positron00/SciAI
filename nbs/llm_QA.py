import os
import requests
import json
from together import Together

from openai import OpenAI
from dotenv import load_dotenv

# allow an option to select with llm to interact with
# preregister and get free API keys from the providers
# store the API keys in a .env file
provider = "TogetherAI"

load_dotenv('.env', override=True)

if provider == "OpenAI":
    pass
elif provider == "Meta":
    meta_api_key = os.getenv('META_API_KEY') # get the Meta API key from the .env file
    #client = Meta(api_key=meta_api_key) # set up the Meta client
elif provider == "Google":
    google_api_key = os.getenv('GOOGLE_API_KEY') # get the Google API key from the .env file
    #client = Google(api_key=google_api_key) # set up the Google client
elif provider == "Groq":
    groq_api_key = os.getenv('GROQ_API_TOKEN') # get the Groq API key from the .env file
    #client = Groq(api_key=groq_api_key) # set up the Groq client
elif provider == "TogetherAI":
    togetherai_api_key = os.getenv('TOGETHERAI_API_KEY') # get the TogetherAI API key from the .env file
    #client = TogetherAI(api_key=togetherai_api_key) # set up the TogetherAI client
elif provider == "Replicate":
    replicate_api_key = os.getenv('REPLICATE_API_TOKEN') # get the Replicate API key from the .env file
    #client = Replicate(api_key=replicate_api_key) # set up the Replicate client
elif provider == "HuggingFace":
    huggingface_api_key = os.getenv('HUGGINGFACE_API_TOKEN') # get the HuggingFace API key from the .env file
    #client = HuggingFace(api_key=huggingface_api_key) # set up the HuggingFace client

# create a llm class
class LLM:
    def __init__(self, model_family="meta-llama", provider="TogetherAI"):
        self.model_family = model_family
        self.provider = provider

    def print_llm_response(self, prompt):
        """This function takes as input a prompt, which must be a string enclosed in quotation marks,
        and passes it to the llm. The function then prints the response of the model.
        """
        llm_response = self.get_llm_response(prompt)
        print(llm_response)

    def get_llm_response(self, prompt):
        """This function takes as input a prompt, which must be a string enclosed in quotation marks,
        and passes it to the llm. The function then returns the response of the model as a string.
        """
        try:
            #if not isinstance(prompt, str):
            #    raise ValueError("Input must be a string enclosed in quotes.")
            
            if self.provider == "OpenAI":
                return self.openai_gpt35(prompt)
            elif self.provider == "TogetherAI":
                return self.llama32(prompt)
                            
        except TypeError as e:
            print("Error:", str(e))

    # use OpenAI's GPT3.5 model
    def openai_gpt35(self, prompt):
        self.model = "gpt-3.5-turbo-0125"
        openai_api_key = os.getenv('OPENAI_API_KEY') # get the OpenAI API key from the .env file
        client = OpenAI(api_key=openai_api_key) # set up the OpenAI client
        completion = client.chat.completions.create(
                    self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful but terse AI assistant who gets straight to the point.",
                        },
                        {"role": "user", 
                         "content": prompt},
                    ],
                    temperature=0.0,
                )
        return completion.choices[0].message.content
        
    # use TogetherAI's Llama3.2 model
    def llama32(self, messages, model_size=11):
        self.model = f"{self.model_family}/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
        url = "https://api.together.xyz/v1/chat/completions"
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0.0,
            "stop": ["<|eot_id|>","<|eom_id|>"],
            "messages": messages
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + togetherai_api_key
        }
        result = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

        if 'error' in result:
            raise Exception(result['error'])

        return result['choices'][0]['message']['content']


from PIL import Image
import matplotlib.pyplot as plt

def display_local_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

import base64

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')

def resize_image(img):
    original_width, original_height = img.size

    if original_width > original_height:
        scaling_factor = max_dimension / original_width     
    else:
        scaling_factor = max_dimension / original_height
        
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image while maintaining aspect ratio
    resized_img = img.resize((new_width, new_height))

    resized_img.save("images/resized_image.jpg")

    print("Original size:", original_width, "x", original_height)
    print("New size:", new_width, "x", new_height)

    return resized_img

"""
# handling image inputs
 
if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "describe the image!"
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image('images/test.jpg')}"
                    }
                }
            ]
        }
    ]

    result = llama32(messages)
    print(result)


# an example of tool calling
display_local_image("images/thumbnail_IMG_6385.jpg")
img = Image.open("images/thumbnail_IMG_6385.jpg")
max_dimension = 1120
resized_img = resize_image(img)
base64_image = encode_image("images/resized_image.jpg")

messages = [
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "Recommend the best place in USA I can have similar experience, put event name, its city and month in JSON format output only"
      },
      {
        "type": "image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{base64_image}"
        }
      }
    ]
  },
]
result = llama32(messages)

messages = [
    {
      "role": "system",
      "content":  f"""
"""Environment: ipython
        Tools: brave_search, wolfram_alpha
        Cutting Knowledge Date: December 2023
"""
"""
      },
    {
      "role": "assistant",
      "content": result
    },
    {
      "role": "user",
      "content": "Search to validate if the event {name} takes place in the {city} value in {month} value, replace {name}, {city} and {month} with values from {result}"
    }
  ]
llm_result = llama32(messages, 90)

import re

def parse_llm_result(llm_result: str):
    # Define the regular expression pattern to extract function name and arguments
    pattern = r"\<\|python\_tag\|\>(\w+\.\w+)\((.+)\)"

    match = re.search(pattern, llm_result)
    if match:
        function_call = match.group(1)  # e.g., brave_search.call
        arguments = match.group(2)      # e.g., query="current weather in New York City"
       
        # Further parsing the arguments to extract key-value pairs
        arg_pattern = r'(\w+)="([^"]+)"'
        arg_matches = re.findall(arg_pattern, arguments)

        # Convert the arguments into a dictionary
        parsed_args = {key: value for key, value in arg_matches}

        return {
            "function_call": function_call,
            "arguments": parsed_args
        }
    else:
        return None


parsed_result = parse_llm_result(llm_result)
#print(parsed_result)

# calling the search API
import sys
#!{sys.executable} -m pip install tavily-python

from tavily import TavilyClient

os.environ["TAVILY_API_KEY "]= ""
TAVILY_API_KEY = os.environ["TAVILY_API_KEY "]
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

result = tavily_client.search(parsed_result['arguments']['query'])
search_result = result["results"][0]["content"]
#search_result

# reprompt Llama with the search result
messages.append({
                    "role": "tool",
                    "content": search_result,
                })

response = llama32(messages)
print(response)

"""