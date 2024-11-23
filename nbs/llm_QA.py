import os
import requests
import json

from openai import OpenAI
from dotenv import load_dotenv

# need to allow an option to select with llm to interact with
which_provider = "OpenAI"

load_dotenv('.env', override=True)

if which_provider == "OpenAI":
    openai_api_key = os.getenv('OPENAI_API_KEY') # get the OpenAI API key from the .env file
    client = OpenAI(api_key=openai_api_key) # set up the OpenAI client
elif which_provider == "Meta":
    meta_api_key = os.getenv('META_API_KEY') # get the Meta API key from the .env file
    #client = Meta(api_key=meta_api_key) # set up the Meta client
elif which_provider == "Google":
    google_api_key = os.getenv('GOOGLE_API_KEY') # get the Google API key from the .env file
    #client = Google(api_key=google_api_key) # set up the Google client
elif which_provider == "Groq":
    groq_api_key = os.getenv('GROQ_API_TOKEN') # get the Groq API key from the .env file
    #client = Groq(api_key=groq_api_key) # set up the Groq client
elif which_provider == "TogetherAI":
    togetherai_api_key = os.getenv('TOGETHERAI_API_KEY') # get the TogetherAI API key from the .env file
    #client = TogetherAI(api_key=togetherai_api_key) # set up the TogetherAI client
elif which_provider == "Replicate":
    replicate_api_key = os.getenv('REPLICATE_API_TOKEN') # get the Replicate API key from the .env file
    #client = Replicate(api_key=replicate_api_key) # set up the Replicate client
elif which_provider == "HuggingFace":
    huggingface_api_key = os.getenv('HUGGINGFACE_API_TOKEN') # get the HuggingFace API key from the .env file
    #client = HuggingFace(api_key=huggingface_api_key) # set up the HuggingFace client

model="gpt-3.5-turbo-0125"

# create a llm class
class LLM:
    def __init__(self, model):
        self.model = model

    def print_llm_response(self, prompt):
        """This function takes as input a prompt, which must be a string enclosed in quotation marks,
        and passes it to OpenAI's GPT3.5 model. The function then prints the response of the model.
        """
        llm_response = self.get_llm_response(prompt)
        print(llm_response)

    def get_llm_response(self, prompt):
        """This function takes as input a prompt, which must be a string enclosed in quotation marks,
        and passes it to OpenAI's GPT3.5 model. The function then saves the response of the model as
        a string.
        """
        try:
            if not isinstance(prompt, str):
                raise ValueError("Input must be a string enclosed in quotes.")
            completion = client.chat.completions.create(
                self.model,
                messages=[
                    {
                    "role": "system",
                        "content": "You are a helpful but terse AI assistant who gets straight to the point.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            response = completion.choices[0].message.content
            return response
        except TypeError as e:
            print("Error:", str(e))

# an example function for using Llama3.2, to be integrated into the LLM class
def llama32(messages, model_size=11):
  model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
  url = "https://api.together.xyz/v1/chat/completions"
  payload = {
    "model": model,
    "max_tokens": 4096,
    "temperature": 0.0,
    "stop": ["<|eot_id|>","<|eom_id|>"],
    "messages": messages
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  return res['choices'][0]['message']['content']


# handling image inputs
from PIL import Image
import matplotlib.pyplot as plt
import base64

def resize_image(img, max_dimension=1120):
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
    

def encode_image(image_path):
  with open(image_path, "rb") as img:
    return base64.b64encode(img.read()).decode('utf-8')
  
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
