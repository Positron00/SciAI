import os

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



def print_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then prints the response of the model.
    """
    llm_response = get_llm_response(prompt)
    print(llm_response)


def get_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then saves the response of the model as
    a string.
    """
    try:
        if not isinstance(prompt, str):
            raise ValueError("Input must be a string enclosed in quotes.")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
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

