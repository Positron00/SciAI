import os
from dotenv import load_dotenv
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
load_dotenv('.env', override=True)

# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying an ML Model with FastAPI')

# List available models using Enum for convenience. This is useful when the options are pre-defined.
class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://serve/docs"


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
def predict(model: Model, file: UploadFile = File(...)):
    pass

if __name__ == "__main__":
    # Allows the server to be run in this interactive environment
    nest_asyncio.apply()

    # This is an alias for localhost which means this particular machine
    host = "127.0.0.1"

    # Spin up the server!    
    uvicorn.run(app, host=host, port=8000, root_path="/serve")