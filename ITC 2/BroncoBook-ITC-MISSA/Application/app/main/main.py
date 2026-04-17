import os
from pathlib import Path

import requests
import json

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
# from ollama import Client # Use the Client class to specify the custom port

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ... rest of your static and health routes ...
static_dir = BASE_DIR / "app" / "static"


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:8050")




app = FastAPI(title="Campus Knowledge Agent", version="1.0.0")

     

app.mount("/static", StaticFiles(directory=static_dir), name="static")


# client = Client(host='http://localhost:8050')


# 2. Define the request schema
class ChatRequest(BaseModel):
    model: str = "llama3"
    prompt: str

   
@app.get("/health")
def health() -> dict:
    return {"ok": True, "ollama_host": OLLAMA_HOST}



@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/chat/ollama")
async def chat_ollama(request: ChatRequest):
    
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False  # Set to False to get a single complete response
    }

    try:
        # Send the POST request
        response = requests.post(OLLAMA_HOST, json=payload)
        response.raise_for_status()
        
        # Parse and print the result
        result = response.json()
        return {"response": result['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        




if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
