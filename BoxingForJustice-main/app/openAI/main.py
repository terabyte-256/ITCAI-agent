from pathlib import Path
import os

import requests


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from openai import OpenAI

load_dotenv()



BASE_DIR = Path(__file__).resolve().parent.parent.parent

#choose which model to use, either local ollama or openai api
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:8050")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


app = FastAPI(title="Campus Knowledge Agent", version="1.0.0")

static_dir = BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create OpenAI client only if key exists
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "provider": LLM_PROVIDER,
        "ollama_host": OLLAMA_HOST,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
    }


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")

@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    prompt = request.message
    if not prompt:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if LLM_PROVIDER == "openai":
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        try:
            response = openai_client.chat.responses.create(
                model=OPENAI_MODEL,
                input = prompt,
            )
            return {
                    "provider": "openai",
                    "model": OPENAI_MODEL,
                    "response": response.output_text
                    }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    elif LLM_PROVIDER == "ollama":
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
        }

        # Parse and print the result
        try:
            response = requests.post(f"{OLLAMA_HOST}/generate", json=payload)
            response.raise_for_status()
            return {
                "provider": "ollama",
                "model": OLLAMA_MODEL,
                "response": response.text
            } # response.text is the plain text from the stream
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
            
            
            
    else:
        raise HTTPException(
            status_code=500,
            detail="Invalid LLM_PROVIDER. Use 'ollama' or 'openai'."
        )





if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8060, reload=True)