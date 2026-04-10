from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama # Ensure you have the latest version: pip install ollama
#import asyncio

app = FastAPI(title="Ollama FastAPI Server")

class ChatRequest(BaseModel):
    model: str = "llama3.1"
    prompt: str

@app.post("/generate")
async def generate_response(request: ChatRequest):
    async def stream_generator():
        try:
            # Use AsyncClient for non-blocking calls
            #client = ollama.AsyncClient()
            for chunk in ollama.generate(
                model=request.model, 
                prompt=request.prompt, 
                stream=True
            ):
                yield chunk['response']
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
