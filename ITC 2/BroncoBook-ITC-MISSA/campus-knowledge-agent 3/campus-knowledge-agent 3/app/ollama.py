from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI(title="Ollama FastAPI Server")

# Define the request structure
class ChatRequest(BaseModel):
    model: str = "llama3"
    prompt: str

@app.get("/")
async def root():
    return {"message": "Ollama-FastAPI server is running!"}

@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # Call the local Ollama service
        response = ollama.generate(
            model=request.model,
            prompt=request.prompt
        )
        return {"response": response['response']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
