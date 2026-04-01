from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
import uuid
import ollama

#a simple FastAPI app that demonstrates how to use the Ollama client to generate responses from a language model. 
# It includes endpoints for health checks, generating responses, and streaming responses. 
# It also implements a simple in-memory cache for question-answer pairs.

app = FastAPI()

# In-memory store for conversation history and cache
conversation_store = {}
cache={}

# Pydantic model for the request body of the /generate endpoint
class GenerateRequest(BaseModel):
    session_id: str
    prompt: str
    question: str

# Health check endpoint to verify that the API is running
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message" : "API is running"
        }

# Simple ping endpoint to test latency and connectivity
@app.get("/ping")
def ping():
    start_time = time.time()
    response = {
        "request_id": str(uuid.uuid4()),
        "message": "pong!",
        "latency_ms": round((time.time() - start_time) * 1000, 2)
    }
    return response


# Simple endpoint to return a greeting message
@app.get("/hello")
def hello():
    return {
        "message": "Hello, World!"
    }

# Endpoint to generate a response from the language model based on the user's prompt and conversation history. 
# It also updates the conversation history with the user's prompt and the assistant's response.

@app.post("/generate/")
def generate(request: GenerateRequest):
    
    if request.session_id not in conversation_store:
        conversation_store[request.session_id] = []
    
    history = conversation_store[request.session_id]
    
    history.append({
        "role": "user",
        "content": request.prompt
        })
    
    start_time = time.time()

    response = ollama.chat(
        model="mistral:latest",
        messages=history
    )
    
    reply = response["message"]["content"]
    
    history.append({
        "role": "assistant",
        "content": reply
        })
    
    return {
        
        "session_id": request.session_id,
        "response": reply
    }
    
# Generator function to stream the response from the language model as it is generated.    
def stream_llm_response(prompt: str):
    stream = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        yield chunk["message"]["content"]
    
# Endpoint to stream the response from the language model as it is generated.    
@app.post("/generate-stream/")
def generate_stream(request: GenerateRequest):
    return StreamingResponse(
        stream_llm_response(request.prompt),
        media_type="text/plain"
        )
    
@app.get("/ask/")
def ask(question: str):
    
    #check cache first
    if question in cache:
        return {
            "response": cache[question],
            "cached": True
        }

    # If not cached, generate a new answer
    response = ollama.chat(
        model="mistral:latest",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    
    
    # Save to cache
    cache[question] = response

    answer = response["message"]["content"]

    # Store the answer in cache
    cache[question] = answer

    return {
        "response": answer,
        "cached": False
    }
