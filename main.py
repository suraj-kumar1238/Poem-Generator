from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline, set_seed
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI()

# Enable CORS (for frontend JS fetch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI model
print("Loading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2")
set_seed(42)
print("Model loaded successfully!")

# Setup template directory
templates = Jinja2Templates(directory="templates")

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint
@app.post("/generate")
async def generate_poem(data: dict):
    topic = data.get("topic", "nature")
    style = data.get("style", "Default")

    # Style prompt
    prompt = {
        "Haiku": f"Write a haiku about {topic}:\n",
        "Romantic": f"Write a romantic poem about {topic}:\n",
        "Sad": f"Write a sad and emotional poem about {topic}:\n",
        "Shakespearean": f"Write a Shakespearean poem about {topic}:\n",
    }.get(style, f"Write a poem about {topic}:\n")

    # Start timer for model response time
    start_time = time.time()
    
    result = generator(prompt, max_length=80, num_return_sequences=1)[0]['generated_text']
    
    # Log the response time
    print(f"Model response time: {time.time() - start_time} seconds")
    
    print(f"Generated text: {result.strip()}")
    return JSONResponse({"poem": result.strip()})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)  # Ensure the port is 7860
