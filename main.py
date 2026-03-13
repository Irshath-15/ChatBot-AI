# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import httpx
import base64
import os

load_dotenv()

app = FastAPI(
    title="MENTORA AI API",
    description="AI Chat + Image Generation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

SYSTEM_PROMPT = """
You are MENTORA, a highly intelligent and friendly AI assistant.
You can help with absolutely anything including:
- Answering questions on any topic
- Writing emails, essays, stories, code
- Solving math and logic problems
- Giving advice and recommendations
- Summarizing and explaining complex topics
- Creative writing and brainstorming
- Translation and language help
- Career and personal guidance

Guidelines:
- Always be helpful, friendly and conversational
- Give clear, well structured answers
- Use examples where helpful
- Be honest when you don't know something
- Keep responses concise but complete
- Match the tone of the user
"""

# Models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ImageRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    reply: str
    conversation_history: list
    image_base64: str = None

# Image Generation
async def generate_image(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                HF_IMAGE_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": prompt},
            )
            if response.status_code == 200:
                image_bytes = response.content
                return base64.b64encode(image_bytes).decode("utf-8")
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {response.text}"
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Routes
@app.get("/")
def root():
    return {
        "status": "MENTORA Backend is running!",
        "version": "2.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        message = request.message.strip()

        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Image generation
        if message.lower().startswith("/image"):
            prompt = message[6:].strip()
            if not prompt:
                return ChatResponse(
                    reply="Please provide a prompt! Example: /image a cat in space",
                    conversation_history=request.conversation_history,
                )
            try:
                image_base64 = await generate_image(prompt)
                updated_history = request.conversation_history.copy()
                updated_history.append({
                    "role": "user",
                    "content": message
                })
                updated_history.append({
                    "role": "assistant",
                    "content": f"Generated image for: {prompt}"
                })
                return ChatResponse(
                    reply=f"Generated image for: {prompt}",
                    conversation_history=updated_history,
                    image_base64=image_base64,
                )
            except Exception as e:
                return ChatResponse(
                    reply=f"Image generation failed: {str(e)}",
                    conversation_history=request.conversation_history,
                )

        # Normal chat
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({
            "role": "user",
            "content": message
        })

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )

        reply = response.choices[0].message.content
        updated_history = request.conversation_history.copy()
        updated_history.append({"role": "user", "content": message})
        updated_history.append({"role": "assistant", "content": reply})

        return ChatResponse(
            reply=reply,
            conversation_history=updated_history,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image_endpoint(request: ImageRequest):
    image_base64 = await generate_image(request.prompt)
    return {"image_base64": image_base64}