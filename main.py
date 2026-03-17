# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
import httpx
import base64
import os

load_dotenv()

# Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_IMAGE_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """
You are MENTORA, a highly intelligent and friendly AI assistant with access to real-time web search.
You have up-to-date information about current events, news, and latest developments.
You can help with absolutely anything including:
- Answering questions on any topic with latest information
- Writing emails, essays, stories, code
- Solving math and logic problems
- Giving advice and recommendations
- Summarizing and explaining complex topics
- Analyzing images, documents and files
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
- When you have web search results, use them to give updated accurate information
"""

app = FastAPI(
    title="MENTORA AI API",
    description="AI Chat + Image Generation + File Analysis + Web Search",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    conversation_history: list
    image_base64: str = None

class FileAnalysisRequest(BaseModel):
    file_content: str = ""
    file_type: str
    file_name: str
    user_message: str = ""
    conversation_history: list = []
    image_base64: str = ""

# Web Search
async def search_web(query: str) -> str:
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )
        results = response.get("results", [])
        if not results:
            return ""
        summary = ""
        for r in results[:3]:
            summary += f"- {r['title']}: {r['content'][:200]}\n"
        return summary
    except Exception:
        return ""

# Image Generation
async def generate_image(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            HF_IMAGE_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
        )
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            raise Exception(f"Image generation failed: {response.text}")

# Routes
@app.get("/")
def root():
    return {
        "status": "MENTORA Backend is running!",
        "version": "4.0.0"
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
                updated_history.append({"role": "user", "content": message})
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

        # Real-time keywords that trigger web search
        real_time_keywords = [
            "latest", "current", "today", "now", "recent", "news",
            "update", "2024", "2025", "2026", "who is", "what is",
            "price", "score", "weather", "stock", "live", "trending",
            "new", "release", "launch", "announce", "happen"
        ]

        # Web search if needed
        web_context = ""
        if any(keyword in message.lower() for keyword in real_time_keywords):
            web_context = await search_web(message)

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if web_context:
            messages.append({
                "role": "system",
                "content": f"Real-time web search results for this query:\n{web_context}\nUse this information to give an accurate and updated answer."
            })

        for msg in request.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({"role": "user", "content": message})

        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
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


@app.post("/analyze-file")
async def analyze_file(request: FileAnalysisRequest):
    try:
        user_msg = request.user_message if request.user_message else "Please analyze this file"

        # Image analysis using vision model
        if request.file_type == "image" and request.image_base64:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_msg
                        }
                    ]
                }
            ]

            response = groq_client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                max_tokens=500
            )

        # PDF or text analysis
        else:
            content_preview = request.file_content[:3000]
            prompt = f"""The user has shared a file named '{request.file_name}'.

File content:
{content_preview}

User request: {user_msg}

Please analyze and respond helpfully."""

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in request.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            messages.append({"role": "user", "content": prompt})

            response = groq_client.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                max_tokens=500
            )

        reply = response.choices[0].message.content
        updated_history = request.conversation_history.copy()
        updated_history.append({
            "role": "user",
            "content": f"[File: {request.file_name}] {user_msg}"
        })
        updated_history.append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "conversation_history": updated_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-image")
async def generate_image_endpoint(request: dict):
    prompt = request.get("prompt", "")
    image_base64 = await generate_image(prompt)
    return {"image_base64": image_base64}