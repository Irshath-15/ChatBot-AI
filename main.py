from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

app = FastAPI(
    title="IT Fresher Chatbot API",
    description="AI-powered IT tutor for beginners",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a friendly and patient IT tutor designed for freshers 
and beginners who are just starting their IT journey.
- Explain IT concepts in very simple, easy-to-understand English
- Always use real-life examples and analogies
- Be encouraging and supportive
- Cover: Networking, OS, Hardware, Software, Cybersecurity, Cloud Computing
- If question is not IT related, politely redirect the user
"""

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    conversation_history: list

@app.get("/")
def root():
    return {
        "status": "✅ IT Chatbot Backend is running!",
        "version": "1.0.0",
        "model": "llama3-8b-8192"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Build messages list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in request.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({
            "role": "user",
            "content": request.message
        })

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500
        )

        reply = response.choices[0].message.content

        # Update history
        updated_history = request.conversation_history.copy()
        updated_history.append({"role": "user", "content": request.message})
        updated_history.append({"role": "assistant", "content": reply})

        return ChatResponse(
            reply=reply,
            conversation_history=updated_history
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}