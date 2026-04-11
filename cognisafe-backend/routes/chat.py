import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/chat", tags=["chat"])

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


class Message(BaseModel):
    role: str      # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages:      List[Message]
    system_prompt: str


@router.post("")
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": req.system_prompt},
            *[{"role": m.role, "content": m.content} for m in req.messages],
        ],
        "max_tokens":   800,
        "temperature":  0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json=payload,
            )
            res.raise_for_status()
            data = res.json()
            reply = data["choices"][0]["message"]["content"]
            return {"reply": reply}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Groq error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))