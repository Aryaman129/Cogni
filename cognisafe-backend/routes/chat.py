import logging
from typing import Optional
from fastapi import APIRouter, Depends, status
from models import User
from dependencies import get_current_user
from services.nebius_client import get_nebius_client
from schemas.ai import ChatRequest, ChatResponse
from .ai import router as ai_router

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Mount AI routes under /api/ai
router.include_router(ai_router)


@router.post("/api/chat", response_model=ChatResponse)
@router.post("/api/chat/", response_model=ChatResponse, include_in_schema=False)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
):
    """
    CogniSafe AI Chat powered by NVIDIA Nemotron 3.5 Lightning via Nebius Token Factory.
    Enforces strict clinical system guardrails and JWT authentication.
    Gracefully degrades to deterministic guidance if Nebius is unconfigured or unavailable.
    """
    client = get_nebius_client()
    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]

    response = await client.chat(
        messages=messages_payload,
        system_prompt=req.system_prompt,
        context_biomarkers=req.context_biomarkers,
    )
    return response