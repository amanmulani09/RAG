from fastapi import APIRouter
from app.models.chat import ChatRequest
router = APIRouter()

@router.post("/chat")
async def chat(request:ChatRequest):
    pass