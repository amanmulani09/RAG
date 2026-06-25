from fastapi import APIRouter

from app.api.chat import router as chat_router
from app.api.document import router as document_router

api_router = APIRouter(prefix="/api")

api_router.include_router(chat_router)
api_router.include_router(document_router)