from fastapi import APIRouter
from app.models.document import DocumentRequest
router = APIRouter()

@router.post('/document')
async def upload_document(request:DocumentRequest):
    pass