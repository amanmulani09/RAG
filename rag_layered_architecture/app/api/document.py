from fastapi import APIRouter

router = APIRouter()

@router.post('/document')
async def upload_document():
    pass