"""
    dependencies module    
"""

from fastapi import Form, UploadFile, File

from models.schemas import MeterUploadData
from services.milvus_service import MilvusService
from services.bedrock_service import BedrockService

bedrock_service = BedrockService()
milvus_service = MilvusService()

def get_milvus_service() -> MilvusService:
    return milvus_service

def get_bedrock_service() -> BedrockService:
    return bedrock_service

async def create_upload_data(
        city: str = Form(...),
        street_name: str = Form(...),
        street_number: str = Form(...),
        file: UploadFile = File(...)
    ) -> MeterUploadData:
    """
        Create validated upload data model
    """

    content = await file.read()
    
    return MeterUploadData(
        city=city,
        street_name=street_name,
        street_number=street_number,
        file_content=content,
        file_name=file.filename or "",
        content_type=file.content_type or ""
    )