from services.milvus_service import MilvusService
from services.bedrock_service import BedrockService

milvus_service = MilvusService()
bedrock_service = BedrockService()

def get_milvus_service() -> MilvusService:
    return milvus_service

def get_bedrock_service() -> BedrockService:
    return bedrock_service