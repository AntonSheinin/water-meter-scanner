"""
    API endpoints 
"""

import os
import uuid
import time
import logging
from datetime import datetime

from fastapi import HTTPException, APIRouter, Depends
from fastapi.responses import HTMLResponse, FileResponse

from api.utils import convert_milvus_result
from services.search_service import search_by_address, search_by_context, search_similar_readings, search_by_recency
from api.dependencies import create_upload_data, get_bedrock_service, get_milvus_service
from models.schemas import UploadResponse, ChatQuery, ChatResponse, MeterUploadData


router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """
        Serve the main web interface
    """

    static_file_path = os.path.join("static", "index.html")

    return FileResponse(static_file_path) if os.path.exists(static_file_path) else HTMLResponse(
        """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Water Meter Scanner</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    h1 { color: #333; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Water Meter Scanner</h1>
                    <p>AI-powered water meter reading extraction and analysis</p>
                    <h3>Available:</h3>
                    <ul>
                        <li><a href="/docs">API Documentation</a></li>
                    </ul>
                </div>
            </body>
            </html>
        """
    )
    
@router.get("/health")
async def health_check(
        milvus_service = Depends(get_milvus_service),
        bedrock_service = Depends(get_bedrock_service)
    ) -> dict:
    """
        Health check endpoint
    """

    return {
        "status": "healthy",
        "milvus": milvus_service.health_check(),
        "bedrock": bedrock_service.health_check()
    }

@router.get("/milvus-info")
async def milvus_info(
        milvus_service = Depends(get_milvus_service),
    ) -> dict:
    """
        Get Milvus collection information
    """

    info = milvus_service.get_collection_info()

    return info if info else HTTPException(status_code=503, detail="Milvus not initialized")
    
@router.post("/upload-meter", response_model=UploadResponse)
async def upload_meter_reading(
        upload_data: MeterUploadData = Depends(create_upload_data),
        milvus_service = Depends(get_milvus_service),
        bedrock_service = Depends(get_bedrock_service)
    ) -> UploadResponse:
    """
        Upload water meter image and extract reading using vision model
    """
    
    address_info = {
        "city": upload_data.city,
        "street_name": upload_data.street_name,
        "street_number": upload_data.street_number
    }

    if not bedrock_service.connected:
            raise HTTPException(status_code=503, detail="Bedrock service not available")
        
    logger.info(f"Processing meter image upload for {address_info}")

    try:        
        # Analyze image using Vision LLM
        vision_result = await bedrock_service.analyze_meter_image(upload_data.file_content, address_info)

    except Exception as exc:
        logger.error(f"❌ Bedrock vision analysis failed: {exc}")
        raise HTTPException(status_code=500, detail="Vision analysis failed")
        
    # Check if analysis was successful
    if not vision_result or not isinstance(vision_result, dict):
        raise HTTPException(status_code=500, detail="Vision analysis failed to return valid results")

    # Check confidence
    if not vision_result.get("reading_visible", False) and vision_result.get("confidence", 0) < 0.3:
        logger.warning(f"Low confidence reading: {vision_result}")
    
    # Generate unique ID for this reading
    reading_id = f"meter_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    # Create full address string
    full_address = f"{upload_data.street_number} {upload_data.street_name}, {upload_data.city}"
    
    try:
        embeddings = await bedrock_service.generate_meter_embeddings(
            address_info, 
            vision_result["meter_value"], 
            vision_result.get("units", "cubic_meters")
        )
    except:
        logger.error(f"❌ Embedding generation failed: {exc}")
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    try:
        # Store in Milvus
        timestamp = int(time.time())
        stored = await milvus_service.store_meter_reading(
            reading_id,
            address_info,
            vision_result["meter_value"],
            vision_result["confidence"],
            embeddings,
            timestamp,
            vision_result.get("units", "cubic_meters"),
            vision_result.get("meter_type", "unknown")
        )

        if not stored:
            logger.warning("Failed to store reading in Milvus, but extraction succeeded")
    
    except Exception as exc:
        logger.error(f"❌ Failed to store reading in Milvus: {exc}")  
        raise HTTPException(status_code=500, detail=f"❌ Failed to store reading in Milvus: {exc}")   

    logger.info(f"✅ Successfully extracted meter reading: {vision_result['meter_value']} at {full_address}")
        
    # Return successful response
    return UploadResponse(
        success=True,
        reading_id=reading_id,
        meter_value=vision_result["meter_value"],
        confidence=vision_result["confidence"],
        address=full_address,
        timestamp=datetime.now(),
        notes=vision_result.get("notes", "")
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(
        query: ChatQuery,
        milvus_service = Depends(get_milvus_service),
        bedrock_service = Depends(get_bedrock_service)
    ) -> ChatResponse:
    """
        Chat with meter data using semantic search
    """

    user_query = query.message.strip()
        
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Check if services are available
    if not bedrock_service.connected:
        raise HTTPException(status_code=503, detail="bedrock service not available")
    
    if not milvus_service.collection:
        raise HTTPException(status_code=503, detail="milvus service not available")
    
    # Determine search strategy based on query
    context_results = []
    user_query_lower = user_query.lower()

    # Check for time-based queries
    if any(word in user_query_lower for word in ["last", "latest", "recent", "newest", "most recent"]):
        context_results = await search_by_recency(milvus_service, user_query, limit=5)
        logger.info(f"Using recency search for query: {user_query}")
        
    # Check for usage-related queries
    elif any(word in user_query_lower for word in ["usage", "high", "low", "consumption", "similar", "pattern", "highest", "lowest"]):
        context_results = await search_by_context(milvus_service, bedrock_service, user_query, limit=5)
        logger.info(f"Using context search for query: {user_query}")
        
    # Check for location-based queries
    elif any(word in user_query_lower for word in ["address", "street", "city", "location", "where", "at"]):
        context_results = await search_by_address(milvus_service, bedrock_service, user_query, limit=5)
        logger.info(f"Using address search for query: {user_query}")
        
    # Default: Use general similarity search from bedrock_service
    else:
        context_results = await search_similar_readings(
            milvus_service, 
            bedrock_service,
            user_query, 
            search_type="combined",  # Search combined embeddings for general queries
            limit=5
        )
        logger.info(f"Using general similarity search for query: {user_query}")
    
    # If no results found, try broader search
    if not context_results:
        logger.warning(f"No results found for query '{user_query}', trying broader search")
        context_results = await search_similar_readings(
            milvus_service,
            bedrock_service,
            user_query, 
            search_type="address",  # Try address-only search as fallback
            limit=10
        )
    
    # If still no results, get recent readings as context
    if not context_results:
        logger.warning("No similar results found, using recent readings as context")
        context_results = await search_by_recency(milvus_service, "", limit=5)
    
    # Log search results for debugging
    logger.info(f"Found {len(context_results)} context results for chat query")
    for i, result in enumerate(context_results[:3]):  # Log first 3 results
        logger.info(f"Result {i+1}: {result.get('full_address')} - {result.get('meter_value')} (similarity: {result.get('similarity_score', 'N/A')})")
    
    
    # Generate response using Bedrock
    response_text = await bedrock_service.generate_chat_response(user_query, context_results)
    
    return ChatResponse(
        response=response_text,
        sources_count=len(context_results)
    )

@router.get("/readings")
async def get_readings_with_vectors(
        limit: int = 20, 
        milvus_service = Depends(get_milvus_service),
    ) -> dict:
    """
        Get readings including vector field info
    """

    if not milvus_service.collection:
            raise HTTPException(status_code=503, detail="Milvus not available")

    try:
        milvus_service.collection.load()
        
        results = milvus_service.collection.query(
            expr="id != ''",
            output_fields=["id", "meter_value", "full_address", "confidence", "address_embedding", "combined_embedding"],
            limit=limit
        )
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
    serializable_results = []

    for result in results:
        converted = convert_milvus_result(result)
        # Add vector info without showing full arrays
        converted["address_embedding_length"] = len(converted.get("address_embedding", []))
        converted["combined_embedding_length"] = len(converted.get("combined_embedding", []))
        converted["address_embedding_sample"] = converted.get("address_embedding", [])[:5]
        converted["combined_embedding_sample"] = converted.get("combined_embedding", [])[:5]
        # Remove full vectors to keep response small
        converted.pop("address_embedding", None)
        converted.pop("combined_embedding", None)
        serializable_results.append(converted)
    
    return {
        "readings": serializable_results, 
        "count": len(serializable_results)
    }
        
    


    