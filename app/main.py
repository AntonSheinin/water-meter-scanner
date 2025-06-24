from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from services.milvus_service import MilvusService
from services.bedrock_service import BedrockService
from models.schemas import UploadResponse, ChatQuery, ChatResponse
import os
import logging
from datetime import datetime
import uuid
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Water Meter Scanner",
    description="A web application for scanning home water usage counters"
)

# Initialize Milvus service
milvus_service = MilvusService()
bedrock_service = BedrockService()

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

def convert_milvus_result(result):
    """Convert Milvus result to JSON-serializable format"""
    converted = {}
    for key, value in result.items():
        if hasattr(value, 'item'):  # numpy scalar
            converted[key] = value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            converted[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Handle lists that might contain numpy objects
            converted[key] = [v.item() if hasattr(v, 'item') else v for v in value]
        else:
            converted[key] = value
    return converted

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    static_file_path = os.path.join("static", "index.html")
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    else:
        # Simple fallback
        return HTMLResponse("""
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
    """)
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check Milvus and Bedrock health
    milvus_health = milvus_service.health_check()
    bedrock_health = bedrock_service.health_check()

    return {
        "status": "healthy",
        "milvus": milvus_health,
        "bedrock": bedrock_health
    }

@app.get("/milvus/info")
async def milvus_info():
    """Get Milvus collection information"""
    info = milvus_service.get_collection_info()
    if info:
        return info
    else:
        raise HTTPException(status_code=503, detail="Milvus not initialized")
    
@app.post("/upload-meter", response_model=UploadResponse)
async def upload_meter_reading(
    file: UploadFile = File(..., description="Water meter image"),
    city: str = Form(..., description="City name"),
    street_name: str = Form(..., description="Street name"), 
    street_number: str = Form(..., description="Street number")
):
    """Upload water meter image and extract reading using Vision LLM"""
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate file size (max 10MB)
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Prepare address information
        address_info = {
            "city": city.strip(),
            "street_name": street_name.strip(),
            "street_number": street_number.strip()
        }
        
        # Validate address info
        if not all(address_info.values()):
            raise HTTPException(status_code=400, detail="All address fields are required")
        
        # Check if Bedrock service is available
        if not bedrock_service.connected:
            raise HTTPException(status_code=503, detail="Vision analysis service not available")
        
        logger.info(f"Processing meter image upload for {address_info}")
        
        # Analyze image using Vision LLM
        vision_result = await bedrock_service.analyze_meter_image(file_content, address_info)
        
        # Check if analysis was successful
        if not vision_result or not isinstance(vision_result, dict):
            raise HTTPException(status_code=500, detail="Vision analysis failed to return valid results")

        # Check if analysis was successful
        if not vision_result.get("reading_visible", False) and vision_result.get("confidence", 0) < 0.3:
            logger.warning(f"Low confidence reading: {vision_result}")
        
        # Generate unique ID for this reading
        reading_id = f"meter_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Create full address string
        full_address = f"{street_number} {street_name}, {city}"
        
        embeddings = await bedrock_service.generate_meter_embeddings(
            address_info, 
            vision_result["meter_value"], 
            vision_result.get("units", "cubic_meters")
        )

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
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Upload processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/chat")
async def chat():
    """Chat with meter data"""
    raise HTTPException(status_code=501, detail="Not implemented")

@app.get("/readings")
async def get_recent_readings(limit: int = 20):
    """Get recent meter readings for debugging"""
    try:
        if not milvus_service.collection:
            raise HTTPException(status_code=503, detail="Milvus not available")
        
        milvus_service.collection.load()
        
        results = milvus_service.collection.query(
            expr="id != ''",
            output_fields=["id", "meter_value", "full_address", "confidence", "timestamp"],
            limit=limit
        )
        
        # Convert all results
        serializable_results = [convert_milvus_result(result) for result in results]
        
        return {"readings": serializable_results, "count": len(serializable_results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/readings/with-vectors")
async def get_readings_with_vectors(limit: int = 20):
    """Get readings including vector field info"""
    try:
        if not milvus_service.collection:
            raise HTTPException(status_code=503, detail="Milvus not available")
        
        milvus_service.collection.load()
        
        results = milvus_service.collection.query(
            expr="id != ''",
            output_fields=["id", "meter_value", "full_address", "confidence", "address_embedding", "combined_embedding"],
            limit=limit
        )
        
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
        
        return {"readings": serializable_results, "count": len(serializable_results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Water Meter Scanner application...")
    
    # Initialize Milvus with error handling
    try:
        milvus_success = await milvus_service.initialize()
        if milvus_success:
            logger.info("✅ Milvus service initialized successfully")
        else:
            logger.warning("⚠️ Milvus service failed to initialize - running without vector search")
    except Exception as e:
        logger.error(f"❌ Milvus initialization error: {str(e)}")
        logger.info("Application will continue without Milvus functionality")
    
    # Initialize Bedrock with error handling
    try:
        bedrock_success = await bedrock_service.initialize()
        if bedrock_success:
            logger.info("✅ Bedrock service initialized successfully")
        else:
            logger.warning("⚠️ Bedrock service failed to initialize - running without AI functionality")
    except Exception as e:
        logger.error(f"❌ Bedrock initialization error: {str(e)}")
        logger.info("Application will continue without Bedrock functionality")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

