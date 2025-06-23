from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from services.milvus_service import MilvusService
from services.bedrock_service import BedrockService
import os
import logging

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
    
@app.get("/bedrock/test-vision")
async def test_vision():
    """Test Bedrock vision capabilities"""
    if not bedrock_service.connected:
        raise HTTPException(status_code=503, detail="Bedrock service not connected")
    
    return {
        "status": "ready",
        "message": "Bedrock vision service is ready for image analysis",
        "vision_model": bedrock_service.vision_model
    }

# Placeholder routes for future implementation
@app.post("/upload-meter")
async def upload_meter_reading():
    """Upload meter image and extract reading"""
    raise HTTPException(status_code=501, detail="Not implemented")

@app.post("/chat")
async def chat():
    """Chat with meter data"""
    raise HTTPException(status_code=501, detail="Not implemented")

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

