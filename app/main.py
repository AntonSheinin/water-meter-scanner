"""
    FastAPI entrypoint
"""

import uvicorn
import logging
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.dependencies import bedrock_service, milvus_service


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Water Meter Scanner",
    description="A web application for scanning home water usage counters"
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """
        Initialize services on startup
    """

    logger.info("Starting Water Meter Scanner application...")
    
    # Initialize Milvus with error handling
    try:
        milvus_success = await milvus_service.initialize()

        if milvus_success:
            logger.info("✅ Milvus service initialized successfully")

        else:
            logger.warning("⚠️ Milvus service failed to initialize - running without vector search")

    except Exception as exc:
        logger.error(f"❌ Milvus initialization error: {str(exc)}")
        logger.info("Application will continue without Milvus functionality")
    
    # Initialize Bedrock with error handling
    try:
        bedrock_success = await bedrock_service.initialize()
        if bedrock_success:
            logger.info("✅ Bedrock service initialized successfully")

        else:
            logger.warning("⚠️ Bedrock service failed to initialize - running without AI functionality")

    except Exception as exc:
        logger.error(f"❌ Bedrock initialization error: {str(exc)}")
        logger.info("Application will continue without Bedrock functionality")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

