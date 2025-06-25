import logging
from datetime import datetime

from services.bedrock_service import BedrockService
from services.milvus_service import MilvusService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_similar_readings(
        milvus_service: MilvusService,
        bedrock_service: BedrockService,
        query: str, 
        search_type: str = "combined", 
        limit: int = 10
    ) -> list:
    """
        Complete similarity search: Query → Embedding → Vector Search → Results
    """

    if not milvus_service.connected:
        await milvus_service.connect()
        
    if not milvus_service.collection:
        await milvus_service.create_collection()

    try:
        query_embedding = await bedrock_service.generate_embedding(query)
        
        # Step 3: Choose vector field based on search type
        vector_field = "combined_embedding" if search_type == "combined" else "address_embedding"
        
        # Step 4: Perform similarity search
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = milvus_service.collection.search(
            [query_embedding],          # Query vector
            vector_field,               # Which embedding field to search
            search_params,              # Search parameters
            limit=limit,                # Number of results
            output_fields=["id", "meter_value", "full_address", "confidence", "timestamp"]
        )
    
    except Exception as exc:
        logger.error(f"❌ Similarity search failed: {str(exc)}")
        return []
        
    formatted_results = []
    for result in results[0]:  # results[0] because we sent 1 query
        formatted_results.append({
            "id": result.entity.get("id"),
            "meter_value": float(result.entity.get("meter_value") or 0),
            "full_address": result.entity.get("full_address"),
            "confidence": float(result.entity.get("confidence") or 0),
            "timestamp": int(result.entity.get("timestamp") or 0),
            "similarity_score": float(result.distance),  # ← Similarity distance
            "rank": len(formatted_results) + 1
        })
    
    logger.info(f"✅ Found {len(formatted_results)} similar readings for query: {query[:50]}...")
    return formatted_results

async def search_by_address(
        milvus_service: MilvusService,
        bedrock_service: BedrockService,
        query: str, 
        limit: int = 10,
    ) -> list:
    """
        Search meters by address similarity
    """

    if not milvus_service.collection:
        return []

    try:
        milvus_service.collection.load()

        if not bedrock_service.connected:
            await bedrock_service.connect()
        
        # Generate query embedding
        query_embedding = await bedrock_service.generate_embedding(query)
        
        # Search in address_embedding field
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = milvus_service.collection.search(
            [query_embedding],
            "address_embedding",
            search_params,
            limit=limit,
            output_fields=["id", "meter_value", "full_address", "confidence", "timestamp", "city", "street_name", "street_number"]
        )
        
        # Format results
        formatted_results = []

        for result in results[0]:
            formatted_results.append({
                "id": result.entity.get("id"),
                "meter_value": float(result.entity.get("meter_value")),
                "full_address": result.entity.get("full_address"),
                "confidence": float(result.entity.get("confidence")),
                "timestamp": int(result.entity.get("timestamp")),
                "city": result.entity.get("city"),
                "street_name": result.entity.get("street_name"),
                "street_number": result.entity.get("street_number"),
                "similarity_score": float(result.distance)
            })
        
        return formatted_results
        
    except Exception as exc:
        logger.error(f"❌ Address search failed: {str(exc)}")
        return []

async def search_by_context(
        milvus_service: MilvusService,
        bedrock_service: BedrockService,
        query: str, 
        limit: int = 10
    ) -> list:
    """
        Search meters by combined context similarity
    """

    if not milvus_service.collection:
        return []
    
    try:   
        milvus_service.collection.load()

        if not bedrock_service.connected:
            await bedrock_service.connect()
        
        query_embedding = await bedrock_service.generate_embedding(query)
        
        # Search in combined_embedding field
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = milvus_service.collection.search(
            [query_embedding],
            "combined_embedding", 
            search_params,
            limit=limit,
            output_fields=["id", "meter_value", "full_address", "confidence", "timestamp", "city", "street_name", "street_number"]
        )
        
        # Format results
        formatted_results = []
        for result in results[0]:
            formatted_results.append({
                "id": result.entity.get("id"),
                "meter_value": float(result.entity.get("meter_value")),
                "full_address": result.entity.get("full_address"),
                "confidence": float(result.entity.get("confidence")),
                "timestamp": int(result.entity.get("timestamp")),
                "city": result.entity.get("city"),
                "street_name": result.entity.get("street_name"),
                "street_number": result.entity.get("street_number"),
                "similarity_score": float(result.distance)
            })
        
        return formatted_results
        
    except Exception as exc:
        logger.error(f"❌ Context search failed: {str(exc)}")
        return []

async def search_by_recency(
        milvus_service: MilvusService,
        query: str, 
        limit: int = 10
    ) -> list:
    """
        Search for recent readings with optional filtering
    """

    if not milvus_service.collection:
        logger.error("Collection not available for recency search")
        return []

    try:             
        milvus_service.collection.load()

    except Exception as exc:
        logger.error(f"❌ Collection load failed: {str(exc)}")
        return []
        
    # Query all records (or filter by query if provided)
    if query.strip():
        # If query provided, try to filter by address content
        # Note: This is a simple text matching, not semantic search
        expr = f"full_address like '%{query}%' or city like '%{query}%' or street_name like '%{query}%'"

    else:
        # Get all records
        expr = "id != ''"
    
    try:
        # Simple query sorted by timestamp (most recent first)
        results = milvus_service.collection.query(
            expr=expr,
            output_fields=[
                "id", "meter_value", "full_address", "confidence", "timestamp", 
                "city", "street_name", "street_number"
            ],
            limit=limit * 2  # Get more records to sort properly
        )

    except Exception as query_error:
        # If filtering fails, fall back to getting all records
        logger.warning(f"Filtered query failed, getting all records: {str(query_error)}")
        results = milvus_service.collection.query(
            expr="id != ''",
            output_fields=[
                "id", "meter_value", "full_address", "confidence", "timestamp", 
                "city", "street_name", "street_number"
            ],
            limit=limit * 2
        )
    
    if not results:
        logger.warning("No records found in collection")
        return []
    
    # Convert Milvus results to serializable format and sort by timestamp
    converted_results = []
    for result in results:
        converted_result = {}
        for key, value in result.items():
            # Handle numpy types
            if hasattr(value, 'item'):  # numpy scalar
                converted_result[key] = value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                converted_result[key] = value.tolist()
            else:
                converted_result[key] = value
        converted_results.append(converted_result)
    
    # Sort by timestamp descending (most recent first)
    sorted_results = sorted(
        converted_results, 
        key=lambda x: x.get("timestamp", 0), 
        reverse=True
    )
    
    # Take only the requested limit
    limited_results = sorted_results[:limit]
    
    # Convert to same format as similarity search results
    formatted_results = []
    for i, result in enumerate(limited_results):
        formatted_result = {
            "id": result.get("id", ""),
            "meter_value": float(result.get("meter_value", 0)),
            "full_address": result.get("full_address", ""),
            "confidence": float(result.get("confidence", 0)),
            "timestamp": int(result.get("timestamp", 0)),
            "city": result.get("city", ""),
            "street_name": result.get("street_name", ""),
            "street_number": result.get("street_number", ""),
            "similarity_score": 0.0,  # Perfect match for timestamp-based search
            "rank": i + 1,
            "search_type": "recency"
        }
        formatted_results.append(formatted_result)
    
    logger.info(f"✅ Found {len(formatted_results)} recent readings (query: '{query[:50]}...')")
    
    # Log first few results for debugging
    for i, result in enumerate(formatted_results[:3]):
        timestamp_readable = datetime.fromtimestamp(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Recent result {i+1}: {result['full_address']} - {result['meter_value']} at {timestamp_readable}")
    
    return formatted_results
