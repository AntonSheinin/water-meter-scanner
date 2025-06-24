from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
import logging
import time

logger = logging.getLogger(__name__)

class MilvusService:
    def __init__(self):
        self.collection_name = "water_meters"
        self.collection = None
        self.connected = False
    
    async def connect(self) -> bool:
        """
            Connect to Milvus database
        """

        max_retries = 3
        retry_delay = 2

        host = os.getenv("MILVUS_HOST", "milvus-standalone")
        port = os.getenv("MILVUS_PORT", "19530")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Milvus at {host}:{port} (attempt {attempt + 1})")
                
                connections.connect(
                    alias="default",
                    host=host,
                    port=port,
                    timeout=10
                )
                
                collections = utility.list_collections()
                logger.info(f"Successfully connected to Milvus. Existing collections: {collections}")
                
                self.connected = True
                return True
                
            except Exception as exc:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(exc)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("All connection attempts failed")
                    self.connected = False
                    return False
    
    async def create_collection(self) -> bool:
        """
            Create the water meters collection with proper schema
        """

        try:
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                logger.info(f'Collection "{self.collection_name}" already exists')
                self.collection = Collection(self.collection_name)
                return True
            
            # Define collection schema
            fields = [
                FieldSchema(
                    name='id', 
                    dtype=DataType.VARCHAR, 
                    max_length=100, 
                    is_primary=True,
                    description='Unique meter reading ID'
                ),
                FieldSchema(
                    name='address_embedding', 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=1024,
                    description='Address semantic embedding'
                ),
                FieldSchema(
                    name='combined_embedding', 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=1024,
                    description='Combined address + context embedding'
                ),
                FieldSchema(
                    name='meter_value', 
                    dtype=DataType.FLOAT,
                    description='Water meter reading value'
                ),
                FieldSchema(
                    name="city", 
                    dtype=DataType.VARCHAR, 
                    max_length=100,
                    description='City name'
                ),
                FieldSchema(
                    name='street_name', 
                    dtype=DataType.VARCHAR, 
                    max_length=200,
                    description='Street name'
                ),
                FieldSchema(
                    name='street_number', 
                    dtype=DataType.VARCHAR, 
                    max_length=20,
                    description='Street number'
                ),
                FieldSchema(
                    name="full_address", 
                    dtype=DataType.VARCHAR, 
                    max_length=500,
                    description='Complete address string'
                ),
                FieldSchema(
                    name='timestamp', 
                    dtype=DataType.INT64,
                    description='Unix timestamp of reading'
                ),
                FieldSchema(
                    name='confidence', 
                    dtype=DataType.FLOAT,
                    description='Extraction confidence score'
                )
            ]
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description='Water meter readings with multi-vector embeddings'
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f'Created collection "{self.collection_name}" successfully')
            return True
            
        except Exception as exc:
            logger.error(f'Failed to create collection: {str(exc)}')
            return False
    
    async def create_indexes(self) -> bool:
        """
            Create vector indexes for efficient search
        """

        try:
            if not self.collection:
                return False
            
            # Index parameters for vector fields
            index_params = {
                'metric_type': 'L2',
                'index_type': 'IVF_FLAT',
                'params': {'nlist': 128}
            }
            
            # Create index for address_embedding
            self.collection.create_index(
                field_name='address_embedding',
                index_params=index_params
            )
            
            # Create index for combined_embedding
            self.collection.create_index(
                field_name='combined_embedding', 
                index_params=index_params
            )
            
            logger.info('Created vector indexes successfully')
            return True
            
        except Exception as exc:
            logger.error(f'Failed to create indexes: {str(e)}')
            return False
    
    async def initialize(self) -> bool:
        """
            Complete initialization process
        """

        try:
            # Connect to Milvus
            logger.info("Initializing Milvus service...")

            if not await self.connect():
                logger.error("Failed to connect to Milvus")
                return False
            
            # Create collection
            if not await self.create_collection():
                logger.error("Failed to create collection")
                return False
            
            # Create indexes
            if not await self.create_indexes():
                logger.error("Failed to create indexes")
                return False
            
            logger.info('Milvus service initialized successfully')
            return True
            
        except Exception as exc:
            logger.error(f'Failed to initialize Milvus service: {str(exc)}')
            return False
    
    def get_collection_info(self) -> dict | None:
        """
            Get collection information
        """

        if not self.collection:
            return None
        
        try:
            # Load collection to get stats
            self.collection.load()
            
            return {
                'name': self.collection.name,
                'description': self.collection.description,
                'num_entities': self.collection.num_entities,
                'schema': {
                    'fields': [
                        {
                            'name': field.name,
                            'type': str(field.dtype),
                            'description': field.description
                        }
                        for field in self.collection.schema.fields
                    ]
                }
            }
        except Exception as exc:
            logger.error(f'Failed to get collection info: {str(exc)}')
            return None
    
    def health_check(self) -> dict:
        """
            Check Milvus connection health
        """

        try:
            if not self.connected:
                return {'status': 'disconnected', 'error': 'Not connected to Milvus'}
            
            # Test connection by listing collections
            collections = utility.list_collections()
            
            return {
                'status': 'healthy',
                'connected': True,
                'collections': collections,
                'target_collection': self.collection_name,
                'collection_exists': self.collection_name in collections
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        
    async def store_meter_reading(
        self, 
        reading_id: str,
        address_info: dict,
        meter_value: float,
        confidence: float,
        embeddings: dict,
        timestamp: int,
        units: str = "cubic_meters",
        meter_type: str = "unknown"
    ) -> bool:
        """Store complete meter reading with embeddings in Milvus"""
        try:
            if not self.collection:
                logger.error("Collection not available for storage")
                return False
            
            # Prepare data for insertion
            data = [
                [reading_id],
                [embeddings["address_embedding"]],
                [embeddings["combined_embedding"]],
                [meter_value],
                [address_info.get("city", "")],
                [address_info.get("street_name", "")],
                [address_info.get("street_number", "")],
                [embeddings["full_address"]],
                [timestamp],
                [confidence]
            ]
            
            # Insert into Milvus
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"✅ Stored meter reading {reading_id} in Milvus")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store meter reading: {str(e)}")
            return False

    async def search_by_address(self, query: str, limit: int = 10) -> list:
        """Search meters by address similarity"""
        # Will implement search in Step 4
        return []

    async def search_by_context(self, query: str, limit: int = 10) -> list:
        """Search meters by combined context similarity"""
        # Will implement search in Step 4
        return []