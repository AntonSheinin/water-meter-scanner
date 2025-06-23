import boto3
import json
import base64
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.vision_model = os.getenv("BEDROCK_VISION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.text_model = os.getenv("BEDROCK_TEXT_MODEL", "anthropic.claude-v2")
        
        # Initialize Bedrock client
        self.bedrock_runtime = None
        self.connected = False
    
    async def connect(self):
        """Initialize Bedrock runtime client"""
        try:
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Test connection by listing available models
            bedrock_client = boto3.client(
                'bedrock',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            models = bedrock_client.list_foundation_models()
            logger.info(f"✅ Connected to AWS Bedrock in {self.region}")
            logger.info(f"Available models: {len(models.get('modelSummaries', []))}")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to AWS Bedrock: {str(e)}")
            self.connected = False
            return False
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def analyze_meter_image(self, image_bytes: bytes, address_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze water meter image using Claude Vision to extract meter reading
        
        Args:
            image_bytes: Raw image data
            address_info: Dictionary with city, street_name, street_number
            
        Returns:
            Dictionary with meter_value, confidence, and metadata
        """
        try:
            if not self.connected:
                raise Exception("Bedrock service not connected")
            
            # Encode image
            image_base64 = self._encode_image(image_bytes)
            
            # Create structured prompt for meter reading extraction
            full_address = f"{address_info.get('street_number', '')} {address_info.get('street_name', '')}, {address_info.get('city', '')}"
            
            prompt = f"""
                        You are an expert at reading water meter displays. Analyze this image of a water meter and extract the current reading.

                        Address: {full_address}

                        Please examine the image carefully and provide a JSON response with the following structure:
                        {{
                            "meter_value": <numeric_value>,
                            "confidence": <0.0_to_1.0>,
                            "meter_type": "<analog|digital>",
                            "units": "<cubic_meters|gallons>",
                            "notes": "<any_observations_or_issues>",
                            "reading_visible": <true|false>
                        }}

                        Instructions:
                        1. Look for the main numeric display showing the water usage
                        2. If it's an analog meter, read the position of the dials/needles
                        3. If it's a digital meter, read the displayed numbers
                        4. Provide confidence based on image clarity and visibility
                        5. Note any issues like glare, obstruction, or unclear readings

                        Be precise with the numeric value and conservative with confidence if the reading is unclear.
                    """

            # Prepare request body for Claude Vision
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Call Bedrock Vision API
            response = self.bedrock_runtime.invoke_model(
                modelId=self.vision_model,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            logger.info(f"Vision model response: {content}")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                # Validate required fields
                if 'meter_value' not in result:
                    raise ValueError("meter_value not found in response")
                
                # Ensure confidence is between 0 and 1
                confidence = float(result.get('confidence', 0.5))
                result['confidence'] = max(0.0, min(1.0, confidence))
                
                # Add metadata
                result['address'] = full_address
                result['model_used'] = self.vision_model
                
                logger.info(f"✅ Successfully extracted meter reading: {result['meter_value']} (confidence: {result['confidence']})")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse vision model response: {str(e)}")
                # Return fallback result
                return {
                    "meter_value": 0.0,
                    "confidence": 0.0,
                    "meter_type": "unknown",
                    "units": "unknown",
                    "notes": f"Failed to parse response: {str(e)}",
                    "reading_visible": False,
                    "address": full_address,
                    "model_used": self.vision_model,
                    "raw_response": content
                }
                
        except Exception as e:
            logger.error(f"❌ Vision analysis failed: {str(e)}")
            return {
                "meter_value": 0.0,
                "confidence": 0.0,
                "meter_type": "unknown",
                "units": "unknown", 
                "notes": f"Analysis failed: {str(e)}",
                "reading_visible": False,
                "address": address_info,
                "model_used": self.vision_model,
                "error": str(e)
            }
    
    async def generate_chat_response(self, query: str, context_data: list) -> str:
        """
        Generate chat response using Claude Text model
        
        Args:
            query: User's question
            context_data: List of relevant meter readings
            
        Returns:
            Generated response string
        """
        try:
            if not self.connected:
                raise Exception("Bedrock service not connected")
            
            # Format context data
            context = self._format_context_for_chat(context_data)
            
            # Create prompt for chat response
            prompt = f"""
                        You are a helpful assistant for water meter readings. Use the following data to answer questions about water usage.

                        Available meter data:
                        {context}

                        User question: {query}

                        Please provide a clear, helpful answer based on the available data. If the question asks for specific values, include the meter readings. 
                        If comparing values, show the numbers clearly. If asking about locations, include the addresses.

                        Answer:
                    """

            # Prepare request for Claude Text model
            request_body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 500,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            # Call Bedrock Text API
            response = self.bedrock_runtime.invoke_model(
                modelId=self.text_model,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            generated_text = response_body.get('completion', '').strip()
            
            logger.info(f"✅ Generated chat response for query: {query}")
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Chat response generation failed: {str(e)}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"
    
    def _format_context_for_chat(self, context_data: list) -> str:
        """Format context data for chat prompt"""
        if not context_data:
            return "No meter readings available."
        
        formatted = "Available meter readings:\n"
        for i, item in enumerate(context_data[:10], 1):  # Limit to top 10
            address = item.get('full_address', 'Unknown address')
            value = item.get('meter_value', 'Unknown')
            formatted += f"{i}. {address}: {value} units\n"
        
        return formatted
    
    def health_check(self) -> Dict[str, Any]:
        """Check Bedrock service health"""
        try:
            if not self.connected:
                return {
                    "status": "disconnected",
                    "error": "Not connected to AWS Bedrock"
                }
            
            return {
                "status": "healthy",
                "connected": True,
                "region": self.region,
                "vision_model": self.vision_model,
                "text_model": self.text_model
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def initialize(self) -> bool:
        """Initialize the Bedrock service"""
        return await self.connect()