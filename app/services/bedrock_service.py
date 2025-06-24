import boto3
import json
import base64
import logging
import os
import re
from PIL import Image, ImageEnhance
import io

logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.vision_model = os.getenv("BEDROCK_VISION_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.text_model = os.getenv("BEDROCK_TEXT_MODEL", "anthropic.claude-v2")
        self.embed_model = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
        
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
        
    async def generate_embedding(self, text: str) -> list:
        """Generate embedding using Bedrock Titan"""
        try:
            if not self.connected:
                raise Exception("Bedrock service not connected")
            
            # Prepare request for Titan Embeddings
            request_body = {
                "inputText": text
            }
            
            # Call Bedrock Embeddings API
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embed_model,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            if not embedding:
                raise ValueError("No embedding returned from Bedrock")
            
            logger.info(f"✅ Generated embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise

    async def generate_meter_embeddings(self, address_info: dict, meter_value: float, units: str) -> dict:
        """Generate both address and combined embeddings for meter reading"""
        
        # Create address text
        full_address = f"{address_info.get('street_number', '')} {address_info.get('street_name', '')}, {address_info.get('city', '')}"
        
        # Create combined context text
        combined_text = f"Water meter at {full_address} reading {meter_value} {units}"
        
        # Generate both embeddings
        address_embedding = await self.generate_embedding(full_address)
        combined_embedding = await self.generate_embedding(combined_text)
        
        return {
            "address_embedding": address_embedding,
            "combined_embedding": combined_embedding,
            "full_address": full_address
        }
            
    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """Enhance image quality for better OCR results"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 1024x1024 for faster processing)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.2)  # Slightly increase contrast
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.1)  # Slightly sharpen
            
            # Save back to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}, using original image")
            return image_bytes
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def analyze_meter_image(self, image_bytes: bytes, address_info: dict) -> dict:
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
            
            # Preprocess image for better recognition
            processed_image_bytes = self._preprocess_image(image_bytes)

            # Encode image
            image_base64 = self._encode_image(processed_image_bytes)
            
            # Create structured prompt for meter reading extraction
            full_address = f"{address_info.get('street_number', '')} {address_info.get('street_name', '')}, {address_info.get('city', '')}"
            
            prompt = f"""
                        You are an expert technician in water-meter reading. Carefully analyze the attached image and return ONLY the JSON specified below—no additional text.

                        Address: {full_address}  

                        INSTRUCTIONS:

                        1. Identify meter type:
                           - analog       - rotating pointer dials (0-9)  
                           - digital      - full LCD/LED numeric display  
                           - mechanical   - rolling number wheels visible through windows  
                           - unclear      - cannot determine type confidently  

                        2. Read value (left→right, largest to smallest units) and note colored digits:
                           - Color-coding note: Digits in a different color (often red) represent the fractional part—i.e., everything right of the decimal point. Include them after “.”

                        3. Type-specific rules:
                           - analog:  Format: XXXXX.XXX (6-8 digits, with three decimals)  
                           - digital: Transcribe all primary digits and any visible decimal places  
                           - mechanical: Read black wheel digits only; ignore smaller digits but doesn't ignore the fractional part.

                        4. Determine units (cubic_meters, gallons, or liters). If unknown, set units to null.

                        5. Visibility & confidence:
                           confidence: float 0.0-1.0 reflecting clarity  
                           - 1.0: perfect clarity  
                           - 0.8: minor glare/tilt  
                           - 0.6: slight obstruction  
                           - 0.4: partial obstruction/low contrast  
                           - 0.2: very unclear, heavy glare/blur  
                           - 0.0: unreadable  

                        OUTPUT (strictly this JSON):
                        {{
                            "meter_value": [the actual reading as a number],
                            "confidence": [0.0 to 1.0 based on image clarity],
                            "meter_type": "analog|digital|mechanical|unclear",
                            "units": "cubic_meters|gallons|liters",
                            "notes": "Description of what you see and why you chose this reading",
                            "reading_visible": true|false
                        }}
                        
                        IMPORTANT: 
                            - meter_value must be a valid number (no leading zeros like 010009129)
                            - If you see 010009129, return it as 10009129.0
                            - Always use decimal format (add .0 if whole number)
                        BE VERY CAREFUL with dial positions. If a pointer is between two number - get a lower number
                    """


            # Prepare request body for Claude Vision
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.1,
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
            content_cleaned = content.strip()
            
            logger.info(f"Vision model response: {content}")
            
            # Extract JSON from response
            try:
                # Clean the response text
                content_cleaned = re.sub(r':\s*0+(\d+)', r': \1', content_cleaned)
                
                # Try multiple strategies to find JSON
                json_result = None
                
                # Strategy 1: Look for JSON block markers
                if "```json" in content_cleaned:
                    start_idx = content_cleaned.find("```json") + 7
                    end_idx = content_cleaned.find("```", start_idx)
                    if end_idx != -1:
                        json_str = content_cleaned[start_idx:end_idx].strip()
                        json_result = json.loads(json_str)
                
                # Strategy 2: Find JSON braces
                elif '{' in content_cleaned and '}' in content_cleaned:
                    start_idx = content_cleaned.find('{')
                    end_idx = content_cleaned.rfind('}') + 1
                    json_str = content_cleaned[start_idx:end_idx]
                    json_result = json.loads(json_str)
                
                # Strategy 3: Try parsing entire response
                else:
                    json_result = json.loads(content_cleaned)
                
                if not json_result:
                    raise ValueError("No valid JSON found in response")
                
                # Validate and fix meter_value
                if 'meter_value' in json_result:
                    # Handle string numbers or leading zeros
                    meter_val = json_result['meter_value']
                    if isinstance(meter_val, str):
                        meter_val = meter_val.lstrip('0') or '0'  # Remove leading zeros
                    json_result['meter_value'] = float(meter_val)
                else:
                    # Try to extract meter value from text if JSON parsing failed
                    value_match = re.search(r'meter[_\s]*value["\s]*:?\s*([0-9]+\.?[0-9]*)', content_cleaned, re.IGNORECASE)
                    if value_match:
                        json_result['meter_value'] = float(value_match.group(1))
                    else:
                        json_result['meter_value'] = 0.0
                
                result = json_result

                # Ensure confidence is between 0 and 1
                confidence = float(result.get('confidence', 0.0))
                result['confidence'] = max(0.0, min(1.0, confidence))
                
                # Add metadata
                result['address'] = full_address
                result['model_used'] = self.vision_model
                
                logger.info(f"✅ Successfully extracted meter reading: {result['meter_value']} (confidence: {result['confidence']})")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse vision model response: {str(e)}")
                logger.error(f"Raw response: {content}")
                
                # Try to extract meter value using regex as fallback
                meter_value = 0.0
                confidence = 0.0
                
                # Look for numeric values in the response
                value_patterns = [
                    r'(\d+\.?\d*)\s*(?:cubic|liters|gallons|m3|m³)',
                    r'reading[:\s]+(\d+\.?\d*)',
                    r'value[:\s]+(\d+\.?\d*)',
                    r'(\d{3,6}\.?\d*)'  # 3-6 digit numbers (typical meter readings)
                ]
                
                for pattern in value_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        meter_value = float(match.group(1))
                        confidence = 0.7  # Medium confidence for regex extraction
                        break
                
                # Return fallback result with extracted info
                result = {
                    "meter_value": meter_value,
                    "confidence": confidence,
                    "meter_type": "unknown",
                    "units": "unknown",
                    "notes": f"JSON parsing failed, used text extraction. Original error: {str(e)}",
                    "reading_visible": meter_value > 0,
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
    
    def health_check(self) -> dict:
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