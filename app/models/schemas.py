"""
    Pydantic schemas
"""

from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field, validator


class MeterUploadData(BaseModel):
    """
        Validated upload data
    """

    city: str = Field(..., min_length=1, max_length=100)
    street_name: str = Field(..., min_length=1, max_length=200)
    street_number: str = Field(..., min_length=1, max_length=20)
    file_content: bytes
    file_name: str
    content_type: str
    
    @validator('city', 'street_name', 'street_number')
    def strip_fields(cls, v):
        return v.strip() if isinstance(v, str) else v
    
    @validator('file_content')
    def validate_file_content(cls, v):
        if len(v) == 0:
            raise ValueError("Empty file")
        
        if len(v) > 10 * 1024 * 1024:
            raise ValueError("File too large")
        
        return v


class AddressInfo(BaseModel):
    """
        Address information for water meter location
    """

    city: str = Field(..., min_length=1, max_length=100, description="City name")
    street_name: str = Field(..., min_length=1, max_length=200, description="Street name")
    street_number: str = Field(..., min_length=1, max_length=20, description="Street number")

class MeterReading(BaseModel):
    """
        Water meter reading data
    """

    id: str = Field(..., description="Unique reading ID")
    address: AddressInfo
    meter_value: float = Field(..., ge=0, description="Meter reading value")
    confidence: float = Field(..., ge=0, le=1, description="Extraction confidence score")
    timestamp: int = Field(..., description="Unix timestamp")
    meter_type: Optional[str] = Field(None, description="Type of meter (analog/digital)")
    units: Optional[str] = Field(None, description="Measurement units")
    notes: Optional[str] = Field(None, description="Additional observations")

class UploadResponse(BaseModel):
    """
        Response from meter image upload
    """

    success: bool
    reading_id: str
    meter_value: float
    confidence: float
    address: str
    timestamp: datetime
    notes: Optional[str] = None
    error: Optional[str] = None

class ChatQuery(BaseModel):
    """
        Chat query request
    """

    message: str = Field(..., min_length=1, max_length=500, description="User question")

class ChatResponse(BaseModel):
    """
        Chat response
    """

    response: str = Field(..., description="Generated response")
    sources_count: Optional[int] = Field(None, description="Number of sources used")

class VisionAnalysisResult(BaseModel):
    """
        Result from vision analysis
    """
    meter_value: float
    confidence: float
    meter_type: str
    units: str
    notes: str
    reading_visible: bool
    address: str
    model_used: str
    raw_response: Optional[str] = None
    error: Optional[str] = None