# Water Meter Scanner

A web application for scanning home water usage counters using computer vision and LLM technology.

## Features

- Upload water meter images with address information
- Extract meter readings using AWS Bedrock Vision LLM
- Store data in Milvus vector database for semantic search
- Chat interface to query meter readings using natural language

## Technology Stack

- **Backend**: FastAPI
- **Database**: Milvus 2.5 (vector database)
- **AI**: AWS Bedrock (Vision LLM + Text LLM)
- **Embeddings**: AWS Titan
- **Deployment**: Docker & Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- AWS account with Bedrock access
- AWS CLI configured or environment variables set

### Setup

1. **Clone and setup project:**
   ```bash
   git clone https://github.com/AntonSheinin/water-meter-scanner
   cd water-meter-scanner
   ```

2. **Configure environment:**
   ```bash
   # Edit .env with your AWS credentials
   nano .env
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Web UI: http://localhost:8000
   - API docs: http://localhost:8000/docs
   - Milvus: http://localhost:19530

## API Endpoints

- `GET /` - Web interface
- `POST /upload-meter` - Upload meter image
- `POST /chat` - Chat with meter data
- `GET /health` - Services health check
- `GET /milvus-info` - Milvus collection info
- `GET /readings` - Stored readings
- `GET /docs` - API Documentation
- `GET /redocs` - ReDoc Documentation
