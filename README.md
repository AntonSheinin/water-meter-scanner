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
- **Embeddings**: sentence-transformers
- **Deployment**: Docker & Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- AWS account with Bedrock access
- AWS CLI configured or environment variables set

### Setup

1. **Clone and setup project:**
   ```bash
   git clone <repository-url>
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

### Development

For development mode:
```bash
# Run only Milvus
docker-compose up milvus-standalone

# Run FastAPI locally
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

## Project Structure

```
water-meter-scanner/
├── docker-compose.yml          # Container orchestration
├── app/                        # FastAPI application
│   ├── Dockerfile             # Application container
│   ├── requirements.txt       # Python dependencies
│   ├── main.py               # FastAPI entry point
│   ├── config/               # Configuration
│   ├── services/             # Business logic services
│   ├── models/               # Data models
│   ├── api/                  # API routes
│   └── static/               # Web UI files
├── uploads/                   # Image storage (auto-created)
└── logs/                     # Application logs (auto-created)
```

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /upload-meter` - Upload meter image (coming soon)
- `POST /chat` - Chat with meter data (coming soon)

## Development Status

- [x] Phase 1.1: Project structure and Docker setup
- [ ] Phase 1.2: Basic FastAPI application
- [ ] Phase 1.3: Milvus connection and schema
- [ ] Phase 2: Vision LLM integration
- [ ] Phase 3: Embedding and storage
- [ ] Phase 4: Query processing and chat
- [ ] Phase 5: Web interface and testing

## Contributing

This is a prototype project. For development:

1. Follow the project structure
2. Add tests for new features
3. Update documentation
4. Use proper error handling

## License

[Your License Here]