# Llama 3.3 70B Instruct Chat Service

A browser-based chat interface powered by Amazon Bedrock's Llama 3.3 70B Instruct model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [AWS App Runner Deployment](#aws-app-runner-deployment)
- [API Reference](#api-reference)
- [Code Documentation](#code-documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This service provides a minimal single-turn chat interface where users can send messages to Llama 3.3 70B Instruct via Amazon Bedrock. Each conversation is independent with no message history retention.

## Features

- **Simple Web UI**: Clean HTML/JavaScript chat interface
- **REST API**: JSON endpoint for programmatic access
- **Robust Error Handling**: Comprehensive exception management
- **Multiple Response Formats**: Handles various Bedrock response structures
- **Automated Testing**: 80%+ test coverage with pytest
- **Production Ready**: Deployable via AWS App Runner

## Project Structure

```
conversational-ai-assistant/
├── app.py                      # Flask web application
├── bedrock_client.py           # AWS Bedrock API wrapper
├── errors.py                   # Custom exceptions
├── memory.py                   # Simple memory store (optional)
├── requirements.txt            # Python dependencies
├── tests/
│   ├── test_bedrock_client.py  # Unit tests for Bedrock client
│   └── test_flask_endpoints.py # Flask endpoint tests
└── README.md                   # This documentation
```

## Prerequisites

- **Python 3.10+**
- **AWS Account** with Bedrock access
- **IAM Role** with `bedrock-runtime:InvokeModel` permission
- **GitHub Repository** (for App Runner deployment)

### Required Environment Variables

```bash
AWS_REGION="us-east-2"
BEDROCK_MODEL_ID="meta.llama3-3-70b-instruct-v1:0"
```

## Local Development

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd conversational-ai-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
export AWS_REGION="us-east-2"
export BEDROCK_MODEL_ID="meta.llama3-3-70b-instruct-v1:0"
```

### 3. Run Application

```bash
python app.py
```

Access the chat interface at: http://localhost:8080

### 4. Test API Endpoint

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, how are you?"}' \
  http://localhost:8080/chat
```

Expected response:
```json
{
  "reply": "Hello! I'm doing well, thank you for asking. How can I assist you today?"
}
```

## AWS App Runner Deployment

### 1. IAM Role Setup

Create an IAM role with the following trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Attach the `AmazonBedrockFullAccess` policy.

### 2. App Runner Configuration

1. **Source Settings**:
   - Repository: Your GitHub repo
   - Branch: `main`
   - Source directory: `/`

2. **Build Settings**:
   - Runtime: Python 3.10
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Port: `8080`

3. **Service Settings**:
   - Environment variables:
     - `AWS_REGION=us-east-2`
     - `BEDROCK_MODEL_ID=meta.llama3-3-70b-instruct-v1:0`
   - Instance role: Select your IAM role

### 3. Deployment Verification

Test your deployed service:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me a joke"}' \
  https://your-app-url.awsapprunner.com/chat
```

## API Reference

### Endpoints

#### `GET /`
Returns the chat interface HTML page.

**Response**: HTML document with embedded JavaScript

#### `POST /chat`
Processes chat messages and returns AI responses.

**Request Body**:
```json
{
  "message": "Your message here"
}
```

**Success Response** (200):
```json
{
  "reply": "AI assistant response"
}
```

**Error Responses**:
- `400`: Invalid request format
- `500`: Configuration error
- `502`: Bedrock invocation failed

## Code Documentation

### Core Components

#### `bedrock_client.py`

```python
class BedrockClient:
    """AWS Bedrock API wrapper for Llama 3.3 70B Instruct"""
    
    def __init__(self):
        """Initialize client with environment variables"""
        
    def invoke(self, prompt: str, max_gen_len: int = 512, 
               temperature: float = 0.5, top_p: float = 0.9) -> str:
        """Send prompt to Llama and return response"""
```

**Key Features**:
- Environment variable validation
- Multiple response format parsing
- Comprehensive error handling
- JSON request/response management

#### `app.py`

```python
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Handle chat API requests"""

@app.route("/", methods=["GET"])
def index():
    """Serve chat interface HTML"""
```

**Key Features**:
- Input validation and sanitization
- Single-turn prompt construction
- Error response formatting
- Embedded HTML/JavaScript UI

#### `errors.py`

```python
class ConfigurationError(Exception):
    """Missing or invalid configuration"""

class BedrockInvocationError(Exception):
    """Bedrock API call failures"""
```

#### `memory.py`

```python
class Memory:
    """Simple slot-based memory store"""
    
    def parse_and_store(self, user_input: str):
        """Extract and store user information"""
        
    def to_prompt(self) -> str:
        """Generate context string for prompts"""
```

**Note**: Currently unused in single-turn implementation.

### Dependencies

```txt
boto3          # AWS SDK
flask          # Web framework
pytest         # Testing framework
pytest-mock    # Mock testing utilities
```

## Testing

### Run Test Suite

```bash
# Install test dependencies
pip install pytest pytest-mock coverage

# Run tests with coverage
coverage run -m pytest
coverage report --omit="*/venv/*"
```

### Test Coverage Goals

- **Minimum**: 80% overall coverage
- **bedrock_client.py**: All response parsing paths
- **app.py**: All endpoints and error conditions
- **Integration**: End-to-end API testing

### Test Structure

```
tests/
├── test_bedrock_client.py    # Unit tests for Bedrock integration
│   ├── Environment variable validation
│   ├── Response parsing (generation, completion, text, choices, messages)
│   ├── Error handling scenarios
│   └── JSON format edge cases
└── test_flask_endpoints.py   # Flask application tests
    ├── GET / endpoint validation
    ├── POST /chat success scenarios
    ├── Input validation testing
    └── Error response verification
```

### Example Test Output

```
Name                          Stmts   Miss  Cover
-------------------------------------------------
bedrock_client.py               50      5    90%
app.py                          80     10    88%
tests/test_bedrock_client.py    50      0   100%
tests/test_flask_endpoints.py   30      0   100%
-------------------------------------------------
TOTAL                         210     15    92%
```

## Troubleshooting

### Common Issues

#### Role Not Appearing in App Runner
- Verify IAM role trust policy includes `apprunner.amazonaws.com`
- Wait 2-3 minutes for role propagation
- Check that you're in the correct AWS region

#### Bedrock Access Denied
```bash
# Verify model access in your region
aws bedrock list-foundation-models --region us-east-2
```

#### Environment Variables Not Set
```python
# Error: Missing AWS_REGION environment variable
# Solution: Set required environment variables before running
export AWS_REGION="us-east-2"
export BEDROCK_MODEL_ID="meta.llama3-3-70b-instruct-v1:0"
```

#### Local Testing Issues
```bash
# Port already in use
# Solution: Kill existing process or use different port
lsof -ti:8080 | xargs kill -9
```

### Debug Mode

Enable Flask debug mode for development:
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
```

### Logging

Add logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

---

## Summary

This documentation provides a complete guide for developing, testing, and deploying the Llama 3.3 70B Instruct Chat Service. The application offers a simple yet robust interface for interacting with Amazon Bedrock's large language models through both web UI and REST API endpoints.

**Key Benefits**:
- Minimal setup and configuration
- Production-ready deployment workflow
- Comprehensive testing coverage
- Clear error handling and debugging support