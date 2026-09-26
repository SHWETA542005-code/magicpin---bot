# Magicpin Vera AI Challenge

An AI-powered engagement bot built using FastAPI and Groq for generating contextual merchant engagement messages and handling conversations.

## Tech Stack

* Python
* FastAPI
* Groq API
* Pydantic
* Render

## Features

* Context-aware message generation
* Merchant and category-specific messaging
* Trigger-based engagement
* Conversation handling
* Duplicate message suppression
* REST API endpoints

## API Endpoints

| Endpoint       | Method | Description                 |
| -------------- | ------ | --------------------------- |
| `/v1/healthz`  | GET    | Health check                |
| `/v1/metadata` | GET    | Team and model information  |
| `/v1/context`  | POST   | Push context                |
| `/v1/tick`     | POST   | Generate engagement actions |
| `/v1/reply`    | POST   | Handle conversation replies |
| `/v1/teardown` | POST   | Clear stored context        |

## Local Setup

Install dependencies:

```bash
cd bot
pip install -r requirements.txt
```

Set `GROQ_API_KEY` in a local `.env` file.

Run the application:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## Deployment

Deployed on Render.

**Live API:** https://magicpin-ai-bot-3gsc.onrender.com

**API Documentation:** https://magicpin-ai-bot-3gsc.onrender.com/docs

## Author

Shweta
