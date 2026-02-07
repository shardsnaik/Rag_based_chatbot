# 🏥 Medical Bot – FastAPI + Quantized MedGemma (GGUF)

# https://conversationaimodel.netlify.app/

A production-ready medical conversational AI backend built using FastAPI, LangChain, and a 5-bit quantized MedGemma 4B GGUF model, designed to run efficiently on CPU-only environments such as AWS Lambda (container-based) or traditional cloud services.

This backend exposes REST APIs for chat, memory, and session management, and is intended to be consumed by a React frontend (Netlify) or any HTTP client.

## 🚀 Key Features

 * ✅ Quantized GGUF model (Q5_K_M) for low-memory CPU inference

* ✅ FastAPI backend with async support

* ✅ LangChain conversational memory (context-aware chat)

* ✅ Session-based chat history

* ✅ Hugging Face Hub integration for model download

* ✅ AWS Lambda compatible (uses /tmp storage + Mangum)

* ✅ CORS-enabled for browser clients (Netlify / React)

## 🧠 Model Details

* Base model: MedGemma 4B (instruction-tuned)

* Format: GGUF

* Quantization: Q5_K_M (CPU-friendly)

* Source: Hugging Face Hub

* Loaded via: llama-cpp-python

The model is **downloaded at startup** from Hugging Face and cached locally:

* ` /tmp/huggingface_cache (AWS Lambda)`

* ` ./model_cache (local / non-Lambda)`

## 🏗️ Architecture Overview

```
React (Netlify)
      |
      |  HTTPS (POST /invocation)
      v
API Gateway
      |
      v
AWS Lambda (FastAPI + LangChain + llama.cpp)
      |
      v
Hugging Face Hub (GGUF model)
```

* The model and API live in the same FastAPI service

* The model is loaded once per container cold start

* Conversation memory is stored in-memory per session

## 📂 Project Structure

```
medical_bot_main_file.py   # Main FastAPI + model + endpoints
.env                       # Environment variables (HF_TOKEN, etc.)
requirements.txt
README.md
```

## 🔑 Environment Variables

Create a .env file or configure in AWS Lambda:
```
HF_TOKEN=your_huggingface_token   # Only required for private models
```

## 🔌 API Endpoints
🏠 Root

``GET /``

Returns API status and available endpoints.

## 💬 Chat (Main Endpoint)

``POST /invocation``

### Request
```
{
  "query": "What are the symptoms of diabetes?"
}
```

### Response
```
{
  "response": "Diabetes symptoms may include...",
  "status": "success"
}
```

## 🧠 Get Conversation Memory

`GET /memory`

Returns formatted chat history for the current session.

## 🧹 Clear Conversation Memory

`GET /clear-memory`

Clears the in-memory conversation history.

## 🆔 Get Session Info

`GET /get_session_id`

Returns session ID, uptime, and message count.

## ❤️ Health Check

`GET /health`

Returns service health and model initialization status.

## 🌐 CORS Configuration

CORS is enabled to support browser clients:

```
allow_origins=["*"]
allow_methods=["GET", "POST", "OPTIONS"]
allow_headers=["*"]
```

⚠️ For production, restrict this to your Netlify domain.

## 🧪 Local Development
### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run locally
python medical_bot_main_file.py


Server starts at:

```
http://127.0.0.1:8000
```

Swagger UI:

`http://127.0.0.1:8000/docs`

## ☁️ AWS Lambda Deployment Notes

* Uses Mangum to adapt FastAPI → Lambda

* Model stored in Hugging Face Hub, cached in /tmp

* Recommended deployment method:

  * Lambda Container Image (Docker)

* Memory: ≥ 6–8 GB

* Timeout: ≥ 60 seconds

⚠️ Cold start will include model download + load.

## ⚠️ Medical Disclaimer

This system is for educational and informational purposes only.
It does not provide medical diagnosis or treatment advice.
Always consult a qualified healthcare professional.

## 📌 Tech Stack

* FastAPI
* LangChain
* llama-cpp-python
* Hugging Face Hub
* AWS Lambda
* Mangum
* Python 3.10+

## ✅ Status

✔ Production-ready
✔ CPU-optimized
✔ Frontend-compatible
✔ Cloud-deployable

