from fastapi import FastAPI, HTTPException, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as ps, ServerlessSpec
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from huggingface_hub import InferenceClient

from concurrent.futures import ThreadPoolExecutor
import os, uvicorn, json, threading, asyncio
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Pinecone API keys and index
pc = ps(api_key=os.getenv("Pinecone_api_key"))
key = os.getenv("huggingface_api_key")
huggingface_client = InferenceClient(token=key)

# Define Pinecone index specifications
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "multi-qa-index"

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=spec)
else:
    print(f"Index '{index_name}' already exists.")

# Initialize Pinecone index
index = pc.Index(index_name)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://conversationaimodel.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Initialize Hugging Face embeddings using Mistral-7B-Instruct-v0.3 model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

executor = ThreadPoolExecutor(max_workers=4)

# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_file):
    all_texts = []
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Splitting the text into smaller chunks using RecursiveCharacterTextSplitter
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = txt_splitter.split_text(text)
    return texts

# Function to extract text from CSV files
def extract_text_from_csv(csv_file):
    df = pd.read_csv(BytesIO(csv_file))
    all_text = []
    for col in df.columns:
        all_text.extend(df[col].astype(str).tolist())
    return all_text

# Function to extract text from JSON files
def extract_text_from_json(json_file):
    data = json.loads(json_file)
    
    def extract_text(data):
        texts = []
        if isinstance(data, dict):
            for key, value in data.items():
                texts.extend(extract_text(value))
        elif isinstance(data, list):
            for item in data:
                texts.extend(extract_text(item))
        else:
            texts.append(str(data))
        return texts
    
    return extract_text(data)

# Function to extract text from TXT files
def extract_text_from_txt(txt_file):
    text = txt_file.decode("utf-8")
    # Split large text into manageable chunks
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    return chunks

# Define the input model for query requests
class QueryRequest(BaseModel):
    query: str

# Upload endpoint for PDFs, CSV, and JSON
@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_extension = file.filename.split(".")[-1].lower()

        # Extract text based on the file type
        if file_extension == "pdf":
            all_texts = extract_text_from_pdfs(BytesIO(contents))
        elif file_extension == "csv":
            all_texts = extract_text_from_csv(contents)
        elif file_extension == "json":
            all_texts = extract_text_from_json(contents)
        elif file_extension == "txt":
            all_texts = extract_text_from_txt(contents)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Embed the text chunks using Hugging Face embeddings
        for i, text in enumerate(all_texts):
            chunk_embedding = embedding_model.embed_query(text)
            index.upsert([(f"chunk-{i}", chunk_embedding, {"text": text})])

        return {"message": "File uploaded and processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize Pinecone retriever
retriever = Pinecone(index=index, embedding=embedding_model.embed_query, text_key='text')

# Initialize the language model
llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.7)

# llm_model = Hu(repo_id='mistralai/Mistral-7B-Instruct-v0.3', max_length=256, temperature=0.4, huggingfacehub_api_token=key,
# max_new_tokens=150,  # Limit token generation for faster responses
# do_sample=True,
# top_p=0.9,
# repetition_penalty=1.1)

def query_mistral(prompt):
    response = huggingface_client.text_generation(
        prompt,
        model="HuggingFaceH4/zephyr-7b-beta",  # ✅ replace with a compatible model
        max_new_tokens=150,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.1
    )
    return response

# Cache for recent responses (simple in-memory cache)
response_cache = {}
cache_lock = threading.Lock()

def get_cached_response(query: str):
    """Get cached response if available"""
    with cache_lock:
        return response_cache.get(query.lower().strip())

def cache_response(query: str, response: str):
    """Cache response with size limit"""
    with cache_lock:
        # Keep cache size manageable
        if len(response_cache) > 100:
            # Remove oldest entries
            keys_to_remove = list(response_cache.keys())[:20]
            for key in keys_to_remove:
                del response_cache[key]
        response_cache[query.lower().strip()] = response

# Initialize Pinecone retriever (lazy initialization)
_retriever = None
_rag_model = None

def get_rag_model():
    global _retriever, _rag_model
    if _retriever is None:
        _retriever = Pinecone(index=index, embedding=embedding_model.embed_query, text_key='text')
        _rag_model = RetrievalQA.from_chain_type(llm=llm, retriever=_retriever.as_retriever())
    return _rag_model


# Route to test basic connection
@app.get('/')
def homePage():
    return {'message': 'Welcome to the HomePage'}

# Route for QA chat
@app.post('/chat')
async def qa_chatbot(req: QueryRequest):
    ques = req.query
    if not ques:
        raise HTTPException(status_code=400, detail="Query failed")
    
    try:
        # answer = rag_model.run(ques)
        cached_response = get_cached_response(ques)
        if cached_response:
            return {"query": ques, "answer": cached_response, "cached": True}
        loop = asyncio.get_event_loop()
        rag_model = get_rag_model()
        answer = await loop.run_in_executor(executor, rag_model.run, ques)

        cache_response(ques, answer)
        return {"query": ques, "answer": answer, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot without RAG, just using the LLM model
@app.post('/llm_bot')
async def llm_bot(req: QueryRequest):
    questions = req.query
    if not questions:
        raise HTTPException(status_code=400, detail="Query failed")

    try:
        cached_response = get_cached_response(questions)
        if cached_response:
            return {"query": questions, "answer": cached_response, "cached": True}
        
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(executor, query_mistral, questions)
        cache_response(questions, answer.content if hasattr(answer, 'content') else str(answer))
        
        return {"query": questions, "answer": answer.content if hasattr(answer, 'content') else str(answer), "cached": False}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health_check():
    return {"status": "healthy", "cache_size": len(response_cache)}

# Clear cache endpoint
@app.post('/clear_cache')
def clear_cache():
    global response_cache
    with cache_lock:
        response_cache.clear()
    return {"message": "Cache cleared successfully"}


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
