from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uvicorn
import pandas as pd
import json
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
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Pinecone API keys and index
pc = ps(api_key=os.getenv("Pinecone_api_key"))
key = os.getenv("huggingface_api_key")

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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Initialize Hugging Face embeddings using Mistral-7B-Instruct-v0.3 model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

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
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Set up the RAG model using the retriever
rag_model = RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_retriever())

# Initialize Hugging Face endpoint for Mistral-7B-Instruct-v0.3 model
llm_model = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', max_length=128, temperature=0.4, huggingfacehub_api_token=key)

# Route to test basic connection
@app.get('/')
def homePage():
    return {'message': 'Welcome to the HomePage'}

# Route for QA chat
@app.post('/chat')
def qa_chatbot(req: QueryRequest):
    ques = req.query
    if not ques:
        raise HTTPException(status_code=400, detail="Query failed")
    
    try:
        answer = rag_model.run(ques)
        return {"query": ques, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot without RAG, just using the LLM model
@app.post('/llm_bot')
def llm_bot(req: QueryRequest):
    questions = req.query
    if not questions:
        raise HTTPException(status_code=400, detail="Query failed")
    
    try:
        answer = llm_model.invoke(questions)
        return {"query": questions, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CORS preflight handler
@app.options("/chat")
def options_handler():
    return JSONResponse(content={"message": "Options Request OK"}, status_code=200)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
