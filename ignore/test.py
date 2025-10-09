# from fastapi import FastAPI, UploadFile, File, HTTPException
# from io import BytesIO
# import pandas as pd
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# import uvicorn

# app = FastAPI()

# # Function to extract text from different file formats
# def extract_text_from_data(file_extension, data):
#     if file_extension == "csv":
#         df = pd.read_csv(BytesIO(data))
#         all_texts = df["text"].tolist()
#         return all_texts
    

#     elif file_extension == "pdf":
#         pdf_loader = PyPDFLoader(BytesIO(data))
#         documents = pdf_loader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         split_docs = text_splitter.split_documents(documents)
#         return [doc.page_content for doc in split_docs]
    

#     elif file_extension == "txt":
#         text = data.decode("utf-8")
#         return [text]
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file format.")
    


# def begin(file_content):
#     try:
#         # Read uploaded file
        
#         file_extension = file_content.filename.split(".")[-1].lower()
        
#         # Extract text from the uploaded file
#         all_texts = extract_text_from_data(file_extension, file_content)

#         # Generate embeddings using SentenceTransformer
#         embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         embeddings = embedding_model.encode(all_texts)

#         # Create a FAISS vectorstore
#         vectorstore = FAISS.from_documents(all_texts, embeddings)


#         # Set up the retriever
#         retriever = vectorstore.as_retriever()

#         return {"message": "Upload successful. Vectorstore created."}

#     except Exception as e:
#         print(e)
#         raise e


# begin('About Yardstick.pdf')




####################################################################################
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
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as ps, ServerlessSpec
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Pinecone API keys and index
pc = ps(api_key=os.getenv("Pinecone_api_key"))
key = os.getenv("huggingface_api_key")

spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "multi-qa-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=spec)
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Initialize Hugging Face embeddings using Mistral-7B-Instruct-v0.3 model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Extract text from PDF
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

# Extract text from CSV
def extract_text_from_csv(csv_file):
    df = pd.read_csv(BytesIO(csv_file))
    all_text = []
    for col in df.columns:
        all_text.extend(df[col].astype(str).tolist())
    return all_text

# Extract text from JSON
def extract_text_from_json(json_file):
    data = json.loads(json_file)
    return [entry["text"] for entry in data]


def extract_text_from_txt(txt_file):
    text = txt_file.decode("utf-8")
    # Split large text into manageable chunks
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    return chunks

# Define the input model
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

# Initialize retriever using Pinecone
retriever = Pinecone(index=index, embedding=embedding_model.embed_query, text_key="text")
# llm = HuggingFaceEndpoint(repo_id ='mistralai/Mistral-7B-Instruct-v0.3', max_length=128, temperature=.4, huggingfacehub_api_token=key)
# Load the LLaMA model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with the specific LLaMA model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text-generation pipeline
generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_length=512)

# Wrap it with LangChain
llm = HuggingFacePipeline(pipeline=generator)
# Use RetrievalQA for question answering
# rag_model = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever.as_retriever()
# )
rag_model = RetrievalQA(llm=llm, retriever=retriever)
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
        # answer = llm.invoke(ques)
        answer = rag_model.run(ques)
        return {"query": ques, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CORS preflight handler
@app.options("/chat")
def options_handler():
    return JSONResponse(content={"message": "Options Request OK"}, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
