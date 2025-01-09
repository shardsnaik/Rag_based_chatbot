from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import uvicorn
import os
from dotenv import load_dotenv
import openai
import tempfile

import chromadb
from langchain.vectorstores import Chroma
load_dotenv()
app = FastAPI()
# Ensure OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from different file formats
def extract_text_from_data(file_extension, data):
    if file_extension == "csv":
        df = pd.read_csv(BytesIO(data))
        all_texts = df["text"].tolist()
        return all_texts

    elif file_extension == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(data)
            tmp_file.flush()
            pdf_loader = PyPDFLoader(tmp_file.name)
            documents = pdf_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            return [doc.page_content for doc in split_docs]

    elif file_extension == "txt":
        text = data.decode("utf-8")
        # Split large text into manageable chunks
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        return chunks

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        file_content = await file.read()
        file_extension = file.filename.split(".")[-1].lower()
        
        # Extract text from the uploaded file
        all_texts = extract_text_from_data(file_extension, file_content)

        # Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
        embedded_texts = embeddings.embed_documents(all_texts)
# Create Chroma vectorstore with embeddings
        client = chromadb.Client()
        collection = client.create_collection("my_collection")  # Create a collection in Chroma
        collection.add(documents=all_texts, metadatas=[{}]*len(all_texts), embeddings=embedded_texts)
          # Set up the retriever
        vectorstore = Chroma(collection=collection)




        # Set up the retriever
        retriever = vectorstore.as_retriever()

        return {"message": "Upload successful. Vectorstore created."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)