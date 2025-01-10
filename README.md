# RAG-Based Chatbot with gpt-3.5-turbo Model
# https://conversationaimodel.netlify.app/
## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to build an intelligent chatbot capable of answering user queries based on a custom document database. The chatbot uses Pinecone for document retrieval and integrates Openai’s `gpt-3.5-turbo` model for generating natural language responses.

The project leverages LangChain’s seamless integration with multiple components like retrievers and LLMs, ensuring an efficient and modular architecture.

---

### How to Use the Model

1. **Initial Model Selection**:
   - Upon accessing the app, you will see an introduction page presenting two model options:
     - **RAG Model**
     - **LLM Model (e.g., GPT)**
   - Note: The user interface for the LLM model is not yet available. However, you can use the API service provided on the GitHub page for accessing this model.
   - Proceed by selecting the **RAG Model**.

2. **Uploading a Dataset**:
   - Once on the RAG model page, you will be prompted to upload your dataset.
   - Supported formats: JSON, TXT, PDF, CSV.
   - Upload your dataset and wait for it to process.
   - A success message will appear at the top of the page when the dataset is fully processed and ready.
     - **Important**: Processing may take some time due to the backend API being deployed on a free instance. Please be patient.

3. **Using the RAG Model**:
   - After a successful upload, you can begin interacting with the RAG model.
   - Enter your queries in the chat interface, and the model will retrieve relevant information and provide accurate answers.

4. **Downloading Chat Conversations**:
   - You can download the complete chat history by clicking the download icon at the top of the page.

5. **Further Documentation**:
   - For more details, check the supplementary GitHub PDF provided in the repository.


## Features

- **RAG Architecture**: Combines retrieval-based and generative approaches to improve the accuracy and relevance of responses.

- **OpenAI LLM Integration**: Uses Open AI's `gpt-3.5-turbo` model for generating coherent and context-aware responses.

- **Pinecone Vector Store**: Handles vector storage and retrieval for documents, enabling efficient and scalable similarity searches.

-- **AIP Integration and Downloadable**: Offers both an API and a user interface for interaction. Chat conversations can be downloaded for reference.

- **Error Handling**: Includes robust error handling to manage invalid queries and system errors.

--**Ready with all data**:Compatible with various data file formats, including JSON, TXT, PDF, and CSV. Automatically processes and ingests data from uploaded files—no manual data entry required.
---

## Project Architecture

1. **Document Retrieval**:
   - Pinecone is used as the vector database to store document embeddings.
   - Queries are embedded and matched with stored embeddings for relevant document retrieval.

2. **Language Model**:
   - The Open AI's `gpt-3.5-turbo` model generates natural language responses based on retrieved documents.

3. **RAG Pipeline**:
   - Combines retrieved document context with the generative capabilities of the Open-Ai model to produce accurate answers.

4. **API Endpoints**:
   - `/`: Basic connectivity test.
   - `/chat`: Accepts a user query and returns a chatbot response.

---

## Installation

### Prerequisites

- Python 3.9 or later
- Pinecone account and API key
- Open AI account and API key
- FastAPI

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-chatbot
   cd rag-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   - Create a `.env` file in the root directory with the following variables:
     ```env
     Open-AI_API_KEY=your_Open-AI_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_ENVIRONMENT=your_pinecone_environment
     ```

4. Initialize Pinecone:
   ```python
   import pinecone
   pinecone.init(api_key="your_pinecone_api_key", environment="your_pinecone_environment")
   ```

5. Run the application:
   ```bash
   uvicorn app:app --reload
   ```

---

## Usage

1. Access the home page to verify the connection:
   - Open your browser and navigate to `http://127.0.0.1:8000/`.

2. Use the `/chat` endpoint to interact with the chatbot:
   - Send a POST request with a query:
     ```json
     {
       "query": "What is Retrieval-Augmented Generation?"
     }
     ```
   - Response:
     ```json
     {
       "query": "What is Retrieval-Augmented Generation?",
       "answer": "Retrieval-Augmented Generation (RAG) is a method that combines information retrieval with generative models..."
     }
     ```

---

## Example Code Snippets

### Embedding Model

```python
from langchain_community.embeddings import OpenAIEmbeddings

Openai_API_KEY = "your_openai_api_key"
embedding_model = OpenaiEmbeddings(model="embed-multilingual-v2.0", api_key=Openai_API_KEY)
```

### Retriever Initialization

```python
from langchain.vectorstores import Pinecone

retriever = Pinecone(
    index=index,
    embedding=embedding_model.embed_query,
    text_key="text"
)
```

### LLM Initialization

```python
from langchain.llms import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=Openai_API_KEY
)
```

### RAG Pipeline

```python
from langchain.chains import RetrievalQA

rag_model = RetrievalQA(
    llm=llm,
    retriever=retriever.as_retriever()
)
```

---

## Known Issues

1. **Deprecation Warning**:
   - Some LangChain components like `Openai` have been deprecated in newer versions.
   - Use updated packages from `langchain-openai` instead.

2. **Embedding Initialization**:
   - Errors related to `user_agent` in `OpenAIEmbeddings` might occur due to library updates.
   - Check compatibility and update packages accordingly.

3. **Pinecone Warnings**:
   - Passing a callable to `embedding` is deprecated. Use `Embeddings` objects instead.


---

---

