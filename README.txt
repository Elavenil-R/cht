UNIQ TECHNOLOGIES RAG CHATBOT
Real Retrieval-Augmented Generation (RAG) Chatbot using FastAPI, ChromaDB, and Local AI

────────────────────────────────────────────────────────────

PROJECT OVERVIEW

This project is a real Retrieval-Augmented Generation (RAG) chatbot built for UNIQ Technologies.
The chatbot uses a local AI model together with a vector database to answer questions strictly from the UNIQ Technologies knowledge base.

The system retrieves relevant information from stored knowledge files using embeddings and ChromaDB vector search, then sends only the retrieved context to the local AI model to generate accurate responses.

If the requested information is not present in the knowledge base, the chatbot responds:

Information not available.

No hardcoded company information is used inside the code.

────────────────────────────────────────────────────────────

TECHNOLOGIES USED

Backend Framework:
FastAPI

Vector Database:
ChromaDB (persistent local vector database)

Embedding and Chat Model:
Local AI model running via LM Studio (OpenAI-compatible API)

Language:
Python

Other Libraries:
Requests
NumPy
Pydantic

────────────────────────────────────────────────────────────

PROJECT STRUCTURE

uniq_rag_bot/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       └── chat.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── schemas/
│   │   └── chat.py
│   │
│   └── services/
│       ├── rag.py
│       ├── vectorstore.py
│       ├── embeddings.py
│       ├── chunking.py
│       ├── knowledge.py
│       └── lm_client.py
│
├── knowledge/
│   ├── uniq1.txt
│   ├── uniq2.txt
│   ├── uniq3.txt
│   └── bot_rules.txt
│
├── storage/
│   (Automatically created ChromaDB vector database)
│
├── .env
├── requirements.txt
├── run.bat
└── README.txt

────────────────────────────────────────────────────────────

HOW THE SYSTEM WORKS (RAG PIPELINE)

Step 1:
Knowledge is stored in text files:
uniq1.txt
uniq2.txt
uniq3.txt

Step 2:
The system splits the text into smaller chunks.

Step 3:
Each chunk is converted into embeddings using the local embedding model.

Step 4:
Embeddings are stored in ChromaDB vector database.

Step 5:
When a user asks a question:
• The question is converted into an embedding
• ChromaDB retrieves the most relevant chunks
• Retrieved context is sent to the local AI model

Step 6:
The AI generates a response strictly based on retrieved context.

Step 7:
If no relevant context is found, the system returns:
Information not available.

────────────────────────────────────────────────────────────

INSTALLATION

Step 1: Create virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux / Mac:

python3 -m venv .venv
source .venv/bin/activate

────────────────────────────────────────────────────────────

Step 2: Install dependencies

pip install -r requirements.txt

────────────────────────────────────────────────────────────

Step 3: Start LM Studio Local Server

Open LM Studio
Load chat model
Load embedding model
Start server

Default server address:

http://localhost:1234/v1

────────────────────────────────────────────────────────────

Step 4: Configure environment

Edit .env file:

LM_BASE_URL=http://localhost:1234/v1
CHAT_MODEL=your_chat_model_name
EMBED_MODEL=your_embedding_model_name

────────────────────────────────────────────────────────────

RUNNING THE APPLICATION

Windows:

run.bat

OR

python -m uvicorn app.main:app --reload

────────────────────────────────────────────────────────────

ACCESS API

Health check:

http://127.0.0.1:8000/health

Chat endpoint:

POST
http://127.0.0.1:8000/api/chat

Example request:

{
"question": "What courses are available?"
}

Example response:

{
"answer": "UNIQ Technologies offers courses in Java, Python, Machine Learning, DevOps, and more."
}

────────────────────────────────────────────────────────────

VECTOR DATABASE

Vector database used: ChromaDB

Location:
storage/

Stores:
Embeddings
Text chunks
Metadata

This enables fast and accurate semantic search.

────────────────────────────────────────────────────────────

KEY FEATURES

Real RAG architecture
Persistent vector database
Local AI model (no cloud required)
No hardcoded company data in code
Accurate retrieval-based responses
Strict fallback when information is unavailable
Modular and scalable FastAPI structure

────────────────────────────────────────────────────────────

AUTHOR

Project developed as a Local AI RAG Chatbot for UNIQ Technologies.
