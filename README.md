# PDF RAG Chatbot (Offline RAG Project)

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions from uploaded PDF documents using local transformer models and vector search.

## Features
- PDF text extraction
- Text chunking
- SentenceTransformer embeddings
- FAISS vector similarity search
- Transformer QA model
- Interactive Streamlit UI
- Runs fully offline (no API key required)

## Tech Stack
Python, Streamlit, FAISS, SentenceTransformers, Transformers

## How It Works
1. Extract text from PDF
2. Split text into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS vector index
5. Retrieve most relevant chunks
6. QA model generates grounded answers

## Run Locally
python -m streamlit run app.py