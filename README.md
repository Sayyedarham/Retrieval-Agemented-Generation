# Retrieval Augmented generation pipeline
A local Retrieval-Augmented Generation (RAG) pipeline that lets you upload documents and ask questions against them with hybrid retrieval.
cc
## Overview

This is a docccccument Q&A system built to:
- ingest files such as PDF, TXT, and DOCX
- split them into chunks
- index them with both vector search and keyword search
- rewrite user queries for better retrieval
- generate answers using a local LLM

The project is split into:
- a **Flask backend** for document processing and query answering
- a **React frontend** for the user interface

## Key Features

- **Multi-format document upload**: PDF, TXT, DOCX
- **Hybrid retrieval**: semantic search with embeddings + keyword search with BM25
- **Query rewriting**: improves search quality before retrieval
- **Local-first setup**: uses Hugging Face embeddings and Ollama for generation
- **Persistent vector storage**: ChromaDB-backed vector store
- **Health endpoint**: quick service check for backend status

## Tech Stack

**Backend**
- Python
- Flask
- Flask-CORS
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- Ollama
- BM25
- NumPy

**Frontend**
- React
- React DOM
- react-scripts
- lucide-react

## How It Works

1. Upload one or more documents.
2. The backend loads the files and splits them into chunks.
3. Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
4. Documents are stored in Chroma for semantic search.
5. BM25 is used alongside vector search for keyword matching.
6. Your query is rewritten for better retrieval.
7. The top results are combined into context.
8. Ollama generates the final answer from the retrieved context.

## Project Structure

```bash
RagProject/
├── backend/
│   ├── App.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
└── README.md
