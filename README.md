# DocChatAI: RAG Chatbot & Report Generator

A Retrieval-Augmented Generation (RAG) application built with:
- Streamlit
- LangChain
- Gemini API, Model: Gemini-2.5-Flash
- FAISS
- Sentence Transformers

## Features
- Upload PDFs and ask questions
- Context-aware answers using RAG
- Generate structured summary reports
- Download reports as text files

## Tech Stack
- Python 3.10
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings

## Evaluation

Retrieval performance was evaluated using a curated set of
question–document pairs on user-uploaded PDFs.

- **Top-3 Retrieval Accuracy:** 100%
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Store:** FAISS
- A query is considered correct if the expected document appears
  among the top-3 retrieved chunks.

- Gemini API

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
