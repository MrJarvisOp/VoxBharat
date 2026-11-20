# VoxBharat

Low-latency vernacular RAG and voice assistant for Indian government welfare schemes, built with Python, FastAPI, Streamlit, and Groq.

## Features

- **PDF Ingestion (`src/ingest.py`):** Extracts text with source and page metadata from scheme PDFs and saves chunked data to `chroma_db/chunks.json`.
- **Cross-Lingual BM25 Retrieval & Grounded Generation (`src/rag_llm.py`):** Expands Hindi/Hinglish queries into search keywords, ranks chunks using pure-Python BM25 scoring, filters out non-chat audio models when resolving Groq chat models, and generates grounded Hindi answers with page citations.
- **Voice Pipeline (`src/voice.py`):** Transcribes uploaded audio via Groq Whisper (`whisper-large-v3-turbo`), runs grounded RAG, and synthesizes Hindi speech replies using `gTTS`.
- **REST API (`src/api.py`):** Provides `/health`, `/ask`, `/voice-ask`, and `/audio/{filename}` endpoints.
- **Web UI (`app.py`):** Interactive Streamlit interface supporting both Text Query and Voice Query workflows with citations and audio playback.

## Setup

1. Create and activate a virtual environment, then install dependencies:
   ```powershell
   python -m pip install -r requirements.txt