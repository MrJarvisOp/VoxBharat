# VoxBharat

Low-latency vernacular RAG assistant for rural civic accessibility, helping citizens find government welfare scheme eligibility, benefits, and required documents from scheme PDFs in conversational Hindi.

## Current Architecture

To avoid native C-extension (`.pyd`) DLL blocks under strict Windows Application Control policies, the current pipeline uses lightweight Python components:

- **Ingestion (`src/ingest.py`):** Extracts text from PDFs in `data/` using `pypdf`, splits text into overlapping chunks with `RecursiveCharacterTextSplitter`, and saves structured records to `chroma_db/chunks.json`.
- **Retrieval & Generation (`src/rag_llm.py`):** Retrieves relevant chunks from `chroma_db/chunks.json` and generates grounded answers in simple conversational Hindi via Groq (`llama-3.1-8b-instant`).

## Project Structure

```text
vox-bharat/
├── data/               
├── chroma_db/          
├── src/
│   ├── ingest.py       
│   └── rag_llm.py      
├── .env                
├── .gitignore
├── requirements.txt
└── README.md