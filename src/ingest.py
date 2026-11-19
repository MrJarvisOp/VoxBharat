import json
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
STORE_PATH = "chroma_db/chunks.json"


def load_pdf_pages(data_dir: str):
    pages = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "text": text,
                        "metadata": {"source": filename, "page": idx + 1},
                    })
    return pages


def ingest_pdfs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)

    pages = load_pdf_pages(DATA_DIR)
    if not pages:
        print(f"No readable PDFs found in '{DATA_DIR}'. Please add at least one PDF first.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    records = []
    for p_idx, page in enumerate(pages):
        splits = splitter.split_text(page["text"])
        for c_idx, chunk_text in enumerate(splits):
            records.append({
                "id": f"p{p_idx}_c{c_idx}",
                "text": chunk_text,
                "metadata": page["metadata"],
            })

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Success: Saved {len(records)} chunks from {len(pages)} pages into '{STORE_PATH}'.")


if __name__ == "__main__":
    ingest_pdfs()