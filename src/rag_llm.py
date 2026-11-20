import json
import math
import os
import re
from collections import Counter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STORE_PATH = "chroma_db/chunks.json"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

STOPWORDS = {
    "the", "is", "in", "at", "of", "on", "and", "a", "an", "to", "for",
    "with", "as", "by", "that", "this", "are", "was", "be", "or", "from",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
}


def load_chunks(store_path: str = STORE_PATH) -> list:
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Chunk store not found at '{store_path}'. Run 'python -m src.ingest' first."
        )
    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def expand_query_for_retrieval(client: Groq, query: str, model_name: str) -> str:
    """Generate English search keywords to improve retrieval over English scheme PDFs."""
    try:
        res = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract concise English search keywords and phrases from the user's query "
                        "for searching government welfare scheme documents. Output only keywords."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        keywords = (res.choices[0].message.content or "").strip()
        return f"{query} {keywords}".strip()
    except Exception:
        return query


def retrieve_top_chunks(query: str, chunks: list, top_k: int = 4) -> list:
    """Rank chunks using pure-Python BM25 scoring with fallback to top chunks."""
    q_tokens = tokenize(query)
    if not q_tokens or not chunks:
        return chunks[:top_k]

    doc_tokens_list = [tokenize(item.get("text", "")) for item in chunks]
    num_docs = len(doc_tokens_list)
    avg_doc_len = sum(len(d) for d in doc_tokens_list) / max(num_docs, 1)

    df = Counter()
    for doc_tokens in doc_tokens_list:
        for term in set(doc_tokens):
            df[term] += 1

    k1 = 1.5
    b = 0.75
    scored = []

    for idx, doc_tokens in enumerate(doc_tokens_list):
        if not doc_tokens:
            continue
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0

        for term in q_tokens:
            if term not in tf:
                continue
            idf = math.log((num_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * (doc_len / max(avg_doc_len, 1.0)))
            score += idf * ((freq * (k1 + 1.0)) / denom)

        if score > 0:
            scored.append((score, chunks[idx]))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [item for _, item in scored[:top_k]]
    return chunks[:top_k]


def resolve_model(client: Groq, preferred_model: str) -> str:
    try:
        models = client.models.list()
        available = [m.id for m in getattr(models, "data", []) if getattr(m, "id", None)]
        if preferred_model in available:
            return preferred_model
        for candidate in [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ]:
            if candidate in available:
                return candidate
        if available:
            return available[0]
    except Exception:
        pass
    return preferred_model


def generate_grounded_answer(query: str, top_k: int = 4) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in the .env file.")

    chunks = load_chunks()
    client = Groq(api_key=api_key)
    model_name = resolve_model(client, GROQ_MODEL)

    expanded_query = expand_query_for_retrieval(client, query, model_name)
    matched = retrieve_top_chunks(expanded_query, chunks, top_k=top_k)

    if not matched:
        return {
            "answer": "माफ़ कीजिए, उपलब्ध दस्तावेज़ों में इस प्रश्न से संबंधित कोई सत्यापित जानकारी नहीं मिली।",
            "sources": [],
        }

    context_blocks = []
    sources = []
    for idx, ch in enumerate(matched, start=1):
        meta = ch.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_blocks.append(f"[Source {idx}: {src}, Page {page}]\n{ch['text']}")
        sources.append({"source": src, "page": page})

    context_str = "\n\n---\n\n".join(context_blocks)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are VoxBharat, a helpful government welfare scheme assistant. "
                    "Answer strictly using the provided CONTEXT. If the answer is not in the context, "
                    "clearly state that you do not have verified information on it and do not invent details. "
                    "Keep the response concise in 3-4 simple, conversational Hindi sentences."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}",
            },
        ],
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
    }


if __name__ == "__main__":
    test_query = "What are the eligibility criteria and cut-off date for this scheme?"
    result = generate_grounded_answer(test_query)
    print("Answer:\n", result["answer"])
    print("\nSources Used:", result["sources"])