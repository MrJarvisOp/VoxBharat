import json
import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STORE_PATH = "chroma_db/chunks.json"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def load_chunks(store_path: str = STORE_PATH):
    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))


def retrieve_top_chunks(query: str, chunks: list, top_k: int = 3):
    q_tokens = tokenize(query)
    if not q_tokens:
        return chunks[:top_k]

    scored = []
    for item in chunks:
        c_tokens = tokenize(item.get("text", ""))
        if not c_tokens:
            continue
        overlap = len(q_tokens.intersection(c_tokens))
        score = overlap / (len(q_tokens) ** 0.5)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0] or [item for _, item in scored[:top_k]]


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


def generate_grounded_answer(query: str, top_k: int = 3) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in the .env file.")

    chunks = load_chunks()
    matched = retrieve_top_chunks(query, chunks, top_k=top_k)

    context_blocks = []
    for idx, ch in enumerate(matched, start=1):
        meta = ch.get("metadata", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_blocks.append(f"[Source {idx}: {src}, Page {page}]\n{ch['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    client = Groq(api_key=api_key)
    model_name = resolve_model(client, GROQ_MODEL)

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

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_query = "What are the eligibility criteria and required documents for this scheme?"
    answer = generate_grounded_answer(test_query)
    print("Answer:\n", answer)