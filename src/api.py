import os
import shutil
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag_llm import STORE_PATH, generate_grounded_answer
from src.voice import run_voice_rag

app = FastAPI(
    title="VoxBharat API",
    description="Low-latency vernacular RAG and voice backend for civic schemes.",
    version="0.1.0",
)

UPLOAD_DIR = "outputs/uploads"
REPLY_DIR = "outputs/replies"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question in Hindi, Hinglish, or English")
    top_k: int = Field(default=4, ge=1, le=10)


class SourceItem(BaseModel):
    source: str
    page: int | str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class VoiceAskResponse(BaseModel):
    transcribed_query: str
    answer: str
    sources: list[SourceItem]
    audio_url: str


@app.get("/health")
def health_check() -> dict:
    """Return service health and whether the chunk store exists."""
    return {
        "status": "ok",
        "chunk_store_ready": os.path.exists(STORE_PATH),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """Answer a text query using grounded BM25 retrieval and LLM."""
    try:
        result = generate_grounded_answer(query=payload.query, top_k=payload.top_k)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceItem(**src) for src in result["sources"]],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc


@app.post("/voice-ask", response_model=VoiceAskResponse)
async def voice_ask(file: UploadFile = File(...)) -> VoiceAskResponse:
    """Accept an audio file, run ASR + RAG + TTS, and return response metadata and audio URL."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(REPLY_DIR, exist_ok=True)

    req_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    input_path = os.path.join(UPLOAD_DIR, f"query_{req_id}{ext}")
    output_path = os.path.join(REPLY_DIR, f"reply_{req_id}.mp3")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_voice_rag(input_audio_path=input_path, output_audio_path=output_path)
        return VoiceAskResponse(
            transcribed_query=result["transcribed_query"],
            answer=result["answer"],
            sources=[SourceItem(**src) for src in result["sources"]],
            audio_url=f"/audio/{os.path.basename(output_path)}",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice RAG failed: {exc}") from exc
    finally:
        await file.close()


@app.get("/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    """Serve a generated reply MP3 file."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(REPLY_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path=file_path, media_type="audio/mpeg", filename=safe_name)