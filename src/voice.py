import os
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from src.rag_llm import generate_grounded_answer

load_dotenv()

ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3-turbo")


def transcribe_audio(audio_path: str) -> str:
    """Convert a speech audio file to text using Groq Whisper API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in the .env file.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file.read()),
            model=ASR_MODEL,
            response_format="text",
        )
    return str(transcription).strip()


def synthesize_speech(text: str, output_path: str = "outputs/response.mp3", lang: str = "hi") -> str:
    """Convert text into an MP3 audio file using gTTS."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return output_path


def run_voice_rag(input_audio_path: str, output_audio_path: str = "outputs/response.mp3") -> dict:
    """Run end-to-end voice RAG: Audio -> Text -> Grounded RAG -> Hindi Audio."""
    query_text = transcribe_audio(input_audio_path)
    rag_result = generate_grounded_answer(query_text)
    audio_file = synthesize_speech(rag_result["answer"], output_path=output_audio_path)

    return {
        "transcribed_query": query_text,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "audio_output": audio_file,
    }


if __name__ == "__main__":
    sample_query = "PM Kisan yojana ke liye kaunse kisan patra hain?"
    sample_audio = synthesize_speech(sample_query, output_path="outputs/test_query.mp3")
    print("Generated sample query audio at:", sample_audio)

    result = run_voice_rag(sample_audio, output_audio_path="outputs/test_answer.mp3")
    print("Transcribed Query:", result["transcribed_query"])
    print("Grounded Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Saved Answer Audio:", result["audio_output"])