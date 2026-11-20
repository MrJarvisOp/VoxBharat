import os
import uuid
import streamlit as st

from src.rag_llm import STORE_PATH, generate_grounded_answer
from src.voice import run_voice_rag

UPLOAD_DIR = "outputs/uploads"
REPLY_DIR = "outputs/replies"

st.set_page_config(
    page_title="VoxBharat — Vernacular Civic Voice RAG",
    page_icon="🎙️",
    layout="centered",
)

st.title("VoxBharat")
st.caption("Low-latency vernacular voice & text RAG assistant for government welfare schemes")

if not os.path.exists(STORE_PATH):
    st.warning("Chunk store not found. Run `python -m src.ingest` first to index scheme PDFs.")

tab_text, tab_voice = st.tabs(["Text Query", "Voice Query"])

with tab_text:
    query = st.text_area(
        "Ask a question in Hindi, Hinglish, or English",
        placeholder="Example: PM Kisan yojana ke liye kaunse kisan patra hain?",
        height=100,
    )
    top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=8, value=4, key="text_topk")

    if st.button("Get Grounded Answer", type="primary"):
        if not query.strip():
            st.error("Please enter a question first.")
        else:
            with st.spinner("Retrieving relevant sections and generating answer..."):
                result = generate_grounded_answer(query=query.strip(), top_k=top_k)

            st.subheader("Answer")
            st.write(result["answer"])

            if result["sources"]:
                st.subheader("Sources")
                for src in result["sources"]:
                    st.markdown(f"- **{src.get('source', 'unknown')}** — Page `{src.get('page', '?')}`")

with tab_voice:
    audio_file = st.file_uploader(
        "Upload an audio question (.mp3, .wav, .m4a)",
        type=["mp3", "wav", "m4a", "ogg"],
    )

    if st.button("Run Voice RAG", type="primary"):
        if audio_file is None:
            st.error("Please upload an audio file first.")
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            os.makedirs(REPLY_DIR, exist_ok=True)

            req_id = uuid.uuid4().hex[:8]
            ext = os.path.splitext(audio_file.name)[1] or ".mp3"
            in_path = os.path.join(UPLOAD_DIR, f"ui_query_{req_id}{ext}")
            out_path = os.path.join(REPLY_DIR, f"ui_reply_{req_id}.mp3")

            with open(in_path, "wb") as f:
                f.write(audio_file.getbuffer())

            with st.spinner("Transcribing audio, retrieving context, and synthesizing Hindi voice..."):
                result = run_voice_rag(input_audio_path=in_path, output_audio_path=out_path)

            st.subheader("Transcribed Query")
            st.write(result["transcribed_query"])

            st.subheader("Grounded Answer")
            st.write(result["answer"])

            if os.path.exists(result["audio_output"]):
                st.subheader("Voice Reply")
                st.audio(result["audio_output"], format="audio/mp3")

            if result["sources"]:
                st.subheader("Sources")
                for src in result["sources"]:
                    st.markdown(f"- **{src.get('source', 'unknown')}** — Page `{src.get('page', '?')}`")