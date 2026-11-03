import streamlit as st

from ragbot import web_guard as guard
from ragbot.pipeline import RagPipeline, load_models, read_pdf

st.title("📄 PDF RAG Chatbot")

# On a public deployment, an optional password keeps strangers from using up the LLM key.
if guard.password_required() and not st.session_state.get("unlocked"):
    attempt = st.text_input("Password", type="password")
    if attempt:
        if guard.password_ok(attempt):
            st.session_state["unlocked"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()

uploaded_file = st.file_uploader(f"Upload PDF (max {guard.upload_limit_mb()} MB)", type="pdf")


@st.cache_resource
def get_models(reader):
    return load_models(reader)


with st.sidebar:
    st.header("Settings")
    reader = st.radio(
        "Answer model",
        list(guard.READERS),
        index=list(guard.READERS).index(guard.default_reader()),
        format_func=lambda r: {"extractive": "DistilBERT (copies a span)",
                               "generative": "Flan-T5 (writes the answer)",
                               "llm": "Hosted LLM: Gemini / Groq / Ollama (see .env)",
                               "azure": "Azure OpenAI GPT (needs API key)"}[r],
    )
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 0, step=50)
    min_score = st.slider(
        "Minimum answer confidence (0 = always answer)", 0.0, 1.0, 0.0, step=0.05
    )

if uploaded_file:
    if uploaded_file.size > guard.upload_limit_mb() * 1024 * 1024:
        st.error(f"That PDF is larger than {guard.upload_limit_mb()} MB.")
        st.stop()
    try:
        embed, qa = get_models(reader)
    except RuntimeError as err:  # e.g. Azure settings missing
        st.error(str(err))
        st.stop()
    # Flan-T5 reads at most 512 tokens, so it gets smaller, fewer chunks.
    size, k = {"extractive": (900, 6), "generative": (500, 3), "azure": (900, 4), "llm": (900, 4)}[reader]
    rag = RagPipeline(
        embed, qa, chunk_size=size, k=k, overlap=overlap,
        min_score=min_score if min_score > 0 else None,
    ).index_text(read_pdf(uploaded_file))

    question = st.text_input("Ask a question")

    if question:
        # Streamlit reruns the script on every click, so count each question once.
        if question != st.session_state.get("last_question"):
            asked = st.session_state.get("asked", 0)
            if asked >= guard.question_limit():
                st.warning(f"This session has reached its limit of {guard.question_limit()} questions. "
                           "Open a new browser tab to continue.")
                st.stop()
            st.session_state["asked"] = asked + 1
            st.session_state["last_question"] = question
        try:
            answer = rag.ask(question)
        except Exception as err:  # never show a traceback on the public page
            print(f"rag.ask failed: {type(err).__name__}: {err}", flush=True)  # goes to Cloud Run logs
            st.error("Sorry, the answer model failed on this question. Please try again, "
                     "or pick another answer model in the sidebar.")
            st.stop()
        st.write("### Answer:")
        st.write(answer.text)
        st.caption(f"Confidence: {answer.score:.2f}")
        if answer.truncated:
            st.caption("Note: the retrieved context was cut to fit the model's 512-token limit.")
        if answer.trace:
            with st.expander("Why this answer? (evidence, tool calls, guardrails)"):
                st.json(answer.trace)
