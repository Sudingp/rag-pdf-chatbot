import streamlit as st

from ragbot.pipeline import RagPipeline, load_models, read_pdf

st.title("📄 PDF RAG Chatbot (No API Version)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


@st.cache_resource
def get_models(reader):
    return load_models(reader)


with st.sidebar:
    st.header("Settings")
    reader = st.radio(
        "Answer model",
        ["extractive", "generative", "llm", "azure"],
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
        answer = rag.ask(question)
        st.write("### Answer:")
        st.write(answer.text)
        st.caption(f"Confidence: {answer.score:.2f}")
        if answer.truncated:
            st.caption("Note: the retrieved context was cut to fit the model's 512-token limit.")
        if answer.trace:
            with st.expander("Why this answer? (evidence, tool calls, guardrails)"):
                st.json(answer.trace)
