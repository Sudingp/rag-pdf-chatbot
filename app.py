import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("📄 PDF RAG Chatbot (No API Version)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering",
                           model="distilbert-base-cased-distilled-squad")
    return embed_model, qa_pipeline

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def chunk_text(text, size=900):
    return [text[i:i+size] for i in range(0, len(text), size)]

if uploaded_file:

    embed_model, qa_pipeline = load_models()

    text = read_pdf(uploaded_file)
    chunks = chunk_text(text)

    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    question = st.text_input("Ask a question")

    if question:
        q_emb = embed_model.encode([question])
        D, I = index.search(np.array(q_emb), k=6)

        context = " ".join([chunks[i] for i in I[0]])

        result = qa_pipeline(
            question=question,
            context=context
        )

        st.write("### Answer:")
        st.write(result["answer"])