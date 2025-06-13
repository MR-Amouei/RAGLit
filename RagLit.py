import os
import streamlit as st
import fitz
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=True)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def query_ollama(question, context, model="phi4-mini:3.8b"):
    prompt = f"You are a helpful assistant. Based solely on the context below, answer the user's question as clearly and confidently as possible.:\n\nContext:\n{context}\n\nQuestion:\n{question}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

st.set_page_config(page_title="PDF to RAG", layout="centered")

model = load_embedding_model()

st.title("📄 PDF Chunking for RAG")
st.markdown("Upload a PDF file. We'll extract the text and split it into clean chunks for Retrieval-Augmented Generation.")

uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("🧠 Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Text successfully extracted.")
    st.text_area("📝 Preview:", text[:1000], height=200)

    with st.spinner("🔪 Splitting into chunks..."):
        chunks = chunk_text(text)

    st.success(f"📦 Split into {len(chunks)} chunks.")

    selected_chunk = st.slider("📍 Preview a chunk:", 0, len(chunks)-1, 0)
    st.text_area(f"📄 Chunk {selected_chunk + 1}:", chunks[selected_chunk], height=250)

    with st.spinner("🔐 Embedding chunks..."):
        embeddings = embed_chunks(chunks)
    st.success(f"✅ {len(embeddings)} embeddings created.")

    st.markdown("---")
    question = st.text_input("❓ Ask a question about the PDF:")

    if question:
        with st.spinner("🔍 Searching relevant chunks..."):
            question_embedding = model.encode([question])[0]
            similarities = cosine_similarity([question_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-5:][::-1]
            top_chunks = [chunks[i] for i in top_indices]
            context = "\n\n".join(top_chunks)

        st.markdown("### 🔝 Top relevant chunks:")
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk[:400]}...")

        with st.spinner("💬 Querying local LLM..."):
            answer = query_ollama(question, context)

        st.markdown("### ✅ Answer:")
        st.write(answer)
