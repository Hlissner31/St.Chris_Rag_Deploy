import streamlit as st
import pickle
import numpy as np
import faiss
from openai import OpenAI
import openai
from utils import get_openai_embedding, search_chunks, answer_question_openai

openai.api_key = st.secrets["openai"]["api_key"]

# Load pre-built FAISS index and metadata
index = faiss.read_index("faiss_index.faiss")
with open("chunks_with_meta.pkl", "rb") as f:
    chunks_with_meta = pickle.load(f)

st.title("📚 RAG Q&A Assistant")

query = st.text_input("Ask a question about the document")

if st.button("Submit") and query:
    answer, context = answer_question_openai(query, k=5, index=index, chunks_with_meta=chunks_with_meta)
    st.markdown("### 💬 Answer")
    st.write(answer)
    st.markdown("## 🔍 Debug")
    st.write("Top chunks context used:")
    st.code(context)

