import streamlit as st
import pickle
import numpy as np
import faiss
from openai import OpenAI
import openai
from utils import get_openai_embedding, search_chunks, answer_question_openai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def log_to_google_sheet(question, answer):
    # Setup scope and credentials
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)

    # Open your sheet (replace with your actual spreadsheet name or ID)
    sheet = client.open("https://docs.google.com/spreadsheets/d/1JYj14H1_fBQ9rKoeDAAZurNePnnYwyWWH7Na8Y11-Vo/edit?gid=0#gid=0").sheet1

    # Append the row
    sheet.append_row([datetime.now().isoformat(), question, answer])


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

    # Log to Google Sheets
    log_to_google_sheet(query, answer)

