import openai
import numpy as np
import faiss
import streamlit as st

# Set your API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Embedding function using text-embedding-ada-002
def get_openai_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002", 
        input=[text]
    )
    return response['data'][0]['embedding']  # Corrected to access the right key in response

# Search chunks function
def search_chunks(query, k, index, chunks_with_meta):
    query_embedding = np.array([get_openai_embedding(query)]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        result = chunks_with_meta[idx].copy()
        result["similarity"] = float(score)
        results.append(result)
    return results

# Answer function using OpenAI API with the chunks and context
def answer_question_openai(query, k, index, chunks_with_meta):
    # Search for the top chunks relevant to the query
    top_chunks = search_chunks(query, k, index, chunks_with_meta)
    
    # Prepare the context text from the top chunks
    context = "\n\n---\n\n".join(
        f"(Page {chunk['page_number']})\n{chunk['chunk_text']}" for chunk in top_chunks
    )
    
    # Prepare the prompt to send to the OpenAI model
    messages = [
        {"role": "system", "content": "Use only the information in the context to answer the question. "
                                     "If the answer is not in the context, respond with: 'Information not found in the document.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ]

    # Get the model's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=300
    )

    # Return both the answer and the context
    return response['choices'][0]['message']['content'].strip(), context
