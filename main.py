import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import ollama
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load OpenAI API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def recursive_chunking(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_text_chunks(chunks, model='nomic-embed-text'):
    embeddings = []
    for chunk in chunks:
        if chunk:
            try:
                embedding = ollama.embeddings(model=model, prompt=chunk)['embedding']
                embeddings.append((chunk, embedding))
            except Exception as e:
                st.error(f"Error embedding chunk: {e}")
    return embeddings

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def query_openai(prompt, context_embeddings):
    if not context_embeddings:
        return "No context embeddings available. Please upload a PDF and try again."

    try:
        query_embedding = ollama.embeddings(model='nomic-embed-text', prompt=prompt)['embedding']
    except Exception as e:
        return f"Error embedding query: {e}"

    similarities = []
    for doc_embedding in context_embeddings:
        try:
            similarity = cosine_similarity(query_embedding, doc_embedding[1])
            similarities.append(similarity)
        except Exception as e:
            st.error(f"Error calculating similarity: {e}")
            similarities.append(-1)  # Default to lowest similarity on error

    # Select the top N relevant chunks (e.g., top 5)
    top_n = 5
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    relevant_chunks = [context_embeddings[i][0] for i in top_indices if similarities[i] != -1]

    if not relevant_chunks:
        return "No relevant context found."

    # Create a concise context
    context = " ".join(relevant_chunks)

    # Simplified RAG, maybe use better chunking for future, do more research on how to implement rag 
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}. {prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error querying OpenAI: {e}"

def main():
    st.set_page_config(layout="wide")

    # Sidebar for file upload and text display
    with st.sidebar:
        st.title("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            pdf_text = read_pdf(uploaded_file)
            chunks = recursive_chunking(pdf_text)
            if chunks:
                embeddings = embed_text_chunks(chunks)
                st.text_area("PDF Content", pdf_text, height=400)
                st.session_state['embeddings'] = embeddings
            else:
                st.session_state['embeddings'] = []

    # Main content
    st.title("Ask Questions about the PDF")
    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        embeddings = st.session_state.get('embeddings', [])
        if uploaded_file is None:
            st.error("Please upload a PDF file first.")
        elif query:
            st.write("Processing your question...")
            response = query_openai(query, embeddings)
            st.write(response)

if __name__ == "__main__":
    main()