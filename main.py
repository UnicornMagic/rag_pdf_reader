import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import ollama
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import uuid

# Load OpenAI API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Chroma DB client
chroma_client = chromadb.Client()

# Create or load a collection
collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

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

def generate_tags(embeddings, top_n=5):
    top_chunks = [embedding[0] for embedding in embeddings[:top_n]]
    combined_text = " ".join(top_chunks)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate tags for the following text."},
                {"role": "user", "content": combined_text}
            ]
        )
        tags = response.choices[0].message.content.strip()  # Updated to access the content correctly
        return tags.split(', ')
    except Exception as e:
        st.error(f"Error generating tags: {e}")
        return []

def store_embeddings(pdf_name, embeddings, tags):
    tags_str = ', '.join(tags)  # Convert the list of tags to a single string
    for chunk, embedding in embeddings:
        doc_id = str(uuid.uuid4())
        collection.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{"pdf_name": pdf_name, "tags": tags_str}],
            embeddings=[embedding]
        )

def check_pdf_exists(pdf_name):
    results = collection.get(
        where={"pdf_name": pdf_name}
    )
    return len(results['documents']) > 0

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

def query_embeddings(query):
    try:
        query_embedding = ollama.embeddings(model='nomic-embed-text', prompt=query)['embedding']
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        st.write("Debug: Results from Chroma DB query:", results)  # Debugging information

        if 'results' not in results:
            st.error("Error: 'results' key not found in query response.")
            return []
        
        return [(result['documents'][0], result['embeddings'][0]) for result in results['results']]
    except Exception as e:
        st.error(f"Error querying embeddings: {e}")
        return []

def query_pdf_tags(pdf_name):
    try:
        results = collection.get(
            where={"pdf_name": pdf_name}
        )
        if 'metadatas' not in results or len(results['metadatas']) == 0:
            st.error("No metadata found for this PDF.")
            return []
        
        tags = []
        for metadata in results['metadatas']:
            tags.extend(metadata.get('tags', "").split(', '))
        
        return list(set(tags))  # Return unique tags
    except Exception as e:
        st.error(f"Error querying PDF tags: {e}")
        return []

def main():
    st.set_page_config(layout="wide")

    # Sidebar for file upload and text display
    with st.sidebar:
        st.title("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            pdf_name = uploaded_file.name
            if check_pdf_exists(pdf_name):
                st.warning("This PDF is already uploaded.")
            else:
                pdf_text = read_pdf(uploaded_file)
                chunks = recursive_chunking(pdf_text)
                if chunks:
                    embeddings = embed_text_chunks(chunks)
                    tags = generate_tags(embeddings)
                    store_embeddings(pdf_name, embeddings, tags)
                    st.success("PDF uploaded and processed successfully.")
                else:
                    st.error("Failed to chunk the PDF text.")

    # Main content
    st.title("Ask Questions about the PDF")
    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if query:
            st.write("Processing your question...")
            pdf_name = uploaded_file.name if uploaded_file is not None else ""
            tags = query_pdf_tags(pdf_name)
            if "what does this PDF contain?" in query.lower() or "what does the book talk about?" in query.lower():
                if tags:
                    st.write(", ".join(tags))
                else:
                    st.write("No relevant tags found for this PDF.")
            else:
                results = query_embeddings(query)
                if results:
                    response = query_openai(query, results)
                    st.write(response)
                else:
                    st.write("No relevant context found in the database.")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()