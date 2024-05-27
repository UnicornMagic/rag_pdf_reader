import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
import uuid

# Load OpenAI API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize embedding model
ollama_emb = OllamaEmbeddings(model="nomic-embed-text")


# Initialize Chroma DB client
chroma_client = chromadb.Client()

# Create or load a collection for embeddings and metadata
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

def embed_text_chunks(chunks):
    embeddings_chunks = ollama_emb.embed_documents(chunks)
    return embeddings_chunks

# def store_embeddings(pdf_name, chunk_embedding_pairs):
#     documents = [chunk for chunk, _ in chunk_embedding_pairs]
#     embeddings = [embedding for _, embedding in chunk_embedding_pairs]
#     ids = [str(uuid.uuid4()) for _ in chunk_embedding_pairs]
#     metadatas = [{"pdf_name": pdf_name} for _ in chunk_embedding_pairs]
    
#     collection.add(
#         documents=documents,
#         ids=ids,
#         metadatas=metadatas,
#         embeddings=embeddings
#     )

# Refactored version not sure if better: 
def store_embeddings(pdf_name, chunks, embeddings_chunks):
    ids = []
    metadatas = []

    for _ in range(len(chunks)):
        ids.append(str(uuid.uuid4()))
        metadatas.append({"pdf_name": pdf_name})
    try:
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings_chunks
        )
    except Exception as e:
        st.error(f"Error storing embeddings: {e}")

def check_pdf_exists(pdf_name):
    results = collection.get(
        where={"pdf_name": pdf_name}
    )
    return len(results['documents']) > 0

def get_uploaded_pdfs():
    results = collection.get()
    pdf_names = {metadata["pdf_name"] for metadata in results["metadatas"]}
    return list(pdf_names)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def query_openai(prompt):
    try:
        query_embedding = ollama_emb.embed_query(prompt)
    except Exception as e:
        return f"Error embedding query: {e}"

    # Retrieve all embeddings from the collection, including embeddings
    try:
        results = collection.get(include=["documents", "embeddings", "metadatas"])
    except Exception as e:
        return f"Error retrieving embeddings from database: {e}"

    # Check if the results contain documents and embeddings
    if not results or 'documents' not in results or 'embeddings' not in results:
        return "No relevant context found."

    documents = results['documents']
    embeddings = results['embeddings']

    if not documents or not embeddings:
        return "No relevant context found."

    # Convert embeddings to the correct format (list of lists of floats)
    # Commented it out but it actually ran faster when uncommented, need further testing 
    # embeddings = [list(map(float, emb)) for emb in embeddings]

    # Calculate cosine similarity between the query embedding and all document embeddings
    similarities = []
    for embedding in embeddings:
        try:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append(similarity)
        except Exception as e:
            st.error(f"Error calculating similarity: {e}")
            similarities.append(-1)  # Default to lowest similarity on error

    # Ensure similarities and documents are numpy arrays for proper indexing
    similarities = np.array(similarities)
    documents = np.array(documents)

    # Select the top N relevant chunks (e.g., top 5)
    top_n = 5
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    relevant_chunks = [documents[i] for i in top_indices if similarities[i] != -1]

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
    
def inspect_stored_data():
    try:
        results = collection.get()
        st.write(f"Inspection Results: {results}")
    except Exception as e:
        st.error(f"Error inspecting stored data: {e}")

def main():
    st.set_page_config(layout="wide")

    # Sidebar for file upload and text display
    with st.sidebar:
        st.title("Upload Section")

        # Display the list of uploaded PDFs
        st.header("Uploaded PDFs")
        uploaded_pdfs = get_uploaded_pdfs()
        for pdf_name in uploaded_pdfs:
            st.write(pdf_name)

        # File uploader for new PDFs
        st.header("Upload a new PDF")
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
                    store_embeddings(pdf_name, chunks, embeddings)
                    st.success("PDF uploaded and processed successfully.")
                else:
                    st.error("Failed to chunk the PDF text.")

    # Main content
    st.title("Query Section")

    query = st.text_input("Enter your question:")

    if st.button("Submit", key="submit_query"):
        if query:
            st.write("Processing your question...")
            response = query_openai(query)
            st.write(response)
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()