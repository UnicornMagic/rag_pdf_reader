import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def query_openai(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content.strip()

def main():
    st.set_page_config(layout="wide")

    # Sidebar for file upload and text display
    with st.sidebar:
        st.title("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            pdf_text = read_pdf(uploaded_file)
            st.text_area("PDF Content", pdf_text, height=400)

    # Main content
    st.title("Ask Questions about the PDF")
    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if uploaded_file is None:
            st.error("Please upload a PDF file first.")
        elif query:
            st.write("Processing your question...")
            response = query_openai(f"{pdf_text}\n\n{query}")
            st.write(response)

if __name__ == "__main__":
    main()