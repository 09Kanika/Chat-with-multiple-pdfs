import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.milvus import Milvus
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Milvus client
milvus_instance = Milvus(api_key=os.getenv("MILVUS_API_KEY"), host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))

# Specify the name of your Milvus collection
collection_name = "multiple-pdf"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)  # Move to the start of the file
        pdf_stream = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if extract_text returns None
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001")
        vectors = [embeddings.embed_document(chunk) for chunk in text_chunks]
        
        # Create Milvus collection if it doesn't exist
        if collection_name not in milvus_instance.list_collections():
            milvus_instance.create_collection(collection_name, dimension=len(vectors[0]))
        
        milvus_instance.insert(collection_name, vectors, chunk_ids=list(range(len(vectors))))
        return collection_name, text_chunks
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        raise





def get_conversational_chain():
    prompt_template = """
    You have to answer in as much detail as possible 
    from the provided content, and make sure to provide every 
    possible detail. If the answer is not given in the 
    provided content, just reply with "The answer is not 
    available in the content provided by you".
    Just don't give a wrong answer in any case.
    \n
    Context:\n{content}\n
    Question:\n{question}\n
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["content", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, index, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="textembedding-gecko-001")
    query_vector = embeddings.embed_document(user_question)
    result = index.query(query_vector, top_k=5, include_metadata=True)
    docs = [match['metadata']['text'] for match in result['matches']]
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with multiple PDFs using Gemini")
    user_question = st.text_input("Ask a question from the PDF file")
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on submit & process.", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:  # Check if any files are uploaded
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        index, text_chunks = get_vector_store(text_chunks)
                        st.session_state.index = index
                        st.session_state.text_chunks = text_chunks
                        st.success("Done")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.error("Please upload at least one PDF file.")
    if user_question and 'index' in st.session_state and 'text_chunks' in st.session_state:
        user_input(user_question, st.session_state.index, st.session_state.text_chunks)

if __name__ == "__main__":
    main()
