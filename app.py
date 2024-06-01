import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import LanceDB
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlaps=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding=001")
    vector_store=LanceDB.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("lance_index")

def get_conversational_chain():
    prompt_template=""""
    You have to answer in as much detail as possible 
    from the provided content, and make sure to provide every 
    possible detail. if the answer is not given in the 
    provided content , just reply with " The answer is not 
    available in the content provided by you",
    just dont give wrong answer in any case.\n
    Context=\n{content}?\n
    Question=\n{question}\n
    Answer: 
    """ 
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["content","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    new_db=LanceDB.load_loacl("lance_index",embeddings)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response=chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("chat pdf")
    st.header("chat with multiple pdf using gemini")
    user_question=st.text_input("ask a que from the PDF file")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF files and click on the submit & process.")
        if st.button("submit & process"):
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
if __name__=="__main__":
    main()  