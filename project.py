# Developing a QA application based on LLM by implementing RAG Architecture and deploying it a web-application using streamlit
# loading the environment_variables

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

# page by page, extracting the text from the pdf and stored as a list of Document objects

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

from langchain.text_splitter import RecursiveCharacterTextSplitter

# splitting the data into chunks based on separators like \n\n, \n, \s

def split_data(pdf_in_doc) :
    r_splitter = RecursiveCharacterTextSplitter(chunk_size= 512, chunk_overlap = 100)
    split_pdf = r_splitter.split_text(pdf_in_doc)
    return split_pdf

from google import generativeai as google_genai
import os

#Leveraging Google's Embedding model
google_genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model= 'models/embedding-001')

# convert the chunks of text data to embedding vectors of fixed dimension

from langchain_community.vectorstores import FAISS
def embed_data(pdf_split) :

    vectorindex = FAISS.from_texts(texts = pdf_split, embedding = embeddings)
    return vectorindex


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# setting up the google chat model (llm) to input into the QA_chain of langchain
# Retrieving embedding vectors most similar to the query using FAISS, meta's search algorithm

def get_answer(vectorindex, query) :

    similar_vals = vectorindex.similarity_search(query)
    llm = ChatGoogleGenerativeAI(model = "gemini-pro", temperature= 0.3,
                                 convert_system_message_to_human=True)

    qa_chain = load_qa_chain(llm = llm, chain_type="stuff")
    # Note that load_qa_chain nowhere uses prompts
    response = qa_chain.invoke({'input_documents' : similar_vals, 'question' : query})

    return response['output_text']
    
# Deploying the model as a web-application using streamlit

import streamlit as st

def main() :
    st.set_page_config("PDF Answerer")
    st.header("Welcome!!! Start Generating Answers from your documents")
    user_query = st.text_input('Ask any question from the documents')

    with st.sidebar :
        uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True)
    

    if st.button('Answer') :
            pdf_reader = get_pdf_text(uploaded_files)
            r_split_docs = split_data(pdf_reader)
            r_vectorindex = embed_data(r_split_docs)
            st.write(get_answer(r_vectorindex,user_query))

if __name__ == "__main__" :
    main()

