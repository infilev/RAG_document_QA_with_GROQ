import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_document import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import PyPDFDirectoryLoader

# from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY")
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model = "Llama3-8b-8192", groq_api_key = groq_api_key)

# Chat prompt template

prompt = ChatPromptTemplate.from_template(

   """
   Answer the questions based on the provided context only.
   Please provide the most accuarte resposne based on the questions
   <context>
   {context}
   <context>
   Question: {input}

   """

)

## Now we perform data ingestion, vectorstores, embeddings with session
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectorLoader("research_papers") ## data ingestion
        st.session_state.docs = st.session_state.loader.load()  ## Documnet loading
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[50])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, embedding=st.session_state.embeddings) ## coverting into vectors

st.title("RAG Document Q&A with Groq and Llama3-8b-8192")


prompt = st.text_input("Enter your query from research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time    

if user_prompt:
    document_chain =create_stuff_documents_chain(llm,prompt) # creates a chain for passing a lsit of documents to the model (in {context})
    retriever = st.session_state.vectors.as_retriever()   # creating a retriever
    retriever_chain = create_retriever_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retriever_chain.invoke({"input":user_prompt})
    print(f"Response time: {time.process_time()- start}")

    st.write(response['answer'])


    ## With  streamlit lets see what is similar search
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("________________________________________________________________")


## NOTE:
## Ollama Embedding will take a lot of time to convert the documents into vector so make sure to use OpenAI.

