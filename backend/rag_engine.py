import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# These imports are included in the 'langchain' package
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore

def get_answer(question, vectorstore):
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question:
    {context}
    
    Question: {input}
    """)
    
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), doc_chain)
    
    response = retrieval_chain.invoke({"input": question})
    return {"result": response["answer"]}