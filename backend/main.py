import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import process_pdf, get_answer
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    vectorstore = process_pdf(file_path)
    os.remove(file_path)
    return {"message": "File processed successfully"}

@app.get("/ask")
async def ask_question(question: str):
    global vectorstore
    if vectorstore is None:
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
        else:
            return {"error": "Please upload a document first"}
    
    result = get_answer(question, vectorstore)
    return {"answer": result["result"]}