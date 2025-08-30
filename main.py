import os
import io
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form

import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load all API keys from the .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI()

# This global variable will act as our simple, in-memory database for the document
vector_store = None

@app.get("/")
def read_root():
    """Root endpoint for the API."""
    return {"message": "Server is running. Use Gemini and Murf AI to talk to your PDFs."}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Processes an uploaded PDF file."""
    global vector_store
    
    file_content = await file.read()
    pdf_stream = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(extracted_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)

    return {
        "filename": file.filename,
        "message": f"PDF processed with Gemini. Created {len(text_chunks)} text chunks."
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Answers a question based on the uploaded PDF and generates voice."""
    global vector_store
    
    if not vector_store:
        return {"error": "No document has been uploaded yet. Please upload a PDF first."}
        
    retriever = vector_store.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    response = qa_chain.invoke(question)
    text_answer = response["result"]

    MURF_API_URL = "https://api.murf.ai/v1/speech/generate"
    
    headers = {
        'api-key': os.getenv("MURF_API_KEY")
    }
    
    payload = {
        "text": text_answer,
        "voiceId": "en-US-natalie"
    }
    
    try:
        murf_response = requests.post(MURF_API_URL, headers=headers, json=payload)
        murf_response.raise_for_status()
        
        audio_url = murf_response.json().get("audioFile")

        return {
            "text_answer": text_answer,
            "audio_url": audio_url
        }
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to call Murf AI API", "details": str(e)}

@app.get("/status/")
def get_status():
    """Checks if a document has been successfully processed."""
    global vector_store
    if vector_store:
        chunk_count = len(vector_store.get()["ids"])
        return {
            "status": "ready", 
            "message": f"A document is loaded with {chunk_count} chunks."
        }
    else:
        return {"status": "not_ready", "message": "No document has been uploaded yet."}