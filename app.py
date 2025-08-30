import os
import io
import requests
import streamlit as st
from dotenv import load_dotenv

import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load all API keys from the .env file
load_dotenv()

# --- BACKEND LOGIC ---
@st.cache_resource
def process_pdf(file):
    """Processes an uploaded PDF file and returns a searchable vector store."""
    file_content = file.read()
    pdf_stream = io.BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(extracted_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_pdf_response(vector_store, question, selected_voice):
    """Answers a question based on the PDF and generates a voice response."""
    retriever = vector_store.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # Use a simple RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    response = qa_chain.invoke(question)
    text_answer = response["result"]

    # Generate voice
    MURF_API_URL = "https://api.murf.ai/v1/speech/generate"
    headers = {'api-key': os.getenv("MURF_API_KEY")}
    payload = {"text": text_answer, "voiceId": selected_voice}
    
    murf_response = requests.post(MURF_API_URL, headers=headers, json=payload)
    murf_response.raise_for_status()
    
    audio_url = murf_response.json().get("audioFile")
    return text_answer, audio_url

# --- STREAMLIT UI ---
st.set_page_config(page_title="Talk to Your PDF", layout="wide")

# --- NEW: HIDE THE DEFAULT AUDIO PLAYER ---
st.markdown("""
<style>
    /* This CSS hides the audio player that Streamlit creates */
    audio {
        display: none;
    }
</style>
""", unsafe_allow_html=True)
# --- END OF NEW CODE ---

st.title("Talk to Your PDF 🗣️")

# Use session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            st.session_state.vector_store = process_pdf(uploaded_file)
            st.session_state.messages = [] 
            st.success('PDF processed! You can now ask questions.')

    st.header("Select a Voice")
    voice_options = {
        "Natalie (Female)": "en-US-natalie",
        "Aaron (Male)": "en-US-aaron",
        "Michelle (Female, American)": "en-US-michelle",
        "Alex (Male, British)": "en-GB-alex"
    }
    selected_voice_name = st.selectbox("Choose a voice:", options=list(voice_options.keys()))
    st.session_state.selected_voice_id = voice_options[selected_voice_name]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat interface
if st.session_state.vector_store is not None:
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                text_answer, audio_url = get_pdf_response(
                    st.session_state.vector_store, prompt, st.session_state.selected_voice_id
                )
                st.markdown(text_answer)
                if audio_url:
                    # --- MODIFIED: ADDED autoplay=True ---
                    st.audio(audio_url, format='audio/wav', autoplay=True)
        
        st.session_state.messages.append({"role": "assistant", "content": text_answer})
else:
    st.info("Please upload a PDF file in the sidebar to begin.")