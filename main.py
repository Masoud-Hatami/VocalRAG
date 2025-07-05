import re
import warnings
from dotenv import load_dotenv
import streamlit as st
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
# Initialize session state for vector store
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

load_dotenv()
AUDIO_DIR = 'VocalRAG/audios/'
template = """
You are a helpful and informative assistant. Your task is to answer the user's question based *only* on the provided context. If the context does not contain the answer, state that you cannot find the answer in the provided information. Keep your answer concise, using a maximum of three sentences.
Question: {question}
Context: {context}
Answer:
"""

def setup_directories():
    try:
        os.makedirs(AUDIO_DIR, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create audio directory: {str(e)}")
        return False
    return True

def initialize_models():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if st.session_state.vector_store is None:
            st.session_state.vector_store = InMemoryVectorStore(embeddings)
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    except Exception as e:
        st.error(f"Failed to initialize models: {str(e)}")
        return None

def upload_audio(file):
    try:
        file_path = os.path.join(AUDIO_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to upload audio: {str(e)}")
        return None

def transcribe_audio(file_path):
    try:
        whisper_model = whisper.load_model("medium.en")
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Failed to transcribe audio: {str(e)}")
        return None

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def index_docs(texts):
    try:
        st.session_state.vector_store.add_texts(texts)
    except Exception as e:
        st.error(f"Failed to index documents: {str(e)}")

def retrieve_docs(query):
    try:
        return st.session_state.vector_store.similarity_search(query)
    except Exception as e:
        st.error(f"Failed to retrieve documents: {str(e)}")
        return []

def answer_question(question, documents):
    try:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return clean_text(chain.invoke({"question": question, "context": context}))
    except Exception as e:
        st.error(f"Failed to answer question: {str(e)}")
        return "An error occurred while processing the question."

def clean_text(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# Main app
st.title("Audio Q&A System")
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("Google API key not configured. Please set GOOGLE_API_KEY in environment variables or Streamlit secrets.")
else:
    if setup_directories():
        model = initialize_models()
        if model:
            uploaded_file = st.file_uploader(
                "Upload Audio",
                type=["mp3", "wav"],
                accept_multiple_files=False
            )

            if uploaded_file:
                file_path = upload_audio(uploaded_file)
                if file_path:
                    text = transcribe_audio(file_path)
                    if text:
                        chunked_texts = split_text(text)
                        index_docs(chunked_texts)
                        st.success("Audio processed successfully!")

            question = st.chat_input("Ask a question about the audio content")
            if question:
                with st.chat_message("user"):
                    st.write(question)
                related_docs = retrieve_docs(question)
                if related_docs:
                    answer = answer_question(question, related_docs)
                    with st.chat_message("assistant"):
                        st.write(answer)
                else:
                    st.error("No relevant information found in the audio.")