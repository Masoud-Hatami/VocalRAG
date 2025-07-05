import re
import streamlit as st
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

template = """
You are a helpful and informative assistant. Your task is to answer the user's question based *only* on the provided context. If the context does not contain the answer, state that you cannot find the answer in the provided information. Keep your answer concise, using a maximum of three sentences.
Question: {question}
Context: {context}
Answer:
"""

audios_directory = 'VocalRAG/audios/'

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def upload_audio(file):
    with open(audios_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def transcribe_audio(file_path):
    whisper_model = whisper.load_model("medium.en")
    result = whisper_model.transcribe(file_path)
    return result["text"]

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_text(text)

def index_docs(texts):
    vector_store.add_texts(texts)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return clean_text(chain.invoke({"question": question, "context": context}))

def clean_text(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav"],
    accept_multiple_files=False
)

if uploaded_file:
    upload_audio(uploaded_file)
    text = transcribe_audio(audios_directory + uploaded_file.name)
    chunked_texts = split_text(text)
    index_docs(chunked_texts)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_docs = retrieve_docs(question)
        answer = answer_question(question, related_docs)
        st.chat_message("assistant").write(answer)