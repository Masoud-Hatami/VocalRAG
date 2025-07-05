# Audio Q&A System

This is a Streamlit application that processes audio files and answers questions based on their content using speech-to-text and RAG (Retrieval-Augmented Generation).

## Features
- Upload MP3 or WAV audio files
- Transcribe audio to text using Whisper
- Process text using Google Generative AI embeddings
- Answer questions based on audio content using Gemini model

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Set up Google API key:
- Create a `.streamlit/secrets.toml` file or set environment variable:
```toml
GOOGLE_API_KEY = "your-api-key-here"
```
3. Create `VocalRAG/audios/` directory for audio storage

## Usage
1. Run the app:
```bash
streamlit run main.py
```
2. Upload an audio file (MP3 or WAV)
3. Ask questions about the audio content
4. View responses in the chat interface

## Requirements
See `requirements.txt` for full list of dependencies.

## Notes
- Audio files are stored temporarily in `VocalRAG/audios/`
- Requires Google Cloud API key for embeddings and chat model
- Uses Whisper medium.en model for transcription