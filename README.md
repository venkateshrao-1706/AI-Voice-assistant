# Voice-Activated AI Assistant 🎙️🤖

A continuous-listening Python voice assistant powered by Groq (Whisper + Llama), LangChain/LangGraph, and secure read-only Gmail integration.

## Features
- **Speech-to-Text:** Instant audio transcription using Groq Whisper (`whisper-large-v3-turbo`).
- **Smart Tools:** Equipped with DuckDuckGo Web Search, Wikipedia API, and read-only Gmail access.
- **Conversational Memory:** Uses LangGraph threads to remember context across turns.
- **Text-to-Speech:** Offline voice feedback using `pyttsx3`.

## Tech Stack
- Python
- Groq API
- LangChain / LangGraph
- `sounddevice` & `scipy`
- `pyttsx3`
- LangChain Google Community (Gmail API)