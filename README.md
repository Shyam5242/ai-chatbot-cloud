# AI Chatbot (Cloud, Gemini)

A free, deployable AI chatbot built with Streamlit and the Gemini API, with PDF question answering and image chat.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

> Looking for the fully private version that runs on your own machine? See **ai-chatbot-local** (Ollama).

![Screenshot](screenshot.png)

## Features

- **Streaming chat** with conversation memory
- **Model switcher** between Gemini Flash models
- **Personalities**: helpful assistant, study buddy, coding tutor, or your own custom prompt
- **Document Q&A (RAG)**: upload PDFs and ask questions; answers come from your documents, with a Sources dropdown (ChromaDB + Gemini embeddings)
- **Image chat**: attach an image and ask questions about it
- **Creativity slider** to control the temperature
- Friendly error messages for rate limits and missing keys

## Tech stack

Python, Streamlit, Google Gemini API (`google-genai`), ChromaDB, pypdf

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.streamlit/secrets.toml` with:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (do **not** commit your API key; `.gitignore` already excludes `secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select this repo and `app.py` as the main file.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## How it works

- **Chat:** messages and history are sent to the selected Gemini model and the reply is streamed back.
- **Document Q&A:** PDFs are split into overlapping chunks, embedded with the Gemini embedding model, and stored in an in-memory ChromaDB collection. For each question, the 4 most similar chunks are retrieved and added to the prompt.
- **Image chat:** the attached image is sent along with your message to the multimodal Gemini model.

## Notes

- The free tier has rate limits. If you see a 429 error, wait a bit and try again.
- Free-tier requests may be used by Google to improve its products, so avoid uploading sensitive documents.
- Model names change over time. If you get a "model not found" error, update `CHAT_MODELS` or `EMBED_MODEL` at the top of `app.py`.
- Scanned PDFs (images of text) are not supported.
- The app may sleep after a period of inactivity on the free Streamlit tier.

## Future improvements

- OCR support for scanned PDFs
- User login and saved chat history
- Persistent document storage
