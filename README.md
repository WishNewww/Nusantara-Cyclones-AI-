# Nusantara Watch AI

An AI chatbot specialized in historical tropical cyclone information around
Indonesia (Southeast Indian Ocean and Western Pacific basins), built with a
Retrieval-Augmented Generation (RAG) backend and a Streamlit application
layer.

## Project structure

```
NusantaraWatchAI-Frontend/
├── app.py              # Streamlit application layer (this layer)
├── config.py           # UI copy, defaults, and option lists
├── requirements.txt    # Application-layer dependencies
├── assets/
│   └── style.css        # Light visual polish
├── rag.py              # Backend (owned separately) — do not modify
└── build_vector.py     # Backend (owned separately) — do not modify
```

The application layer communicates with the backend exclusively through
the single public function:

```python
answer = ask_question(question)
```

No FAISS, embedding, retrieval, or LLM logic is implemented in `app.py`.

## Requirements

- Python 3.9+
- The backend files (`rag.py`, `build_vector.py`) and their own
  dependencies already installed/configured, as provided by the backend
  engineer.

## How to launch the application

1. **Install application-layer dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure the backend dependencies are also installed** (FAISS,
   embeddings, LangChain, LLM SDKs, etc.), as specified by the backend
   engineer's own requirements file.

3. **Run the app** from the project root (the folder containing `app.py`
   and `rag.py`):

   ```bash
   streamlit run app.py
   ```

4. Streamlit will print a local URL (typically `http://localhost:8501`).
   Open it in your browser.

5. In the sidebar, enter your **API key**, choose a **model**, and
   optionally adjust **temperature** and **maximum retrieved documents**.
   Then start chatting, or click one of the suggested example questions
   on the welcome screen.

## Notes for the backend engineer

- `app.py` first attempts a richer call,
  `ask_question(question, model=..., temperature=..., max_docs=...)`, and
  automatically falls back to the documented `ask_question(question)`
  signature if the richer call raises `TypeError`. This means no changes
  are required on your side; the extra parameters are purely optional and
  forward-compatible.
- The selected API key is exported as an environment variable (e.g.
  `GEMINI_API_KEY` or `LLAMA_API_KEY`, see `config.MODEL_OPTIONS`) in case
  `rag.py` reads credentials from the environment.
