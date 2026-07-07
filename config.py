"""
config.py
=========
Central configuration for the Nusantara Watch AI Streamlit application.

This module contains ONLY presentation-layer configuration (titles, copy,
UI defaults, option lists). It does not import or depend on the backend
(rag.py / build_vector.py) in any way, keeping the application layer
cleanly separated from the AI/RAG layer.
"""

# --------------------------------------------------------------------------
# General application metadata
# --------------------------------------------------------------------------
APP_NAME = "Nusantara Cyclone Watch"
APP_ICON = "🌪️"
PAGE_TITLE = f"{APP_ICON} {APP_NAME}"
LAYOUT = "wide"

APP_TAGLINE = "Your AI assistant for historical tropical cyclone data around Indonesia."

SIDEBAR_DESCRIPTION = (
    "Nusantara Watch AI answers questions about historical tropical cyclones in the "
    "Southeast Indian Ocean and Western Pacific basins, using a "
    "Retrieval-Augmented Generation (RAG) backend built on top of "
    "historical best-track cyclone records."
)

WELCOME_MESSAGE = (
    f"👋 **Welcome to {APP_NAME}!**\n\n"
    "Ask me anything about historical tropical cyclones that have affected "
    "Indonesia and surrounding basins — tracks, intensity, timing, and "
    "comparisons between storms. Try one of the example questions below, "
    "or type your own question."
)

ABOUT_TEXT = (
    "**Nusantara Watch AI** is a Retrieval-Augmented Generation (RAG) chatbot "
    "specialized in historical tropical cyclone data for the Southeast "
    "Indian Ocean and Western Pacific basins.\n\n"
    "- **Backend**: RAG pipeline (FAISS + embeddings + LLM)\n"
    "- **Frontend**: Streamlit application layer\n"
    "- **Data**: Historical best-track cyclone records\n\n"
    "This interface only communicates with the backend through a single "
    "public function, `ask_question()`. All retrieval, embedding, and "
    "generation logic is owned and maintained separately."
)

# --------------------------------------------------------------------------
# Example / suggested questions shown on the welcome screen
# --------------------------------------------------------------------------
# Each entry is one "case file" card. The icon is chosen to match what the
# question actually asks (track/position, intensity, comparison, timing),
# not decoration for its own sake.
EXAMPLE_QUESTIONS = [
    {"icon": "🌀", "question": "Tell me about Cyclone Seroja."},
    {"icon": "📈", "question": "Which cyclone had the highest intensity?"},
    {"icon": "⚖️", "question": "Compare Cyclone Tracy and Cyclone Seroja."},
    {"icon": "🗓️", "question": "What happened during 1998?"},
]

# --------------------------------------------------------------------------
# Typing indicator (three pulsing dots) shown while waiting on the backend,
# in place of a static placeholder character.
# --------------------------------------------------------------------------
TYPING_INDICATOR_HTML = (
    '<span class="nw-typing"><span></span><span></span><span></span></span>'
)

# --------------------------------------------------------------------------
# Model / generation configuration options (optional controls)
# --------------------------------------------------------------------------
# Maps a human-friendly model label to the environment variable name that
# should hold its API key. Only used if the backend happens to read the
# corresponding environment variable; the app never assumes this.
#
# IMPORTANT: "Gemini" must map to "GOOGLE_API_KEY", not "GEMINI_API_KEY".
# rag.py's LLM client is `langchain_google_genai.ChatGoogleGenerativeAI`,
# which (via the underlying google-generativeai SDK) only ever reads the
# `GOOGLE_API_KEY` environment variable or an explicit `google_api_key`
# constructor argument - it does NOT read `GEMINI_API_KEY`. Mapping to the
# wrong name here means the key typed into the sidebar is silently never
# seen by the LLM, which then falls back to trying OAuth/ADC and fails
# with a 401 "ACCESS_TOKEN_TYPE_UNSUPPORTED" error on every question.
MODEL_OPTIONS = {
    "Gemini": "GOOGLE_API_KEY",
    "Llama": "LLAMA_API_KEY",
}
DEFAULT_MODEL = "Gemini"

DEFAULT_TEMPERATURE = 0.3
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0
TEMPERATURE_STEP = 0.05

DEFAULT_MAX_DOCS = 4
MIN_MAX_DOCS = 1
MAX_MAX_DOCS = 10

# --------------------------------------------------------------------------
# UX / behavior settings
# --------------------------------------------------------------------------
CHAT_INPUT_PLACEHOLDER = "Ask about a cyclone, e.g. 'Tell me about Cyclone Seroja'"
ASSISTANT_AVATAR = "🌪️"
USER_AVATAR = "🧑"
SPINNER_TEXT = "Nusantara Watch AI is analyzing historical records..."

# --------------------------------------------------------------------------
# Friendly, user-facing error messages
# --------------------------------------------------------------------------
ERROR_MESSAGES = {
    "missing_api_key": (
        "🔑 No API key detected. Please enter your API key in the sidebar "
        "before starting a conversation."
    ),
    "empty_question": "✏️ Please type a question before sending.",
    "backend_error": (
        "⚠️ Nusantara Watch AI ran into a problem while retrieving an answer. "
        "Please try rephrasing your question or try again in a moment."
    ),
    "unexpected_error": (
        "❌ An unexpected error occurred. If this keeps happening, please "
        "contact the project maintainer."
    ),
    "backend_unavailable": (
        "🚫 The Nusantara Watch AI backend (rag.py) could not be loaded. Please "
        "make sure the backend files are present and correctly configured."
    ),
}

CUSTOM_CSS_PATH = "assets/style.css"
