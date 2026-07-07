"""
app.py
======
Nusantara Watch AI — Streamlit application layer.

This module owns everything related to the user-facing experience:
layout, chat interface, session state, sidebar configuration, and error
handling. It does NOT implement any retrieval-augmented generation (RAG)
logic. The only contact point with the backend is the single public
function `ask_question(question)` exposed by rag.py.

Responsibility boundary
------------------------
- ALLOWED : import and call `ask_question` from rag.py
- NOT ALLOWED : modify rag.py / build_vector.py, or reimplement any
  FAISS / embedding / retrieval / LLM logic here.
"""

import os

import streamlit as st

import config

# --------------------------------------------------------------------------
# Backend import (treated strictly as an external API)
# --------------------------------------------------------------------------
try:
    from rag import ask_question
    BACKEND_AVAILABLE = True
except Exception:  # noqa: BLE001 - we want to catch any import-time failure
    BACKEND_AVAILABLE = False


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all Streamlit session_state keys used by the app.

    Streamlit reruns the whole script on every interaction, so any state
    that must survive a rerun (conversation history, UI flags, user
    configuration) is stored in `st.session_state`.
    """
    defaults = {
        "messages": [],          # list[dict]: {"role": "user"/"assistant", "content": str}
        "is_processing": False,  # True while waiting for a backend response
        "pending_question": None,  # question queued by an example button
        "api_key": "",
        "model": config.DEFAULT_MODEL,
        "temperature": config.DEFAULT_TEMPERATURE,
        "max_docs": config.DEFAULT_MAX_DOCS,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation() -> None:
    """Reset the conversation history without touching user configuration."""
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.is_processing = False


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the sidebar: project info, configuration, and controls."""
    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} {config.APP_NAME}")
        st.caption(config.SIDEBAR_DESCRIPTION)

        st.divider()

        st.subheader("Configuration")

        st.session_state.api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help="Your LLM provider API key. This is kept only in your "
                 "browser session and is never stored on disk.",
        )

        st.session_state.model = st.selectbox(
            "Model",
            options=list(config.MODEL_OPTIONS.keys()),
            index=list(config.MODEL_OPTIONS.keys()).index(st.session_state.model),
            help="Select the LLM backend the RAG pipeline should use "
                 "(if supported by the backend).",
        )

        with st.expander("Advanced settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=config.MIN_TEMPERATURE,
                max_value=config.MAX_TEMPERATURE,
                value=st.session_state.temperature,
                step=config.TEMPERATURE_STEP,
                help="Higher values make answers more creative; lower "
                     "values make them more deterministic.",
            )
            st.session_state.max_docs = st.slider(
                "Maximum retrieved documents",
                min_value=config.MIN_MAX_DOCS,
                max_value=config.MAX_MAX_DOCS,
                value=st.session_state.max_docs,
                help="Maximum number of documents the retriever should "
                     "fetch as context for each answer.",
            )

        st.divider()

        st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
            on_click=clear_conversation,
            disabled=st.session_state.is_processing,
        )

        st.divider()

        with st.expander("ℹ️ About this project"):
            st.markdown(config.ABOUT_TEXT)


# --------------------------------------------------------------------------
# Main page — header & welcome / example questions
# --------------------------------------------------------------------------
def render_header() -> None:
    """Render the page title, tagline, and the signature scan bar."""
    st.markdown(
        f'<div class="nusantarawatchai-title">{config.APP_ICON} {config.APP_NAME}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="nusantarawatchai-tagline">{config.APP_TAGLINE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nw-scan-bar"></div>', unsafe_allow_html=True)


def render_welcome_and_examples() -> None:
    """Show a welcome message and clickable example questions.

    Only shown when the conversation is empty, mirroring the empty-state
    pattern used by ChatGPT.
    """
    st.markdown(config.WELCOME_MESSAGE)

    st.write("**Try asking:**")

    cols = st.columns(2)
    for idx, example in enumerate(config.EXAMPLE_QUESTIONS):
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="nw-case-eyebrow">Case {idx + 1:02d}</div>'
                    f'<div class="nw-case-question">{example["icon"]} '
                    f'{example["question"]}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Ask this",
                    key=f"example_{idx}",
                    use_container_width=True,
                    disabled=st.session_state.is_processing,
                ):
                    st.session_state.pending_question = example["question"]
                    st.rerun()


# --------------------------------------------------------------------------
# Chat rendering
# --------------------------------------------------------------------------
def render_chat_history() -> None:
    """Render all previously exchanged messages using chat components."""
    for message in st.session_state.messages:
        avatar = (
            config.USER_AVATAR if message["role"] == "user" else config.ASSISTANT_AVATAR
        )
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


# --------------------------------------------------------------------------
# Backend communication
# --------------------------------------------------------------------------
def call_backend(question: str) -> str:
    """Call the backend's `ask_question` function safely.

    The public contract is `answer = ask_question(question)`. To stay
    forward-compatible in case the backend later accepts optional
    configuration (model, temperature, max_docs) without breaking today's
    single-argument contract, this wrapper first tries a richer call and
    transparently falls back to the documented single-argument call.

    Raises
    ------
    Exception
        Re-raises any exception from the backend so the caller can present
        a friendly, contextual error message.
    """
    # Make user-provided configuration available via environment variables,
    # in case the backend chooses to read them. This never mutates rag.py.
    if st.session_state.api_key:
        env_var = config.MODEL_OPTIONS.get(st.session_state.model)
        if env_var:
            os.environ[env_var] = st.session_state.api_key

    try:
        return ask_question(
            question,
            model=st.session_state.model,
            temperature=st.session_state.temperature,
            max_docs=st.session_state.max_docs,
        )
    except TypeError:
        # Backend only implements the documented single-argument signature.
        return ask_question(question)


def handle_question(question: str) -> None:
    """Validate, send, and record a single question/answer turn.

    This is the single entry point used both by manual chat input and by
    the clickable example questions, keeping behavior consistent.
    """
    question = (question or "").strip()

    if not question:
        st.warning(config.ERROR_MESSAGES["empty_question"])
        return

    if not BACKEND_AVAILABLE:
        st.error(config.ERROR_MESSAGES["backend_unavailable"])
        return

    if not st.session_state.api_key:
        st.warning(config.ERROR_MESSAGES["missing_api_key"])
        return

    if st.session_state.is_processing:
        # Prevent duplicate/concurrent requests.
        return

    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar=config.USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=config.ASSISTANT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown(config.TYPING_INDICATOR_HTML, unsafe_allow_html=True)
        with st.spinner(config.SPINNER_TEXT):
            try:
                answer = call_backend(question)
                placeholder.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:  # noqa: BLE001 - surface any backend error
                friendly_message = config.ERROR_MESSAGES["backend_error"]
                placeholder.error(friendly_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": friendly_message}
                )
                # Keep the technical detail available for debugging without
                # exposing a stack trace to end users.
                st.caption(f"Technical detail: {exc}")

    st.session_state.is_processing = False


# --------------------------------------------------------------------------
# App entry point
# --------------------------------------------------------------------------
def load_custom_css() -> None:
    """Inject the optional custom stylesheet, if present."""
    css_path = config.CUSTOM_CSS_PATH
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def main() -> None:
    """Configure the page and orchestrate rendering."""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.APP_ICON,
        layout=config.LAYOUT,
    )

    load_custom_css()
    init_session_state()
    render_sidebar()
    render_header()

    if not BACKEND_AVAILABLE:
        st.error(config.ERROR_MESSAGES["backend_unavailable"])

    if not st.session_state.messages:
        render_welcome_and_examples()

    render_chat_history()

    # Handle a question queued by an example button click.
    if st.session_state.pending_question:
        queued_question = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_question(queued_question)

    # Manual chat input. Disabled while a request is in flight to prevent
    # duplicate submissions.
    user_input = st.chat_input(
        config.CHAT_INPUT_PLACEHOLDER,
        disabled=st.session_state.is_processing,
    )
    if user_input:
        handle_question(user_input)


if __name__ == "__main__":
    main()
