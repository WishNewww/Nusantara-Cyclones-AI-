"""
rag.py
======
Retrieval-Augmented Generation layer for Nusantara Watch AI.

Responsibilities
----------------
1.  Load the local HuggingFace embedding model (must match the one used
    in build_vector.py) and the persisted FAISS index it produced.
2.  Wrap the index in a similarity-search retriever (top-k = 5).
3.  Assemble a strict, grounded prompt that forbids the LLM from using
    anything other than the retrieved cyclone documents.
4.  Expose a single public entry point, `ask_question(question, *, model=None,
    temperature=None, max_docs=None) -> str`, that a Streamlit (or any
    other) frontend can call directly. The extra keyword arguments are
    optional and backward-compatible with the plain `ask_question(question)`
    call.

This module intentionally has no UI code in it - it is pure backend logic
so it can be imported unchanged by a future `streamlit_app.py`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "langchain-community is required for the FAISS vector store. "
        "Install it with: pip install langchain-community"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("rag")

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# Must match the embedding model used when the index was built (build_vector.py).
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

TOP_K = 5

# This backend currently only implements a Gemini chat LLM. Any other
# value passed via the optional `model=` kwarg (e.g. a frontend offering
# "Llama") is explicitly rejected below rather than silently ignored.
SUPPORTED_MODELS = {"Gemini"}

SYSTEM_PROMPT = """You are Nusantara Watch AI, an assistant specialized in historical \
tropical cyclone data for the Western Pacific and Southeast Indian regions \
(1945-2025).

STRICT RULES - follow all of them:
1. Only answer using the CONTEXT provided below. The context comes directly \
from a verified cyclone observation dataset.
2. Never use your own general knowledge about cyclones, storms, or weather \
history. Never fabricate names, dates, coordinates, or intensity values.
3. If the answer is not present in the CONTEXT, clearly state that the \
information is unavailable in the dataset. Do not guess.
4. When comparing multiple cyclones, only compare values that are explicitly \
present in the CONTEXT for each of them.
5. Cite the cyclone name and region you drew each fact from when it helps \
the user verify the answer.
6. Don't just list the retrieved fields back verbatim - synthesize them into \
a genuine narrative. The CONTEXT already includes a few pre-computed \
interpretive fields (Intensity Category, Intensity Trend) precisely so you \
can use them narratively without needing to invent your own classification \
- e.g. explain what the storm's peak Intensity Category means in plain \
language, describe its movement using the first/last position and \
direction fields together rather than listing them separately, and note \
what an "Increasing"/"Decreasing"/"Fluctuating" trend implies about how the \
storm evolved over its observed lifetime. Every word of this synthesis must \
still trace back to a field actually present in the CONTEXT - narrate the \
data more richly, never add information the data doesn't contain.

CONTEXT:
{context}
"""

USER_PROMPT = "QUESTION:\n{question}"


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def load_embeddings() -> HuggingFaceEmbeddings:
    """Load the local HuggingFace embedding model used for query-time
    similarity search. Must be the same model used in build_vector.py,
    since query and document vectors have to live in the same space.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Load the FAISS index built by build_vector.py."""
    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{FAISS_INDEX_DIR}'. "
            "Run `python build_vector.py` first to build the vector store."
        )
    # allow_dangerous_deserialization is required by langchain-community's
    # FAISS loader because it unpickles the docstore. This is safe here
    # because the index is built locally by build_vector.py, not loaded
    # from an untrusted third party.
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_retriever(vector_store: FAISS, k: int = TOP_K):
    """Build a similarity-search retriever with top-k = k."""
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def build_prompt() -> ChatPromptTemplate:
    """Build the grounded RAG prompt."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )


def build_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Build the generation model.

    `temperature` defaults to 0 to minimize creative drift away from the
    retrieved context. Constructing a fresh instance per call is cheap
    (no network call happens until `.invoke()`), which is what lets this
    read the caller's current API key from the environment every time
    instead of freezing in whichever key happened to be set the first
    time a chain was ever built (see `ask_question` below).

    API key resolution
    -------------------
    `ChatGoogleGenerativeAI` (via the underlying `google-generativeai` SDK)
    only reads the `GOOGLE_API_KEY` environment variable by default - it
    does NOT read `GEMINI_API_KEY`, despite that being the name used
    elsewhere in this project's own tooling (the Streamlit frontend's
    `config.MODEL_OPTIONS`, TUTORIAL.md's Colab secrets cell, and a
    comment in build_vector.py all assumed otherwise). Left as-is, any
    caller that only sets `GEMINI_API_KEY` gets a 401
    "ACCESS_TOKEN_TYPE_UNSUPPORTED" error, because the SDK falls back to
    trying OAuth/ADC instead of an API key.

    To make this backend actually match that documented intent instead of
    silently breaking it, resolve the key explicitly here and accept
    either name, preferring GOOGLE_API_KEY if both happen to be set.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Set the GOOGLE_API_KEY (preferred) or "
            "GEMINI_API_KEY environment variable before calling ask_question()."
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=api_key,
    )


def format_docs(docs: list[Document]) -> str:
    """Flatten retrieved Documents into a single context string, keeping
    each cyclone's content clearly delimited and attributed.

    Uses metadata["full_text"] (the complete profile + observation
    history) rather than doc.page_content. page_content is deliberately
    a SHORT summary optimized for embedding quality (see
    build_vector.render_embedding_summary) and is not what should be
    shown to the LLM - the full report is. Falls back to page_content
    for backward compatibility with an index built before this change.
    """
    if not docs:
        return "No matching cyclone records were found in the dataset."

    blocks = []
    for doc in docs:
        name = doc.metadata.get("name", "Unknown")
        region = doc.metadata.get("region", "Unknown region")
        content = doc.metadata.get("full_text", doc.page_content)
        blocks.append(f"--- Cyclone: {name} ({region}) ---\n{content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Lazily-initialized singletons
# ---------------------------------------------------------------------------
# A Streamlit frontend reruns the script on every interaction, so we cache
# the expensive objects (embeddings, vector store, chain) at module level
# and only build them once per process.

# ---------------------------------------------------------------------------
# Cached backend resources
# ---------------------------------------------------------------------------
# A Streamlit frontend reruns the script on every interaction (and, in a
# typical deployment, serves multiple browser sessions from the SAME
# Python process). That has one important consequence for what is safe
# to cache at module level:
#
#   - `embeddings` and `vector_store` are expensive to build, deterministic,
#     and identical for every user/question - safe (and desirable) to
#     build once per process and reuse forever.
#   - `llm` and `retriever` are cheap to construct (no network call
#     happens until `.invoke()`/`.get_relevant_documents()`), but they are
#     NOT user-independent: `llm` binds an API key at construction time,
#     and `retriever` binds a user-configurable `k`. Caching a single one
#     of these at module level would silently keep serving the FIRST
#     user's API key / settings to every later session on the same
#     process. So these are rebuilt on every `ask_question()` call
#     instead, keeping the RAG pipeline itself (retrieval -> prompt ->
#     LLM) exactly as before.

_embeddings = None
_vector_store = None


def _get_backend_resources():
    """Build (once per process) and return the (embeddings, vector_store)
    pair. These are the only components safe to share across users."""
    global _embeddings, _vector_store
    if _embeddings is None:
        _embeddings = load_embeddings()
    if _vector_store is None:
        _vector_store = load_vector_store(_embeddings)
    return _embeddings, _vector_store


def ask_question(
    question: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_docs: int | None = None,
) -> str:
    """Public entry point: answer a natural-language question about the
    tropical cyclone dataset using RAG.

    Parameters
    ----------
    question:
        A natural-language question, e.g. "Tell me about Cyclone Seroja."
    model:
        Optional chat-model label from the frontend (e.g. "Gemini"). This
        backend only implements Gemini; any other value returns a clear,
        non-crashing message instead of silently being ignored.
    temperature:
        Optional sampling temperature for the LLM. Defaults to 0 (the
        original, deterministic behavior) if not provided.
    max_docs:
        Optional top-k override for the retriever. Defaults to TOP_K (5)
        if not provided.

    Returns
    -------
    The model's answer as a plain string, grounded strictly in the
    retrieved cyclone documents. Returns a helpful error message (rather
    than raising) if the backend is not ready or misconfigured, so a
    Streamlit UI can display it directly without needing its own
    try/except.
    """
    if not question or not question.strip():
        return "Please provide a question about a tropical cyclone."

    if model is not None and model not in SUPPORTED_MODELS:
        return (
            f"The '{model}' model is not supported by this backend yet. "
            f"Currently supported model(s): {', '.join(sorted(SUPPORTED_MODELS))}."
        )

    try:
        _, vector_store = _get_backend_resources()
    except (FileNotFoundError, EnvironmentError) as exc:
        logger.error(str(exc))
        return f"Nusantara Watch AI backend is not ready: {exc}"

    k = max_docs if isinstance(max_docs, int) and max_docs > 0 else TOP_K
    temp = temperature if isinstance(temperature, (int, float)) else 0

    try:
        retriever = build_retriever(vector_store, k=k)
        prompt = build_prompt()
        llm = build_llm(temperature=temp)  # reads the CURRENT API key, every call
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(question.strip())
    except Exception as exc:  # noqa: BLE001 - surface any runtime API error to the caller
        logger.exception("Error while answering question.")
        return f"An error occurred while generating the answer: {exc}"


if __name__ == "__main__":
    # Simple CLI smoke test: `python rag.py "Tell me about Cyclone Seroja"`
    import sys

    query = " ".join(sys.argv[1:]) or "Tell me about Cyclone Seroja."
    print(ask_question(query))
