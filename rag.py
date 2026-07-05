"""
rag.py
======
Retrieval-Augmented Generation layer for CycloneGPT.

Responsibilities
----------------
1.  Load the Gemini embedding model (must match the one used in
    build_vector.py) and the persisted FAISS index it produced.
2.  Wrap the index in a similarity-search retriever (top-k = 5).
3.  Assemble a strict, grounded prompt that forbids the LLM from using
    anything other than the retrieved cyclone documents.
4.  Expose a single public entry point, `ask_question(question: str) -> str`,
    that a Streamlit (or any other) frontend can call directly.

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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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

# Must match the embedding model used when the index was built.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

TOP_K = 5

SYSTEM_PROMPT = """You are CycloneGPT, an assistant specialized in historical \
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

CONTEXT:
{context}
"""

USER_PROMPT = "QUESTION:\n{question}"


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def load_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Load the Gemini embedding model used for query-time similarity search.

    Uses task_type='RETRIEVAL_QUERY' (as opposed to 'RETRIEVAL_DOCUMENT'
    used at index time in build_vector.py) - this asymmetric setup is
    recommended by Google for retrieval-style embeddings and improves
    match quality between short queries and long documents.
    """
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise EnvironmentError(
            "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
            "in your environment or in a .env file."
        )
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
    )


def load_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
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


def build_llm() -> ChatGoogleGenerativeAI:
    """Build the generation model. Temperature is kept low (0) to minimize
    creative drift away from the retrieved context."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def format_docs(docs: list[Document]) -> str:
    """Flatten retrieved Documents into a single context string, keeping
    each cyclone's content clearly delimited and attributed."""
    if not docs:
        return "No matching cyclone records were found in the dataset."

    blocks = []
    for doc in docs:
        name = doc.metadata.get("name", "Unknown")
        region = doc.metadata.get("region", "Unknown region")
        blocks.append(f"--- Cyclone: {name} ({region}) ---\n{doc.page_content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Lazily-initialized singletons
# ---------------------------------------------------------------------------
# A Streamlit frontend reruns the script on every interaction, so we cache
# the expensive objects (embeddings, vector store, chain) at module level
# and only build them once per process.

_chain = None


def _get_chain():
    """Build (once) and return the full retrieval -> prompt -> LLM chain."""
    global _chain
    if _chain is not None:
        return _chain

    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever = build_retriever(vector_store)
    prompt = build_prompt()
    llm = build_llm()

    _chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return _chain


def ask_question(question: str) -> str:
    """Public entry point: answer a natural-language question about the
    tropical cyclone dataset using RAG.

    Parameters
    ----------
    question:
        A natural-language question, e.g. "Tell me about Cyclone Seroja."

    Returns
    -------
    The model's answer as a plain string, grounded strictly in the
    retrieved cyclone documents. Returns a helpful error message (rather
    than raising) if the backend is not ready, so a Streamlit UI can
    display it directly without needing its own try/except.
    """
    if not question or not question.strip():
        return "Please provide a question about a tropical cyclone."

    try:
        chain = _get_chain()
    except (FileNotFoundError, EnvironmentError) as exc:
        logger.error(str(exc))
        return f"CycloneGPT backend is not ready: {exc}"

    try:
        return chain.invoke(question.strip())
    except Exception as exc:  # noqa: BLE001 - surface any runtime API error to the caller
        logger.exception("Error while answering question.")
        return f"An error occurred while generating the answer: {exc}"


if __name__ == "__main__":
    # Simple CLI smoke test: `python rag.py "Tell me about Cyclone Seroja"`
    import sys

    query = " ".join(sys.argv[1:]) or "Tell me about Cyclone Seroja."
    print(ask_question(query))
