"""
inspect_retrieval.py
=====================
Standalone diagnostic tool - NOT part of the production backend.

Bypasses the LLM entirely and inspects the FAISS index + retriever
directly, so you can tell whether a bad answer is a RETRIEVAL problem
or a GENERATION problem. Run this after `python build_vector.py`.

Usage:
    python inspect_retrieval.py "Tell me about Cyclone ANN 1."
    python inspect_retrieval.py --stats
"""

from __future__ import annotations

import sys

from rag import _get_backend_resources


def show_index_stats() -> None:
    """Print basic facts about the persisted index so you can sanity-check
    it was built from the data/documents you think it was."""
    _, vector_store = _get_backend_resources()
    total = vector_store.index.ntotal
    print(f"Total vectors in FAISS index: {total}")

    # Pull a few sample names straight out of the docstore to eyeball.
    names = []
    for doc_id in list(vector_store.docstore._dict.keys())[:5]:
        doc = vector_store.docstore._dict[doc_id]
        names.append(doc.metadata.get("name"))
    print(f"Sample cyclone names in the index: {names}")

    has_full_text = any(
        "full_text" in vector_store.docstore._dict[doc_id].metadata
        for doc_id in list(vector_store.docstore._dict.keys())[:5]
    )
    print(f"Documents carry the metadata['full_text'] fix: {has_full_text}")


def inspect_query(question: str, k: int = 5) -> None:
    """Run retrieval ONLY (no LLM call) and print exactly what came back,
    with distance scores, so you can see whether the right cyclone was
    even retrieved before blaming the prompt or the model."""
    _, vector_store = _get_backend_resources()

    print(f"Query: {question!r}\n")
    results = vector_store.similarity_search_with_score(question, k=k)
    if not results:
        print("No documents retrieved at all.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"#{rank}  score={score:.4f}  name={doc.metadata.get('name')!r}  "
              f"region={doc.metadata.get('region')!r}")
        print(f"      embedding text: {doc.page_content[:120]}...")
    print()
    print("Lower score = closer match (FAISS default is L2 distance).")
    print("If the cyclone you expect isn't in this list at all, it's a")
    print("RETRIEVAL problem (embedding/index), not a prompt/LLM problem.")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        show_index_stats()
    else:
        query = (
            " ".join(a for a in sys.argv[1:] if a != "--stats")
            or "Tell me about Cyclone ANN 1."
        )
        inspect_query(query)
