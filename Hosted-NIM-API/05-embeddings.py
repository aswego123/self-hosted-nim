"""
=======================================================================
Lesson 05 — Embeddings: turning text into vectors with a NIM
=======================================================================

WHAT THIS TEACHES
-----------------
- NIM is not only chat. There are embedding NIMs whose job is to convert
  text -> a fixed-length numeric vector (a "dense embedding").
- The two roles of QA embeddings:
    * `input_type="passage"` -> vectors for documents you STORE
    * `input_type="query"`   -> vectors for user QUESTIONS
  Using the right role gives noticeably better retrieval quality.
- Cosine similarity: how "close" two vectors are. This is what a vector
  database computes at query time. We do it by hand here to build
  intuition.

WHY IT MATTERS
--------------
This is the foundation of RAG (Retrieval-Augmented Generation) — Step 4.
The pipeline is:
    docs -> embedding NIM -> vectors -> vector DB
    question -> embedding NIM -> vector -> similarity search -> top-k docs
    top-k docs + question -> chat NIM -> grounded answer

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/05-embeddings.py
"""

from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Embedding NIM. 1024-dim vectors, tuned for question-answering retrieval.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Higher = more similar."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # A tiny "corpus": 3 candidate passages we could retrieve.
    passages = [
        "NVIDIA NIM packages optimized inference microservices as Docker containers.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "TensorRT-LLM optimizes large language models for NVIDIA GPUs.",
    ]
    query = "How do I deploy LLMs efficiently on GPUs?"

    # 1) Embed the passages. NVIDIA's QA embeddings support an "input_type"
    #    hint via the extra_body kwarg. Passing "passage" tells the model
    #    "these are documents to be stored".
    passage_resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=passages,
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    passage_vecs = [d.embedding for d in passage_resp.data]

    # 2) Embed the query. Use input_type="query" — the model produces a
    #    slightly different vector optimized for asking, not storing.
    query_resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    query_vec = query_resp.data[0].embedding

    print("first 8 dims of query vector:", query_vec[:8])
    print("vector length (should be ~1.0):", math.sqrt(sum(x*x for x in query_vec)))

    print(f"embedding dim : {len(query_vec)}")
    print(f"query         : {query}\n")

    # 3) Score each passage by cosine similarity to the query.
    scored = [(cosine(query_vec, v), p) for v, p in zip(passage_vecs, passages)]
    scored.sort(key=lambda x: x[0], reverse=True)

    print("ranked passages (higher = more relevant):")
    for score, passage in scored:
        print(f"  {score:+.4f}  {passage}")

    # Intuition check: the two NIM/TensorRT lines should beat the Eiffel
    # Tower line by a wide margin.
    return 0


if __name__ == "__main__":
    sys.exit(main())
