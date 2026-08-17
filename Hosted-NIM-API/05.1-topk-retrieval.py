"""
=======================================================================
Lesson 05a — Top-k retrieval: writing a 1-file vector database
=======================================================================

WHAT THIS TEACHES
-----------------
- What a vector DB actually does at query time (nothing more than what
  you're about to code by hand):
      1. Embed the query.
      2. Score it against every stored vector using cosine similarity.
      3. Sort descending, return the top-k.
- The `retrieve_topk` signature — you'll see this shape in FAISS,
  Milvus, pgvector, Chroma, LangChain retrievers, etc.
- Why looking at the RUNNERS-UP (positions 2..N), not just #1, is the
  single most useful debug technique in RAG.

WHY IT MATTERS
--------------
Once you've written this, "we use a vector database" stops being magic.
The database is doing the same math — it just scales it (ANN indexes,
sharding, persistence). Understanding the math is 90% of debugging RAG.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/05a-topk-retrieval.py
"""

from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Higher = more similar."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def retrieve_topk(
    query_vec: list[float],
    corpus_vecs: list[list[float]],
    corpus_texts: list[str],
    k: int = 3,
) -> list[tuple[float, str]]:
    """Return the top-k most-similar passages to a query vector.

    This is the *entire* retrieval step of a vector database — no more,
    no less. Real DBs replace the O(N) sort with an ANN index for scale,
    but the semantics you get back are identical.
    """
    scored = [
        (cosine(query_vec, v), t) for v, t in zip(corpus_vecs, corpus_texts)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def embed_batch(
    client: OpenAI, texts: list[str], input_type: str
) -> list[list[float]]:
    """Embed a list of strings in a single API call.

    Batching is essential for cost/latency — see lesson 05.
    """
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [d.embedding for d in resp.data]


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # A slightly bigger corpus so top-k is actually interesting.
    # Notice we mix:
    #   * on-topic NVIDIA/LLM sentences
    #   * an adjacent-topic one (Kubernetes — deployment but not LLMs)
    #   * two off-topic distractors
    passages = [
        "NVIDIA NIM packages optimized inference microservices as Docker containers.",
        "TensorRT-LLM optimizes large language models for NVIDIA GPUs.",
        "vLLM is a high-throughput inference engine for LLMs.",
        "Kubernetes is a container orchestration platform.",
        "Triton Inference Server serves models with dynamic batching.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Chocolate chip cookies are best served warm.",
        "Prometheus collects metrics from services over HTTP.",
    ]

    # Two queries expressing different intents. Same corpus, different
    # winners — that's the retriever earning its keep.
    queries = [
        "How do I serve big language models fast on GPUs?",
        "How do I monitor a production service?",
    ]

    # ONE batched call at "index time" — this is what you'd do offline
    # when ingesting your knowledge base into a vector DB.
    print(f"embedding {len(passages)} passages in one call...")
    passage_vecs = embed_batch(client, passages, input_type="passage")
    print(f"done. vector dim = {len(passage_vecs[0])}\n")

    # Per-query embedding + retrieval loop.
    for query in queries:
        print("=" * 70)
        print(f"query: {query}")
        query_vec = embed_batch(client, [query], input_type="query")[0]

        # We ask for top-5 so we also see the RUNNERS-UP. In real RAG
        # you'd usually pass the top-3 or top-5 into the LLM's context,
        # not just #1 — because embeddings are lossy and rerankers or
        # the LLM itself sort out the true winner from the shortlist.
        top = retrieve_topk(query_vec, passage_vecs, passages, k=5)
        print("\n rank  score   passage")
        for i, (score, text) in enumerate(top, start=1):
            marker = "  <-- top-1" if i == 1 else ""
            print(f"  {i:>3}  {score:+.4f}  {text}{marker}")
        print()

    # WHAT TO NOTICE
    # --------------
    # 1) The two queries pick completely different top-1s, using the
    #    SAME 8-passage index. That's the retriever doing its job.
    # 2) The gap between #1 and #2 tells you how "confident" the match
    #    is. Small gap = ambiguous query, big gap = clear winner.
    # 3) In real RAG, we send the top-k (usually 3-5) to the LLM. This
    #    hedges against embedding lossiness — the LLM can pick.
    return 0


if __name__ == "__main__":
    sys.exit(main())
