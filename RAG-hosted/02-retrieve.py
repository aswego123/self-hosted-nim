"""
=======================================================================
Lesson 02 — Retrieve: cosine top-k against the saved index
=======================================================================

WHAT THIS TEACHES
-----------------
- The "online" retrieval half of RAG. Given a query:
    1. Embed it (input_type="query" — asymmetric to "passage").
    2. Score against every stored vector (dot product = cosine, because
       we pre-normalized in lesson 01).
    3. Sort desc, return top-k with ids + text.
- Why we return top-K, not top-1: reranking (lesson 03) needs candidates
  to rerank, and the generator (lesson 04) benefits from multiple views
  on the same question.

WHY LOCAL EVENTUALLY WINS THIS STEP
-----------------------------------
Query embedding is on the HOT path of every user request. Local
nv-embedqa NIM saves the 50-200 ms network hop per query.

HOW TO RUN
----------
python RAG-hosted/02-retrieve.py "how does NIM speed up LLM serving?"
python RAG-hosted/02-retrieve.py               # uses a default query
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"
TOP_K = 10

HERE = Path(__file__).parent
INDEX_PATH = HERE / "index.npz"
META_PATH = HERE / "meta.json"
ENV_PATH = HERE.parent / "Hosted-NIM-API" / ".env"

DEFAULT_QUERY = "How does NIM make LLM deployment fast?"


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def retrieve_topk(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    ids: list[str],
    meta_by_id: dict[str, dict],
    k: int,
) -> list[tuple[float, str, str]]:
    """Return [(score, id, text)] sorted by score descending."""
    scores = matrix @ query_vec  # cosine, because both sides are unit-norm
    order = np.argsort(-scores)[:k]
    return [(float(scores[i]), ids[i], meta_by_id[ids[i]]["text"]) for i in order]


def main() -> int:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print(f"ERROR: set NVIDIA_API_KEY in {ENV_PATH}", file=sys.stderr)
        return 1
    if not INDEX_PATH.exists():
        print("ERROR: index.npz not found. Run 01-ingest.py first.", file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

    data = np.load(INDEX_PATH, allow_pickle=False)
    matrix = data["vectors"]
    ids = [str(x) for x in data["ids"]]
    meta_by_id = {p["id"]: p for p in json.loads(META_PATH.read_text())}

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    q_vec = embed_query(client, query)
    hits = retrieve_topk(q_vec, matrix, ids, meta_by_id, k=TOP_K)

    print(f"query : {query}\n")
    print(f"top-{TOP_K} by cosine similarity:")
    for rank, (score, pid, text) in enumerate(hits, start=1):
        snippet = text if len(text) <= 110 else text[:107] + "..."
        print(f"  {rank:2d}. {score:+.4f}  [{pid}]  {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
