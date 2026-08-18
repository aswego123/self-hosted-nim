"""
=======================================================================
Lesson 03 — Rerank: a cross-encoder NIM that reorders the top-k
=======================================================================

WHAT THIS TEACHES
-----------------
- The difference between a BI-encoder (embeddings) and a CROSS-encoder
  (rerank):
    * Bi-encoder: embeds query and passage INDEPENDENTLY, compares with
      cosine. Cheap, scalable, "roughly right".
    * Cross-encoder: feeds (query, passage) into ONE model that outputs
      a joint relevance score. More accurate. Too slow to run over the
      whole corpus, perfect for reranking the top-20.
- The two-stage retrieval pattern used by every production RAG:
      retrieve top-N with embeddings  →  rerank to top-K with cross-encoder
- Why the rerank score scale is unrelated to the embedding score scale
  — rerank returns log-odds, not cosine. Only ordering is comparable.

WHY LOCAL EVENTUALLY WINS THIS STEP
-----------------------------------
Rerank runs on EVERY query, over N passages each — a hidden cost
multiplier hosted. Local nv-rerankqa NIM is ~10 ms/pair on a modest GPU.

HOW TO RUN
----------
python RAG-hosted/03-rerank.py "how does NIM speed up LLM serving?"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank
from openai import OpenAI

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
RERANK_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Two-stage config: cast a wide net with embeddings, keep few after rerank.
RETRIEVE_N = 20
FINAL_K = 5

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


def retrieve_stage1(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    ids: list[str],
    meta_by_id: dict[str, dict],
    n: int,
) -> list[tuple[float, str, str]]:
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:n]
    return [(float(scores[i]), ids[i], meta_by_id[ids[i]]["text"]) for i in order]


def rerank_stage2(
    query: str,
    candidates: list[tuple[float, str, str]],
    api_key: str,
    k: int,
) -> list[tuple[float, str, str]]:
    """Cross-encode (query, passage) and reorder. Returns top-k."""
    reranker = NVIDIARerank(model=RERANK_MODEL, api_key=api_key, top_n=k)
    docs = [
        Document(page_content=text, metadata={"id": pid})
        for _, pid, text in candidates
    ]
    ranked = reranker.compress_documents(query=query, documents=docs)
    return [
        (float(d.metadata.get("relevance_score", 0.0)),
         d.metadata["id"],
         d.page_content)
        for d in ranked
    ]


def print_table(title: str, rows: list[tuple[float, str, str]]) -> None:
    print(f"\n{title}")
    for rank, (score, pid, text) in enumerate(rows, start=1):
        snippet = text if len(text) <= 100 else text[:97] + "..."
        print(f"  {rank:2d}. {score:+.4f}  [{pid}]  {snippet}")


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

    print(f"query : {query}")
    q_vec = embed_query(client, query)
    stage1 = retrieve_stage1(q_vec, matrix, ids, meta_by_id, n=RETRIEVE_N)
    print_table(f"Stage 1 — embedding top-{RETRIEVE_N} (cosine):", stage1[:FINAL_K])

    stage2 = rerank_stage2(query, stage1, api_key=api_key, k=FINAL_K)
    print_table(f"Stage 2 — rerank top-{FINAL_K} (log-odds; scale differs):", stage2)

    # Simple diff view: how many positions did each survivor move?
    pre_ranks = {pid: i for i, (_, pid, _) in enumerate(stage1, start=1)}
    print("\nposition shifts (rerank rank <- embed rank):")
    for new_rank, (_, pid, _) in enumerate(stage2, start=1):
        old_rank = pre_ranks.get(pid, "?")
        arrow = "=" if old_rank == new_rank else "↑" if isinstance(old_rank, int) and old_rank > new_rank else "↓"
        print(f"  [{pid}]  {old_rank:>3} → {new_rank}  {arrow}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
