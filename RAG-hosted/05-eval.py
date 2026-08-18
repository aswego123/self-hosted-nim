"""
=======================================================================
Lesson 05 — Eval: recall@k and MRR, with and without rerank
=======================================================================

WHAT THIS TEACHES
-----------------
- The two RAG retrieval metrics you actually need:
    * recall@k  — was any relevant passage in the top-k? (0/1 per query)
    * MRR       — 1 / rank of the first relevant passage. Rewards putting
                  the right answer HIGH, not just anywhere in top-k.
- Why rerank often lifts MRR more than recall@k:
    recall@k already found the right passage; rerank just moves it up.
    That matters because the generator pays more attention to the top of
    its context window.
- The ONE experiment that justifies adding a reranker to your stack.

HOW TO RUN
----------
python RAG-hosted/05-eval.py
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

RETRIEVE_N = 20
K_AT = 5

HERE = Path(__file__).parent
INDEX_PATH = HERE / "index.npz"
META_PATH = HERE / "meta.json"
GOLD_PATH = HERE / "gold.json"
ENV_PATH = HERE.parent / "Hosted-NIM-API" / ".env"


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def retrieve_stage1(query_vec, matrix, ids, n):
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:n]
    return [ids[i] for i in order]


def rerank_stage2(query, candidate_ids, meta_by_id, api_key, k):
    reranker = NVIDIARerank(model=RERANK_MODEL, api_key=api_key, top_n=k)
    docs = [
        Document(page_content=meta_by_id[pid]["text"], metadata={"id": pid})
        for pid in candidate_ids
    ]
    ranked = reranker.compress_documents(query=query, documents=docs)
    return [d.metadata["id"] for d in ranked]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def main() -> int:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print(f"ERROR: set NVIDIA_API_KEY in {ENV_PATH}", file=sys.stderr)
        return 1
    if not INDEX_PATH.exists():
        print("ERROR: index.npz not found. Run 01-ingest.py first.", file=sys.stderr)
        return 1

    data = np.load(INDEX_PATH, allow_pickle=False)
    matrix = data["vectors"]
    ids = [str(x) for x in data["ids"]]
    meta_by_id = {p["id"]: p for p in json.loads(META_PATH.read_text())}
    gold = json.loads(GOLD_PATH.read_text())

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    print(f"eval set: {len(gold)} queries, k={K_AT}, retrieve_n={RETRIEVE_N}\n")
    print(f"{'query':<60} {'r@k emb':>8} {'r@k rr':>8} {'MRR emb':>8} {'MRR rr':>8}")
    print("-" * 96)

    emb_recall, rr_recall, emb_mrr, rr_mrr = [], [], [], []
    for item in gold:
        q = item["query"]
        relevant = set(item["relevant_ids"])

        q_vec = embed_query(client, q)
        emb_hits = retrieve_stage1(q_vec, matrix, ids, n=RETRIEVE_N)
        rr_hits = rerank_stage2(q, emb_hits, meta_by_id, api_key=api_key, k=K_AT)

        r_e = recall_at_k(emb_hits, relevant, K_AT)
        r_r = recall_at_k(rr_hits, relevant, K_AT)
        m_e = reciprocal_rank(emb_hits, relevant)
        m_r = reciprocal_rank(rr_hits, relevant)

        emb_recall.append(r_e); rr_recall.append(r_r)
        emb_mrr.append(m_e); rr_mrr.append(m_r)

        q_short = q if len(q) <= 58 else q[:55] + "..."
        print(f"{q_short:<60} {r_e:>8.2f} {r_r:>8.2f} {m_e:>8.3f} {m_r:>8.3f}")

    def mean(xs): return sum(xs) / len(xs)
    print("-" * 96)
    print(f"{'AVERAGE':<60} "
          f"{mean(emb_recall):>8.2f} {mean(rr_recall):>8.2f} "
          f"{mean(emb_mrr):>8.3f} {mean(rr_mrr):>8.3f}")

    print("\nlegend: r@k = recall@k (higher is better);  MRR = mean reciprocal rank")
    print("        emb  = embedding retrieval only")
    print("        rr   = embedding retrieval + rerank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
