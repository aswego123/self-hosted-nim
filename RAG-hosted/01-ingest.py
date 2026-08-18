"""
=======================================================================
Lesson 01 — Ingest: embed the corpus and save a vector index
=======================================================================

WHAT THIS TEACHES
-----------------
- The "offline" half of RAG: one-time embedding of your knowledge base.
  Every real vector DB (FAISS, pgvector, Chroma, Milvus, ...) does the
  same thing under the hood — read docs, embed in batches, store the
  vectors alongside the source text.
- Batching matters. We embed all passages in ONE HTTP call, not 24.
- Persisting the index: a numpy .npz for vectors + a JSON for metadata.
  Retrieval (lesson 02) loads these back with zero embedding cost.

WHY LOCAL EVENTUALLY WINS THIS STEP
-----------------------------------
Ingest embeds the WHOLE corpus. On real data that is millions of chunks,
which means millions of hosted API calls. Self-hosting a nv-embedqa NIM
turns that into one GPU-hour and $0 per re-ingest.

HOW TO RUN
----------
python RAG-hosted/01-ingest.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"

HERE = Path(__file__).parent
CORPUS_PATH = HERE / "corpus.json"
INDEX_PATH = HERE / "index.npz"
META_PATH = HERE / "meta.json"

# Reuse Step 2's .env — same NVIDIA_API_KEY.
ENV_PATH = HERE.parent / "Hosted-NIM-API" / ".env"


def embed_batch(client: OpenAI, texts: list[str], input_type: str) -> list[list[float]]:
    """One API call → one vector per input. Reused verbatim in later lessons."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [d.embedding for d in resp.data]


def main() -> int:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print(f"ERROR: set NVIDIA_API_KEY in {ENV_PATH}", file=sys.stderr)
        return 1

    corpus = json.loads(CORPUS_PATH.read_text())
    ids = [p["id"] for p in corpus]
    texts = [p["text"] for p in corpus]
    print(f"loaded {len(corpus)} passages from {CORPUS_PATH.name}")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    t0 = time.perf_counter()
    vectors = embed_batch(client, texts, input_type="passage")
    dt = time.perf_counter() - t0
    print(f"embedded {len(texts)} passages in {dt*1000:.0f} ms "
          f"(one batched call, dim={len(vectors[0])})")

    matrix = np.asarray(vectors, dtype=np.float32)
    # Pre-normalize so retrieval can use a plain dot product = cosine.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    np.savez(INDEX_PATH, vectors=matrix, ids=np.asarray(ids))
    META_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"wrote {INDEX_PATH.name} ({matrix.nbytes/1024:.1f} KB) "
          f"and {META_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
