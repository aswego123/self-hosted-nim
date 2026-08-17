"""
=======================================================================
Lesson 05.3 — Nearest-neighbor visualization (seeing the embedding space)
=======================================================================

WHAT THIS TEACHES
-----------------
- Embeddings live in HIGH-dimensional space (1024 dims for e5-v5).
  You cannot picture 1024 dimensions — but you can PROJECT them to 2D.
- PCA (Principal Component Analysis) is the standard projection:
    * finds the 2 directions of maximum variance in the data,
    * projects every vector onto those 2 axes,
    * preserves as much "spread" as possible in the flattened view.
- After PCA, cosine similarity in 1024D roughly becomes visual distance
  on the plot. Related passages cluster; unrelated ones fly apart.
- This is *exactly* how real teams debug bad retrieval — plot, look,
  fix the corpus or the query.

WHY IT MATTERS
--------------
Once you SEE the geometry, embeddings stop feeling like a black box.
You'll have concrete intuitions like:
    "the query landed in the wrong neighborhood — my chunks are wrong"
    "these two chunks are duplicates, no wonder both get retrieved"
    "off-topic chunks form their own cluster — dedupe / re-tag them"

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/05.3-nn-visualization.py
    then open the generated PNG file:
../embed_scatter.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"

OUT_PNG = Path(__file__).resolve().parent.parent / "embed_scatter.png"


def embed(client: OpenAI, texts: list[str], input_type: str) -> np.ndarray:
    """Embed a list of strings, return an (N, D) NumPy array."""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # Three topical CLUSTERS so we can see them separate cleanly on the plot.
    # (Real corpora look messier, but this makes the geometry obvious.)
    passages = [
        # --- LLM serving cluster ---
        "NVIDIA NIM packages optimized inference microservices as Docker containers.",
        "TensorRT-LLM optimizes large language models for NVIDIA GPUs.",
        "vLLM is a high-throughput inference engine for LLMs.",
        "Triton Inference Server serves models with dynamic batching.",
        # --- DevOps / infra cluster ---
        "Kubernetes is a container orchestration platform.",
        "Prometheus collects metrics from services over HTTP.",
        "Grafana visualizes time-series data as dashboards.",
        "Helm charts package Kubernetes manifests for reuse.",
        # --- Off-topic (should be far from both clusters) ---
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Chocolate chip cookies are best served warm.",
        "Beethoven composed nine symphonies before his death.",
    ]

    # Two queries, each aimed at a different cluster.
    queries = [
        "How do I serve big language models fast on GPUs?",
        "How do I monitor a production Kubernetes cluster?",
    ]

    # Color-code by cluster so you can visually check the model got it right.
    passage_labels = (
        ["llm-serving"] * 4 + ["devops"] * 4 + ["off-topic"] * 3
    )
    colors = {"llm-serving": "#1f77b4", "devops": "#2ca02c", "off-topic": "#7f7f7f"}

    print(f"embedding {len(passages)} passages + {len(queries)} queries ...")
    p_vecs = embed(client, passages, input_type="passage")
    q_vecs = embed(client, queries, input_type="query")
    print(f"  passage matrix shape: {p_vecs.shape}")
    print(f"  query   matrix shape: {q_vecs.shape}")

    # ---- PCA to 2 dims ---------------------------------------------------
    # We fit PCA on the PASSAGES only. That way the 2D axes are chosen to
    # spread OUR CORPUS out well — then we project the queries into the
    # same coordinate system. This matches how a real retriever works:
    # the index geometry is fixed at ingest time; queries are visitors.
    pca = PCA(n_components=2)
    p_xy = pca.fit_transform(p_vecs)
    q_xy = pca.transform(q_vecs)

    var = pca.explained_variance_ratio_
    print(
        f"  PCA keeps {var.sum() * 100:5.1f}% of variance in 2D "
        f"(axis1={var[0] * 100:.1f}%, axis2={var[1] * 100:.1f}%)"
    )

    # ---- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))

    for (x, y), text, label in zip(p_xy, passages, passage_labels):
        ax.scatter(x, y, s=140, c=colors[label], edgecolors="black", zorder=3)
        # Show a short excerpt next to each dot.
        ax.annotate(
            text[:45] + ("…" if len(text) > 45 else ""),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    for (x, y), q in zip(q_xy, queries):
        ax.scatter(x, y, s=260, c="red", marker="*", edgecolors="black", zorder=4)
        ax.annotate(
            f"QUERY: {q[:50]}",
            (x, y),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="darkred",
        )

    # Legend for the passage clusters (queries are always red stars).
    for label, colour in colors.items():
        ax.scatter([], [], c=colour, s=120, edgecolors="black", label=label)
    ax.scatter([], [], c="red", marker="*", s=200, edgecolors="black", label="query")
    ax.legend(loc="best")

    ax.set_title(
        "Embedding space projected to 2D (PCA)\n"
        f"kept {var.sum() * 100:.1f}% of variance across 2 axes"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"\nsaved -> {OUT_PNG}")

    # ---- Interpretation guide -------------------------------------------
    # WHAT TO LOOK FOR IN THE PNG:
    # 1. Three colored clusters should be visually distinct. Blue (LLM
    #    serving) and green (DevOps) will be near each other because both
    #    are "software infra"; grey (off-topic) will float alone.
    # 2. The red star for query 1 should land INSIDE (or right next to)
    #    the blue cluster.
    # 3. The red star for query 2 should land INSIDE the green cluster.
    # 4. If a star lands in the WRONG cluster, that's a bad retrieval —
    #    same signal you'd chase in production RAG debugging.
    return 0


if __name__ == "__main__":
    sys.exit(main())
