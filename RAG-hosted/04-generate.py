"""
=======================================================================
Lesson 04 — Generate: retrieve → rerank → grounded answer with citations
=======================================================================

WHAT THIS TEACHES
-----------------
- Full RAG in one file:
    query  →  embed  →  top-N cosine  →  rerank  →  top-K passages
        →  build context block  →  chat NIM  →  answer + citations
- Prompt engineering for grounded answers:
    * Give the model a short, non-negotiable system prompt.
    * Number the passages ([P07], [P12], ...) so the model can cite them.
    * Tell it to say "I don't know" when context is insufficient.
- Why the citation format matters: it turns hallucinations into
  auditable failures — a claim without a [Pxx] is suspect.

WHY LOCAL EVENTUALLY WINS THIS STEP
-----------------------------------
The generation call sees your documents AND your users' queries. Local
NIM keeps both on your machine — mandatory for medical, legal, and
internal-code corpora.

HOW TO RUN
----------
python RAG-hosted/04-generate.py "how does NIM speed up LLM serving?"
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
CHAT_MODEL = "meta/llama-3.1-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"

RETRIEVE_N = 20
FINAL_K = 5

HERE = Path(__file__).parent
INDEX_PATH = HERE / "index.npz"
META_PATH = HERE / "meta.json"
ENV_PATH = HERE.parent / "Hosted-NIM-API" / ".env"

DEFAULT_QUERY = "How does NIM make LLM deployment fast?"

SYSTEM_PROMPT = (
    "You are a precise assistant that answers ONLY from the provided context passages.\n"
    "Rules:\n"
    "  1. Every factual claim MUST cite the supporting passage id(s) in square brackets, e.g. [P07].\n"
    "  2. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided context.\"\n"
    "  3. Do not invent passage ids. Do not use outside knowledge.\n"
    "  4. Prefer 3-6 short bullet points. Be concise."
)


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def retrieve_stage1(query_vec, matrix, ids, meta_by_id, n):
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:n]
    return [(float(scores[i]), ids[i], meta_by_id[ids[i]]["text"]) for i in order]


def rerank_stage2(query, candidates, api_key, k):
    reranker = NVIDIARerank(model=RERANK_MODEL, api_key=api_key, top_n=k)
    docs = [Document(page_content=t, metadata={"id": pid}) for _, pid, t in candidates]
    ranked = reranker.compress_documents(query=query, documents=docs)
    return [
        (float(d.metadata.get("relevance_score", 0.0)), d.metadata["id"], d.page_content)
        for d in ranked
    ]


def build_context_block(passages: list[tuple[float, str, str]]) -> str:
    lines = ["<context>"]
    for _, pid, text in passages:
        lines.append(f"[{pid}] {text}")
    lines.append("</context>")
    return "\n".join(lines)


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
    stage1 = retrieve_stage1(q_vec, matrix, ids, meta_by_id, n=RETRIEVE_N)
    stage2 = rerank_stage2(query, stage1, api_key=api_key, k=FINAL_K)

    context = build_context_block(stage2)
    used_ids = [pid for _, pid, _ in stage2]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    answer = resp.choices[0].message.content or ""

    print(f"query   : {query}")
    print(f"context : {used_ids}\n")
    print("answer:")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
