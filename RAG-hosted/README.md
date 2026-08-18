# RAG-hosted — Step 4 of the NVIDIA NIM learning path

Build a small but **complete** Retrieval-Augmented Generation (RAG) pipeline
using **hosted NVIDIA NIMs** — no GPU, no Docker required.

Everything here runs against `https://integrate.api.nvidia.com/v1`.
The same code will run against a self-hosted NIM (Step 3) by changing
`base_url` to `http://localhost:8000/v1`. That's the whole point.

---

## Why this step matters

RAG is not one model — it's a **pipeline of three model roles** that need to
work together. NVIDIA ships a purpose-built NIM for each role, all speaking
the same OpenAI schema:

```
                 ┌────────────────────────┐
   user query ──►│  1. Embed  (nv-embedqa)│──┐
                 └────────────────────────┘  │
                                             ▼
                                    ┌───────────────────┐
                                    │  cosine top-k     │
                                    │  over vector store│
                                    └───────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────┐
                 │  2. Rerank (nv-rerankqa)│  (cross-encoder, per query)
                 └────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────┐
                 │  3. Generate (llama-*) │──► grounded answer + citations
                 └────────────────────────┘
```

### Why local models eventually replace the hosted ones

| RAG step | Runs per… | Why local wins |
|---|---|---|
| **Embed corpus** | ingest (once) | Millions of chunks. Hosted = huge bill + rate limits. Local = one GPU-hour. |
| **Embed query** | every request | Saves 50–200 ms round trip on the hot path. |
| **Rerank** | every request × top-k | Cross-encoder is called on **N passages per query**. Cost multiplier. |
| **Generate** | every request | Docs + queries never leave your network. Long context is free. |

### Why NIM (vs. rolling your own server)

- **Same OpenAI schema across embed / rerank / chat.** One SDK, three roles.
- **TensorRT-LLM under the hood** — auto-picks the optimized engine for your GPU.
- **Continuous batching + streaming** built in (critical for embed/rerank load).
- **`/v1/health/*` and `/metrics`** out of the box (Prometheus-ready).
- **Hosted mirror = zero-friction dev loop.** Prototype hosted today, `docker run` tomorrow, **same code**.

---

## Files, in the order to run them

| # | File | What you learn |
|---|---|---|
| — | [`corpus.json`](corpus.json) | ~24 short passages across 4 topics + distractors. Your "knowledge base". |
| 01 | [`01-ingest.py`](01-ingest.py) | Load corpus → batch-embed with `nv-embedqa-e5-v5` → save `index.npz` + `meta.json`. This is what a vector DB does "offline". |
| 02 | [`02-retrieve.py`](02-retrieve.py) | Load the index → embed a query → cosine top-k. The whole retrieval story in ~60 lines. |
| 03 | [`03-rerank.py`](03-rerank.py) | Take top-20 from `02`, run a **rerank NIM** cross-encoder, keep top-5. Print a before/after table so you can *see* the lift. |
| 04 | [`04-generate.py`](04-generate.py) | Feed reranked context into a chat NIM with a strict "answer only from context, cite passage IDs" system prompt. Full RAG in one file. |
| — | [`gold.json`](gold.json) | 5 hand-labeled (query → relevant passage ids) pairs. Your eval set. |
| 05 | [`05-eval.py`](05-eval.py) | Compute **recall@k** and **MRR** with retrieval alone vs. retrieval + rerank. Prove the reranker earns its keep. |

---

## Setup

You already have the venv and `NVIDIA_API_KEY` from Step 2 — reuse them.

```bash
cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
source .venv/bin/activate

# Sanity check your key still works:
python Hosted-NIM-API/smoke-test-key.py
```

No new dependencies are needed — everything in this folder uses packages
already in [`Hosted-NIM-API/requirements.txt`](../Hosted-NIM-API/requirements.txt)
(`openai`, `numpy`, `python-dotenv`, `langchain-nvidia-ai-endpoints`).

The `.env` file at `Hosted-NIM-API/.env` is loaded automatically by every
script in this folder — no need to copy it.

---

## Walkthrough

```bash
# 1. Build the vector index (one-time, ~2s for 24 passages)
python RAG-hosted/01-ingest.py

# 2. Try a query and see raw top-k
python RAG-hosted/02-retrieve.py "How does NIM make LLM deployment fast?"

# 3. Same query, but with rerank on — watch positions shuffle
python RAG-hosted/03-rerank.py "How does NIM make LLM deployment fast?"

# 4. End-to-end: retrieve → rerank → generate a grounded answer with citations
python RAG-hosted/04-generate.py "How does NIM make LLM deployment fast?"

# 5. Measure quality on the gold set
python RAG-hosted/05-eval.py
```

---

## What to look for in the output

### After `02-retrieve.py`
The top-1 is usually right for easy queries, but scan positions 2–10 — that's
where reranking earns its keep. If the "obvious" answer is at rank 4, the LLM
will still see it (we send top-20 to the reranker).

### After `03-rerank.py`
The rerank scores use a **different scale** (log-odds, not cosine).
Don't compare them numerically to the embedding scores — compare the
**ordering**. You'll see items jump multiple places.

### After `04-generate.py`
The answer should:
- Only make claims that are supported by the retrieved passages.
- Cite passage IDs like `[P07]` after each claim.
- Say "I don't know" when the corpus doesn't cover the question.

### After `05-eval.py`
You should see rerank **increase MRR** (mean reciprocal rank) even when
recall@5 stays the same — that means the right passage moves higher up,
which the generator cares about a lot.

---

## Models used

| Role | Model | Notes |
|---|---|---|
| Embed | `nvidia/nv-embedqa-e5-v5` | 1024-dim; QA-optimized; asymmetric query/passage prompts. |
| Rerank | `nvidia/nv-rerankqa-mistral-4b-v3` | Cross-encoder, higher accuracy than embed-only. Swap-in: `nvidia/llama-3.2-nv-rerankqa-1b-v2`. |
| Generate | `meta/llama-3.1-8b-instruct` | Small enough to be cheap, big enough to follow the "cite the passages" rule. |

All three are available hosted at build.nvidia.com **and** as `nvcr.io/nim/*`
containers — that's the whole hosted-to-local flip story.

---

## Flipping to self-hosted later (Step 3)

Every script here has a `BASE_URL` constant at the top. When you have a GPU
and have followed [`Self-hosted NIM in Docker/01-first-launch.md`](../Self-hosted%20NIM%20in%20Docker/01-first-launch.md),
change it to `http://localhost:8000/v1` — nothing else changes. Same schema,
same models, same code.

---

## What's deliberately NOT in this step

- **Real vector DB (FAISS / pgvector / chromadb).** Numpy is enough for 24
  passages; scaling the retriever is orthogonal to learning RAG.
- **Chunking strategy.** The corpus is pre-chunked. Chunking is its own
  rabbit hole — worth its own lesson later.
- **Hybrid search (BM25 + dense).** Great next add-on, but not needed to
  see the effect of rerank.
- **Streaming answers.** Add later; the pattern is identical to
  `03-openai-sdk-streaming.py` from Step 2.
- **Guardrails / structured output.** Also great next add-ons.
