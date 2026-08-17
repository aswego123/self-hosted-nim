"""
=======================================================================
Lesson 05b — Batching & latency: the single biggest RAG perf lever
=======================================================================

WHAT THIS TEACHES
-----------------
1. The huge gap between "one call per text" and "one call for N texts".
2. How throughput scales as batch size grows (1, 8, 32, 64, 96).
3. Where the sweet spot is: for a hosted API, network/auth overhead
   dominates small batches; the GPU dominates large ones.
4. The two vocabulary words you'll hear forever after:
     - LATENCY    = wall time for MY request
     - THROUGHPUT = texts processed per second across the system

WHY IT MATTERS
--------------
In real RAG you ingest thousands of chunks. If you send them one at a
time you might wait 10 minutes. Batched into groups of 64 you're done
in seconds. Same money, same GPU — you just asked correctly.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/05b-batching-latency.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# A pool of realistic-length passages. We'll sample from this list to
# build batches of various sizes. Keeping strings similar in length
# makes the comparison fair.
POOL = [
    "NVIDIA NIM packages optimized inference microservices as Docker containers.",
    "TensorRT-LLM optimizes large language models for NVIDIA GPUs.",
    "vLLM is a high-throughput inference engine for LLMs.",
    "Triton Inference Server serves models with dynamic batching.",
    "Kubernetes is a container orchestration platform.",
    "Prometheus collects metrics from services over HTTP.",
    "PostgreSQL is a mature open-source relational database.",
    "Redis is an in-memory key-value store often used for caching.",
    "gRPC uses HTTP/2 and Protobuf for high-performance RPC.",
    "React is a component-based UI library maintained by Meta.",
    "Kafka is a distributed log-based messaging platform.",
    "FastAPI is a modern Python web framework built on Starlette.",
]


@dataclass
class Result:
    label: str
    n_texts: int
    wall_s: float
    per_text_ms: float
    throughput_tps: float


def embed_batch(client: OpenAI, texts: list[str]) -> None:
    """Fire one embeddings call, ignore the return — we only care about time."""
    client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": "passage", "truncate": "END"},
    )


def sample_texts(n: int) -> list[str]:
    """Return exactly n texts by cycling through the pool."""
    return [POOL[i % len(POOL)] for i in range(n)]


def time_one_call(client: OpenAI, texts: list[str]) -> float:
    """Time a single API call and return elapsed seconds."""
    t0 = time.perf_counter()
    embed_batch(client, texts)
    return time.perf_counter() - t0


def bench_sequential(client: OpenAI, n: int, warm: bool = True) -> Result:
    """Send n texts as n separate API calls. This is the slow path."""
    texts = sample_texts(n)
    if warm:
        embed_batch(client, texts[:1])  # avoid measuring cold-start on 1st call
    t0 = time.perf_counter()
    for t in texts:
        embed_batch(client, [t])
    wall = time.perf_counter() - t0
    return Result(
        label=f"sequential x{n}",
        n_texts=n,
        wall_s=wall,
        per_text_ms=wall / n * 1000,
        throughput_tps=n / wall,
    )


def bench_batched(client: OpenAI, n: int, warm: bool = True) -> Result:
    """Send n texts in ONE API call. This is the fast path."""
    texts = sample_texts(n)
    if warm:
        embed_batch(client, texts[:1])
    wall = time_one_call(client, texts)
    return Result(
        label=f"batched x{n}",
        n_texts=n,
        wall_s=wall,
        per_text_ms=wall / n * 1000,
        throughput_tps=n / wall,
    )


def print_row(r: Result) -> None:
    print(
        f"  {r.label:<18}  wall={r.wall_s:6.2f}s   "
        f"per-text={r.per_text_ms:7.1f} ms   "
        f"throughput={r.throughput_tps:6.1f} texts/s"
    )


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # =====================================================================
    # PART 1 — Head-to-head: 32 texts, one-at-a-time vs one big batch
    # =====================================================================
    N = 32
    print(f"\n=== PART 1: {N} texts sequential vs batched ===\n")
    seq = bench_sequential(client, N)
    bat = bench_batched(client, N)
    print_row(seq)
    print_row(bat)
    speedup = seq.wall_s / bat.wall_s
    print(f"\n  --> batched is {speedup:.1f}x faster end-to-end for the same {N} texts.")

    # =====================================================================
    # PART 2 — Sweep batch sizes to find where throughput plateaus
    # =====================================================================
    print("\n=== PART 2: batch-size sweep (single call per batch) ===\n")
    sizes = [1, 4, 8, 16, 32, 64, 96]
    results: list[Result] = []
    for n in sizes:
        # Repeat 3 times and take the median to smooth out network jitter.
        samples = []
        for _ in range(3):
            samples.append(time_one_call(client, sample_texts(n)))
        median_wall = statistics.median(samples)
        r = Result(
            label=f"batch={n:>3}",
            n_texts=n,
            wall_s=median_wall,
            per_text_ms=median_wall / n * 1000,
            throughput_tps=n / median_wall,
        )
        results.append(r)
        print_row(r)

    # Small-print: highlight the batch size with the best throughput.
    best = max(results, key=lambda r: r.throughput_tps)
    print(
        f"\n  --> peak throughput at {best.label} "
        f"({best.throughput_tps:.1f} texts/s, {best.per_text_ms:.1f} ms/text)."
    )

    # =====================================================================
    # WHAT TO NOTICE (write these into your mental model)
    # =====================================================================
    # * per-text ms falls FAST as batch grows: 1 -> 4 -> 8 is huge.
    # * Somewhere between 32 and 96 the curve flattens: the GPU is now
    #   the bottleneck, not the network round-trip. That's the sweet spot.
    # * Wall time for batch=64 is NOT 64x the wall time for batch=1. Often
    #   it's only 2-4x. That's the compounding math that makes RAG viable.
    # * In real RAG: batch your ingestion pass into groups of 64-128,
    #   never one-at-a-time. This one habit saves hours per import.
    return 0


if __name__ == "__main__":
    sys.exit(main())
