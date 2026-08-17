"""
=======================================================================
Step 3, Lesson 04 — Hosted vs Local: measure the difference
=======================================================================

WHAT THIS TEACHES
-----------------
- Send the SAME prompt at BOTH endpoints (hosted + your local NIM).
- Measure two metrics per call:
    * TTFT   (time-to-first-token)  -> user-perceived latency
    * total  (wall time)              -> throughput indicator
- See that the CLIENT CODE IS IDENTICAL. The only difference is a URL.
- Local usually wins on latency (no internet round-trip) but shares the
  GPU with other apps on your box. Hosted wins on scalability.

WHY IT MATTERS
--------------
Real teams often use BOTH: hosted for burst, local for cost/privacy.
This script shows how a hybrid strategy would evaluate them.

PREREQUISITE
------------
- Step 2 done, `Hosted-NIM-API/.env` has NVIDIA_API_KEY.
- Local NIM running (01-first-launch.md) and 02-call-local-nim.py works.

HOW TO RUN
----------
    cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
    ./.venv/bin/python "Self-hosted NIM in Docker/04-compare-hosted-vs-local.py"
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ----- config ----------------------------------------------------------------
HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_BASE_URL = os.getenv("LOCAL_NIM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("LOCAL_NIM_MODEL", "meta/llama-3.1-8b-instruct")

# We load Step 2's .env for the hosted key so both endpoints work.
HOSTED_ENV = Path(__file__).resolve().parent.parent / "Hosted-NIM-API" / ".env"

PROMPT = "In one paragraph (~80 words), explain what NVIDIA NIM is to a backend engineer."
N_SAMPLES = 3   # per endpoint, we run the prompt 3x and take the median


@dataclass
class Sample:
    label: str
    ttft_s: float
    total_s: float
    prompt_tokens: int
    completion_tokens: int


def stream_and_measure(client: OpenAI, label: str) -> Sample:
    """Fire one streaming request; measure TTFT and total wall time."""
    started = time.perf_counter()
    ttft: float | None = None
    completion_tokens = 0
    prompt_tokens = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.2,
        max_tokens=200,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        # Usage-only final chunk carries no choices.
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        if piece and ttft is None:
            ttft = time.perf_counter() - started
        completion_tokens = completion_tokens or (completion_tokens + len(piece.split()))
    total = time.perf_counter() - started
    return Sample(
        label=label,
        ttft_s=ttft if ttft is not None else float("nan"),
        total_s=total,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def median_sample(samples: list[Sample]) -> Sample:
    return Sample(
        label=samples[0].label,
        ttft_s=statistics.median(s.ttft_s for s in samples),
        total_s=statistics.median(s.total_s for s in samples),
        prompt_tokens=samples[0].prompt_tokens,
        completion_tokens=int(statistics.median(s.completion_tokens for s in samples)),
    )


def local_up() -> bool:
    ready = LOCAL_BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/health/ready"
    try:
        return requests.get(ready, timeout=3).status_code == 200
    except requests.RequestException:
        return False


def main() -> int:
    # ---- clients ----
    load_dotenv(HOSTED_ENV)
    hosted_key = os.getenv("NVIDIA_API_KEY")
    if not hosted_key or hosted_key.startswith("nvapi-REPLACE"):
        print(f"ERROR: NVIDIA_API_KEY missing at {HOSTED_ENV}", file=sys.stderr)
        return 1
    hosted = OpenAI(base_url=HOSTED_BASE_URL, api_key=hosted_key)

    if not local_up():
        print(
            f"ERROR: local NIM not ready at {LOCAL_BASE_URL}\n"
            "  Start it first — see 01-first-launch.md",
            file=sys.stderr,
        )
        return 2
    local = OpenAI(base_url=LOCAL_BASE_URL, api_key="not-used")

    # ---- run N samples on each ----
    print(f"prompt: {PROMPT!r}")
    print(f"model : {MODEL}")
    print(f"runs  : {N_SAMPLES} per endpoint\n")

    hosted_runs = [stream_and_measure(hosted, "hosted") for _ in range(N_SAMPLES)]
    local_runs = [stream_and_measure(local, "local") for _ in range(N_SAMPLES)]

    h = median_sample(hosted_runs)
    l = median_sample(local_runs)

    # ---- report ----
    print("=" * 68)
    print(f"{'endpoint':<10} {'ttft (s)':>10} {'total (s)':>11} "
          f"{'prompt tok':>12} {'completion tok':>16}")
    print("-" * 68)
    for s in (h, l):
        try:
            print(
                f"{s.label:<10} {s.ttft_s:>10.2f} {s.total_s:>11.2f} "
                f"{s.prompt_tokens:>12} {s.completion_tokens:>16}"
            )
        except OpenAIError as e:
            print(f"{s.label:<10} ERROR: {e}")
    print("=" * 68)

    # ---- interpretation ----
    if l.ttft_s < h.ttft_s:
        print(
            f"\nlocal TTFT is {(h.ttft_s - l.ttft_s) * 1000:.0f} ms faster "
            "(no internet round-trip)."
        )
    else:
        print(
            f"\nhosted TTFT is {(l.ttft_s - h.ttft_s) * 1000:.0f} ms faster "
            "(NVIDIA's cluster is beefier than your GPU)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
