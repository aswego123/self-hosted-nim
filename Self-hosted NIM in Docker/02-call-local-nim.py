"""
=======================================================================
Step 3, Lesson 02 — Call your LOCAL NIM (the punchline of the whole step)
=======================================================================

WHAT THIS TEACHES
-----------------
- The client code is IDENTICAL to Step 2's `02-openai-sdk-basic.py`.
  Only the `base_url` changes — from
      https://integrate.api.nvidia.com/v1        (hosted, Step 2)
  to
      http://localhost:8000/v1                   (self-hosted, Step 3)
- Self-hosted NIMs don't check the API key by default. You still MUST
  pass one (the OpenAI SDK requires it), but any non-empty string works.
- Health-check-before-call: we probe /v1/health/ready first and give a
  clear error if the container isn't up yet.

WHY IT MATTERS
--------------
This is the whole "portability" story of NIM. Your app doesn't know or
care whether inference happens on NVIDIA's cluster or on your box.

PREREQUISITE
------------
Follow 01-first-launch.md until you see:
    curl http://localhost:8000/v1/health/ready
    -> {"object":"health.response","message":"Service is ready."}

HOW TO RUN
----------
    cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
    ./.venv/bin/python "Self-hosted NIM in Docker/02-call-local-nim.py"
"""

from __future__ import annotations

import os
import sys
import time

import requests
from openai import OpenAI, OpenAIError

# Default local NIM endpoint. Change host/port if you used a different -p.
LOCAL_BASE_URL = os.getenv("LOCAL_NIM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("LOCAL_NIM_MODEL", "meta/llama-3.1-8b-instruct")

# Self-hosted NIMs don't validate the key, but the OpenAI SDK requires
# a non-empty string. This placeholder is fine.
LOCAL_KEY = "not-used-by-local-nim"


def wait_ready(base_url: str, timeout_s: int = 5) -> bool:
    """Quick check that the NIM is up and ready to serve requests."""
    ready_url = base_url.rstrip("/").removesuffix("/v1") + "/v1/health/ready"
    try:
        r = requests.get(ready_url, timeout=timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main() -> int:
    if not wait_ready(LOCAL_BASE_URL):
        print(
            f"ERROR: no NIM responding at {LOCAL_BASE_URL}\n"
            "  * Is the container running? -> docker ps\n"
            "  * Did readiness pass?       -> curl http://localhost:8000/v1/health/ready\n"
            "  * See 01-first-launch.md",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_KEY)

    print(f"base_url : {LOCAL_BASE_URL}")
    print(f"model    : {MODEL}\n")

    # -------- Non-streaming call --------
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a concise engineer."},
                {"role": "user",   "content": "In one sentence, what is NVIDIA NIM?"},
            ],
            temperature=0,
            max_tokens=64,
        )
    except OpenAIError as e:
        print(f"ERROR calling local NIM: {e}", file=sys.stderr)
        return 2
    dt = time.perf_counter() - t0

    print("--- non-streaming reply ---")
    print(resp.choices[0].message.content)
    if resp.usage:
        print(
            f"\n[usage] prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}   wall={dt:.2f}s"
        )

    # -------- Streaming call (same pattern as Lesson 03 in Step 2) --------
    print("\n--- streaming reply (token-by-token) ---")
    ttft: float | None = None
    started = time.perf_counter()
    for chunk in client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a one-line joke about GPUs."}],
        temperature=0.7,
        max_tokens=64,
        stream=True,
    ):
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - started
            print(piece, end="", flush=True)
    total = time.perf_counter() - started
    print(f"\n\n[stream] ttft={ttft:.2f}s  total={total:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
