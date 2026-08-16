"""
=======================================================================
Lesson 03 — Streaming: token-by-token responses via the OpenAI SDK
=======================================================================

WHAT THIS TEACHES
-----------------
- What `stream=True` actually gives you: an iterator over "deltas".
- Why streaming matters:
    * Time-to-first-token (TTFT) — user sees output almost immediately.
    * Perceived latency drops dramatically for long answers.
    * You can start post-processing (parsing JSON, updating UI) mid-stream.
- The delta pattern: each chunk carries a small piece of `content`,
  not the full growing string. You accumulate as it arrives.

Compare this to lesson 01 which did the SAME thing at the HTTP layer.
The SDK version handles all the SSE parsing for you.

WHY IT MATTERS
--------------
Every ChatGPT-style UI you've seen streams. When you self-host NIM
(Step 3), streaming is a one-line change too — this pattern is identical.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/03-openai-sdk-streaming.py
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "meta/llama-3.1-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    prompt = (
        "Write a 6-line haiku-style poem about a GPU serving an LLM. "
        "Number each line."
    )

    print(f">> prompt: {prompt}\n")
    print("--- streamed response ---")

    started = time.perf_counter()
    first_token_at: float | None = None
    total_chars = 0

    # `stream=True` returns a generator. Each iteration yields one chunk.
    # A chunk is a ChatCompletionChunk with the SAME shape as a full
    # response, but choices[0].delta is a small partial payload.
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
        stream=True,
    )

    for chunk in stream:
        # `choices` may briefly be empty on the very first chunk on some
        # providers; guard for it.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = delta.content or ""
        if piece:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(piece, end="", flush=True)
            total_chars += len(piece)

    elapsed = time.perf_counter() - started
    ttft = (first_token_at - started) if first_token_at else float("nan")
    print("\n\n--- timing ---")
    print(f"time-to-first-token : {ttft:.2f} s")
    print(f"total wall time     : {elapsed:.2f} s")
    print(f"chars streamed      : {total_chars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
