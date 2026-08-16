"""
=======================================================================
Lesson 00 — Smoke test: is my NVIDIA NIM API key wired up correctly?
=======================================================================

WHAT THIS TEACHES
-----------------
1. Where the hosted NIM lives on the internet:
      https://integrate.api.nvidia.com/v1
2. That NIM speaks the SAME wire format as OpenAI, so we can literally
   `from openai import OpenAI` and just change `base_url`.
3. How to keep the API key OUT of source code (12-factor style):
   - .env file  ->  loaded by python-dotenv  ->  read via os.getenv
4. How to read the response object: model, message content, token usage.

WHY IT MATTERS
--------------
Every later lesson (streaming, embeddings, vision, LangChain, RAG) will
reuse these same three ingredients: base_url + api_key + a model name.
If this file runs green, everything downstream will "just work".

HOW TO RUN
----------
1) cp .env.example .env         # then paste your ROTATED nvapi-... key
2) ../.venv/bin/python Hosted-NIM-API/smoke-test-key.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# --- Config ---------------------------------------------------------------
# Model name format on NIM is always "vendor/model-id".
# meta/llama-3.1-8b-instruct is the smallest general-purpose chat model
# available on hosted NIM — fast, cheap, universally accessible.
MODEL = "meta/llama-3.1-8b-instruct"

# The one URL you have to remember. Every hosted NIM (chat, embeddings,
# rerank, vision) is served under this base.
BASE_URL = "https://integrate.api.nvidia.com/v1"


def main() -> int:
    # load_dotenv() looks for a .env file in the current dir and walks up.
    # It sets any KEY=VALUE lines it finds into os.environ.
    load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print(
            "ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env "
            "(copy .env.example -> .env and paste your rotated key).",
            file=sys.stderr,
        )
        return 1

    # THE KEY INSIGHT of the whole NIM story:
    # The OpenAI Python SDK works unchanged. We only override `base_url`.
    # Everything else — messages format, response schema, streaming — is
    # identical to OpenAI's API.
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Reply with exactly: NIM is live."}
            ],
            # temperature=0 makes the reply deterministic — good for a
            # smoke test where we want a predictable string back.
            temperature=0,
            max_tokens=16,
        )
    except OpenAIError as e:
        # Common causes: bad key, revoked key, no credits, network block.
        print(f"ERROR calling NIM: {e}", file=sys.stderr)
        return 2

    # `resp` is a pydantic model. The important fields are:
    #   resp.model                        -> which model actually served
    #   resp.choices[0].message.content   -> the assistant's reply text
    #   resp.usage                        -> token accounting (billing)
    print(f"model:  {resp.model}")
    print(f"reply:  {resp.choices[0].message.content!r}")
    if resp.usage:
        print(
            f"tokens: prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
