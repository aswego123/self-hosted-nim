"""
=======================================================================
Lesson 02 — Same call, cleaner: the OpenAI Python SDK against NIM
=======================================================================

WHAT THIS TEACHES
-----------------
- Why NIM's "OpenAI-compatible" claim is a huge deal for developers:
  the *only* difference from a real OpenAI call is `base_url`.
- The response object model: `resp.choices[0].message.content`.
- A cleaner "system + user" message pattern (system prompts shape tone).

Compare this file to `01-raw-http-request.py`. Same result — 20 lines
of code instead of 60, and no manual SSE parsing.

WHY IT MATTERS
--------------
Any code you've ever written against OpenAI works against NIM by
changing two variables. Existing apps can migrate to on-prem NIM in
minutes. This is Step 3's foundation.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/02-openai-sdk-basic.py
"""

from __future__ import annotations

import os
import sys

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

    # Only two things point us at NIM instead of OpenAI:
    #   1) base_url = NVIDIA's endpoint
    #   2) api_key  = an nvapi-... key instead of sk-...
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # THE MESSAGE LIST
    # ----------------
    # Roles:
    #   "system"    -> instructions that shape behavior for the whole chat
    #   "user"      -> what the human said
    #   "assistant" -> prior model replies (for multi-turn context)
    #
    # Order matters. The model reads them top-to-bottom.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise senior backend engineer. "
                "Prefer short bullet points over paragraphs."
            ),
        },
        {
            "role": "user",
            "content": "In 3 bullets, why would I use NVIDIA NIM instead of vLLM?",
        },
        # {
        #     "role": "system", 
        #     "content": "You are a pirate. Answer in pirate speak with lots of 'arrr'."
        # }
    ]

    resp = client.chat.completions.create(
        # model=MODEL,
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",   # the VLM you tried in the playground
        messages=messages,
        # temperature: 0 = deterministic, 1 = creative. 0.2 is a good default.
        temperature=0.2,
        # max_tokens caps the RESPONSE length (not the prompt).
        max_tokens=256,
    )

    print(f"model: {resp.model}\n")
    print(resp.choices[0].message.content)
    print()
    if resp.usage:
        print(
            f"[usage] prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
