"""
=======================================================================
Lesson 01 — Raw HTTP: what actually goes over the wire to NIM
=======================================================================

WHAT THIS TEACHES
-----------------
- The exact HTTP request NIM expects: URL, headers, JSON body.
- What "Server-Sent Events" (SSE) streaming looks like as raw bytes
  (the `data: {...}` lines and the `data: [DONE]` terminator).
- Why you almost never write code at this level — but should read it
  ONCE so `stream=True` in the SDK stops feeling like magic.

The payload here mirrors the snippet NVIDIA gave you on
build.nvidia.com — but with the key pulled from .env instead of
hardcoded.

WHY IT MATTERS
--------------
Every other SDK (openai, langchain-nvidia-ai-endpoints, llamaindex-nvidia)
is a thin wrapper around exactly this HTTP call. Understanding the raw
shape makes debugging trivial: if a wrapper misbehaves, curl the
endpoint and compare.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/01-raw-http-request.py
"""

from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.1-8b-instruct"
# STREAM = True
STREAM = False


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    # HEADERS
    # -------
    # Authorization: standard "Bearer <token>" — same as OpenAI.
    # Accept: tells the server which response format we want.
    #   "text/event-stream" -> streamed chunks (SSE)
    #   "application/json"  -> one JSON body at the end
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if STREAM else "application/json",
    }

    # PAYLOAD
    # -------
    # This is the "OpenAI chat completions" schema. NIM implements it
    # 1:1 so any OpenAI-compatible client works.
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Explain NVIDIA NIM in 3 sentences to a backend engineer.",
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": STREAM,
    }

    # requests.post with stream=True gives us a generator over lines
    # instead of buffering the whole body.
    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=STREAM)
    response.raise_for_status()

    if not STREAM:
        # Non-streaming response: one big JSON object.
        print(json.dumps(response.json(), indent=2))
        return 0

    # STREAMING (SSE) FORMAT — read this carefully once:
    #
    # The server sends a series of lines like:
    #   data: {"id":"...","choices":[{"delta":{"content":"NVIDIA"}}]}
    #   data: {"id":"...","choices":[{"delta":{"content":" NIM"}}]}
    #   ...
    #   data: [DONE]
    #
    # Each `data:` line is one chunk. `[DONE]` is a sentinel, not JSON.
    print("--- streamed reply (assembled) ---")
    for raw in response.iter_lines():
        if not raw:
            continue  # SSE uses blank lines as event separators
        line = raw.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        # The final chunk from NIM often has choices=[] and only usage
        # metadata — skip it. Real content chunks always have >=1 choice.
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        piece = delta.get("content", "")
        if piece:
            print(piece, end="", flush=True)
    print()  # trailing newline
    return 0


if __name__ == "__main__":
    sys.exit(main())
