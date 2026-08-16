"""
=======================================================================
Lesson 04 — LangChain: NIM as a first-class ChatModel
=======================================================================

WHAT THIS TEACHES
-----------------
- How to plug NIM into LangChain via `ChatNVIDIA`.
- Why frameworks matter: once a model is a LangChain `ChatModel`, you
  get for free:
    * Prompt templates
    * Output parsers (JSON, pydantic, etc.)
    * Chains and LCEL composition (|)
    * Tools / function calling
    * Memory + agents (LangGraph)
- Streaming through LangChain: same iterator pattern, different API.

WHY IT MATTERS
--------------
Step 4 (RAG) and any future agent work in this repo will use LangChain
components. This file is where the framework meets the NIM.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/04-langchain-chat-nvidia.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

MODEL = "meta/llama-3.1-8b-instruct"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    # ChatNVIDIA wraps the hosted NIM endpoint as a LangChain ChatModel.
    # Under the hood it hits the same /v1/chat/completions URL.
    # If NVIDIA_API_KEY is in the environment it's picked up automatically,
    # but we pass it explicitly here for clarity.
    llm = ChatNVIDIA(
        model=MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=256,
    )

    # LangChain messages are typed objects, not plain dicts.
    # SystemMessage, HumanMessage, AIMessage map to system/user/assistant.
    messages = [
        SystemMessage(
            content="You are a concise senior backend engineer. Use bullets."
        ),
        HumanMessage(
            content="In 3 bullets, why would I use NVIDIA NIM instead of vLLM?"
        ),
    ]

    # ----- 1. One-shot invoke ----------------------------------------------
    print("--- invoke() (non-streaming) ---")
    result = llm.invoke(messages)
    # `result` is an AIMessage. .content is the string reply.
    print(result.content)
    print()

    # ----- 2. Streaming ----------------------------------------------------
    # Same as OpenAI SDK streaming, but yields AIMessageChunk objects.
    # We concatenate .content to render tokens as they arrive.
    print("--- stream() (token-by-token) ---")
    for chunk in llm.stream(messages):
        # Some NIM models emit a separate "reasoning_content" field in
        # additional_kwargs — used by nemotron / thinking-style models.
        # Print it dimmed so you can see it if present.
        extra = chunk.additional_kwargs or {}
        if "reasoning_content" in extra and extra["reasoning_content"]:
            print(extra["reasoning_content"], end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
