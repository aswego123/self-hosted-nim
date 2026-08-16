"""
=======================================================================
Lesson 06 — Vision (multimodal): sending an image to a VLM NIM
=======================================================================

WHAT THIS TEACHES
-----------------
- How multimodal messages are shaped in the OpenAI schema:
    A single "user" message whose `content` is a LIST of "parts".
    Each part has a `type`: "text" or "image_url".
- Two ways to send an image:
    1. A public URL      -> {"url": "https://..."}
    2. A base64 data URL -> {"url": "data:image/jpeg;base64,..."}
  We use option 1 by default (nothing extra to ship). A local-file
  helper is included below so you can swap easily.
- Which model to pick: any VLM NIM. We use the one you were already
  playing with on build.nvidia.com.

WHY IT MATTERS
--------------
The moment you can pass images to a NIM, you can build:
  * document understanding (invoices, forms, screenshots)
  * accessibility (describe-the-image)
  * visual QA agents
All using the SAME chat completions endpoint. No new API to learn.

HOW TO RUN
----------
../.venv/bin/python Hosted-NIM-API/06-vision-multimodal.py
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Vision-capable NIM you already used on build.nvidia.com.
VLM_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# A safe, public image from Wikimedia Commons (an NVIDIA GPU photo).
# You can swap to any public URL, or to a local file (see helper below).
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/"
    "Nvidia_H100.jpg/640px-Nvidia_H100.jpg"
)


def file_to_data_url(path: str | Path) -> str:
    """Turn a local image file into a data: URL suitable for image_url.

    Not used by default, but shown so you know how. Small images only —
    the request body is limited (typically a few MB).
    """
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.startswith("nvapi-REPLACE"):
        print("ERROR: set NVIDIA_API_KEY in Hosted-NIM-API/.env", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    # KEY IDEA — MULTIMODAL MESSAGE SHAPE:
    # `content` is a LIST of parts, not a plain string.
    # Order matters: the model reads text and images left-to-right.
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "What is shown in this image? "
                    "Reply in 2 short sentences and name the product if you can."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": SAMPLE_IMAGE_URL},
            },
        ],
    }

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[user_message],
        temperature=0.2,
        max_tokens=256,
    )

    print(f"model : {resp.model}")
    print(f"image : {SAMPLE_IMAGE_URL}\n")
    print("--- reply ---")
    print(resp.choices[0].message.content)
    if resp.usage:
        print(
            f"\n[usage] prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
