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
    1. Public URL       -> {"url": "https://..."}    NIM fetches it.
    2. base64 data URL  -> {"url": "data:image/jpeg;base64,..."}
                                                     You fetch/encode.
- Why we default to base64:
    Some CDNs (Wikimedia in particular) block certain thumbnail sizes
    or hotlink patterns. When NIM tries to fetch on your behalf and the
    CDN says 400, NIM wraps it as a 500 back to you. Sending base64
    removes that variable entirely — NIM never touches the network.

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

import requests
from dotenv import load_dotenv
from openai import OpenAI

VLM_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Public source image. We download it locally, then send its bytes to
# NIM as base64 — avoiding hotlink policies on the CDN side.
# picsum.photos is a free stable image service with no hotlink limits.
# id=237 is a well-known photo of a Labrador puppy. Deterministic URL.
SOURCE_IMAGE_URL = "https://picsum.photos/id/237/512/512"

# Where we cache the downloaded image so we don't re-download every run.
CACHE_DIR = Path(__file__).resolve().parent / "assets"
CACHE_PATH = CACHE_DIR / "sample_gpu.jpg"

# Wikimedia requires a descriptive User-Agent (per their robot policy).
UA = "NIM-Learning-Repo/1.0 (educational; contact: local user)"


def download_image(url: str, dest: Path) -> Path:
    """Fetch the image with a proper UA and cache it under assets/.

    We do the fetch ourselves so we control the User-Agent and can send
    the bytes to NIM inline (base64), instead of asking NIM to hotlink.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def file_to_data_url(path: str | Path) -> str:
    """Turn a local image file into a data: URL for the image_url field."""
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

    # Step 1 — fetch the image locally (once, then cached).
    local_path = download_image(SOURCE_IMAGE_URL, CACHE_PATH)
    size_kb = local_path.stat().st_size / 1024
    print(f"image cached at : {local_path}  ({size_kb:.1f} KB)")

    # Step 2 — turn its bytes into a data: URL. This is what NIM will read.
    data_url = file_to_data_url(local_path)
    print(f"data url length : {len(data_url):,} chars (base64 is ~33% larger than raw)\n")

    # KEY IDEA — MULTIMODAL MESSAGE SHAPE:
    # `content` is a LIST of parts, not a plain string.
    # Order matters: the model reads text and images left-to-right.
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe what you see in this image in 2 short sentences. "
                    "Mention any animals, colors, and mood you can infer."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
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
    print("--- reply ---")
    print(resp.choices[0].message.content)
    if resp.usage:
        # Notice prompt_tokens will be substantially higher than a
        # text-only prompt — the image itself is encoded into tokens.
        print(
            f"\n[usage] prompt={resp.usage.prompt_tokens} "
            f"completion={resp.usage.completion_tokens} "
            f"total={resp.usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
