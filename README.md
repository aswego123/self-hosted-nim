# NVIDIA NIM — Learning Repo

Hands-on notes and runnable code for exploring **NVIDIA Inference Microservices (NIM)** end-to-end, from the hosted API to self-hosted containers and RAG.

## What's in here

| Path | What it is |
|---|---|
| [NVIDIA_NIM_Study_Guide.md](NVIDIA_NIM_Study_Guide.md) | Structured study guide extracted + expanded from the deck. Start here for concepts. |
| [Hosted-NIM-API/](Hosted-NIM-API/) | **Step 2** — Runnable Python lessons calling the hosted NIM API. Start here for code. |
| `.venv/` | Python 3.13 virtualenv (created locally, not committed). |
| `.gitignore` | Excludes `.env`, `.venv/`, `__pycache__/`. |

## One-time setup

```bash
# 1. Create the venv + install deps (already done once):
python3 -m venv .venv
source .venv/bin/activate
pip install -r Hosted-NIM-API/requirements.txt

# 2. Get a free NVIDIA API key at https://build.nvidia.com  (Account -> API Keys)

# 3. Copy the template and paste your key:
cp Hosted-NIM-API/.env.example Hosted-NIM-API/.env
# then edit Hosted-NIM-API/.env  ->  NVIDIA_API_KEY=nvapi-...
```

## Everyday workflow

```bash
cd /home/anjali/Downloads/NVDIA-NIM
source .venv/bin/activate

# Verify key + connectivity
python Hosted-NIM-API/smoke-test-key.py

# Walk through the lessons in order
python Hosted-NIM-API/01-raw-http-request.py
python Hosted-NIM-API/02-openai-sdk-basic.py
python Hosted-NIM-API/03-openai-sdk-streaming.py
python Hosted-NIM-API/04-langchain-chat-nvidia.py
python Hosted-NIM-API/05-embeddings.py
python Hosted-NIM-API/06-vision-multimodal.py
```

See [Hosted-NIM-API/README.md](Hosted-NIM-API/README.md) for what each lesson teaches.

## Full learning path

1. **Explore** — playground at [build.nvidia.com](https://build.nvidia.com).
2. **Hosted API** — you are here → [Hosted-NIM-API/](Hosted-NIM-API/).
3. **Self-hosted NIM** — run `nvcr.io/nim/...` locally with Docker + GPU.
4. **RAG app** — embedding NIM + vector DB + rerank NIM + chat NIM.
5. **Kubernetes** — NIM Operator + Helm chart.
6. **Domain NIMs** — Riva (speech), BioNeMo (science), more VLMs.

## Security note

- API keys live only in `Hosted-NIM-API/.env` (gitignored).
- Never paste `nvapi-...` values into source files, commits, chats, or screenshots.
- If a key leaks, rotate immediately at build.nvidia.com → API Keys.
