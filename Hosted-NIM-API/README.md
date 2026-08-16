# Hosted-NIM-API — Step 2 of the NVIDIA NIM learning path

Small, self-contained Python scripts that call the **hosted** NVIDIA NIM
endpoint at `https://integrate.api.nvidia.com/v1`. No Docker, no GPU —
you just need a free NVIDIA developer API key.

## Setup (once)

```bash
# 1. Rotate / create an API key at https://build.nvidia.com  (Account -> API Keys)
# 2. Copy the template and paste your key:
cp Hosted-NIM-API/.env.example Hosted-NIM-API/.env
$EDITOR Hosted-NIM-API/.env

# 3. The venv + deps are already installed at the repo root:
../.venv/bin/python --version
```

## Files, in the order to read/run them

| # | File | What you learn |
|---|---|---|
| 00 | [smoke-test-key.py](smoke-test-key.py) | Confirm your key + `base_url` work. Prints token usage. |
| 01 | [01-raw-http-request.py](01-raw-http-request.py) | What NIM's HTTP request/response really look like, including SSE streaming. |
| 02 | [02-openai-sdk-basic.py](02-openai-sdk-basic.py) | Same call in ~20 lines with the OpenAI SDK. The "drop-in" story. |
| 03 | [03-openai-sdk-streaming.py](03-openai-sdk-streaming.py) | Token-by-token streaming, with TTFT + wall-time measurement. |
| 04 | [04-langchain-chat-nvidia.py](04-langchain-chat-nvidia.py) | `ChatNVIDIA` as a LangChain ChatModel — invoke + stream. |
| 05 | [05-embeddings.py](05-embeddings.py) | An embedding NIM + cosine similarity. Foundation for RAG. |
| 06 | [06-vision-multimodal.py](06-vision-multimodal.py) | Sending an image URL to a VLM NIM. |

## Run any script

```bash
cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
./.venv/bin/python Hosted-NIM-API/smoke-test-key.py
./.venv/bin/python Hosted-NIM-API/01-raw-http-request.py
# ...etc
```

## Concepts covered (Step 2 checklist)

- [ ] Base URL + model naming (`vendor/model-id`)
- [ ] API key hygiene (env var, never in source)
- [ ] OpenAI-compatible schema (chat, streaming, embeddings, vision)
- [ ] Raw SSE vs SDK-managed streaming
- [ ] LangChain integration path
- [ ] Query vs passage embeddings + cosine similarity
- [ ] Multimodal message parts (`type: image_url`)

## Not in Step 2 (comes later)

- Step 3 — Run a NIM container locally on your GPU with `docker run`.
- Step 4 — Build a RAG app (embeddings NIM + vector DB + rerank NIM + chat NIM).
- Step 5 — Kubernetes deployment via the NIM Operator + Helm.
- Step 6 — Domain NIMs: Riva (speech), BioNeMo (science), more VLMs.
