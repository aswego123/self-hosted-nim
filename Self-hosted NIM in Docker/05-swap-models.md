# 05 — Swap models: run a different NIM (embedding, VLM, smaller LLM)

The launch pattern from `01-first-launch.md` is **identical** for every NIM.
Only three things change:

1. The **image name** (`nvcr.io/nim/vendor/model`)
2. The **port** (if you want multiple NIMs running side-by-side)
3. Sometimes the **VRAM requirement** (which model you can even fit)

---

## The universal launch pattern

```bash
docker run --rm --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u "$(id -u):$(id -g)" \
  -p <HOST_PORT>:8000 \
  --name <NAME> \
  nvcr.io/nim/<VENDOR>/<MODEL>:latest
```

Substitute the two ALL-CAPS placeholders per the table below.

## Common NIMs cheat-sheet

| Purpose | Image | VRAM (FP16) | Suggested port | `--name` |
|---|---|---|---|---|
| Small chat (this Step's default) | `nvcr.io/nim/meta/llama-3.1-8b-instruct` | ~16 GB | 8000 | `llama31-8b` |
| Bigger chat, better quality | `nvcr.io/nim/meta/llama-3.1-70b-instruct` | ~140 GB (needs multi-GPU or 4-bit) | 8000 | `llama31-70b` |
| Reasoning / NVIDIA-tuned | `nvcr.io/nim/nvidia/llama-3.1-nemotron-70b-instruct` | ~140 GB | 8000 | `nemotron-70b` |
| Tiny chat (fits <16 GB VRAM) | `nvcr.io/nim/microsoft/phi-3-mini-4k-instruct` | ~8 GB | 8000 | `phi3-mini` |
| QA embeddings (for RAG) | `nvcr.io/nim/nvidia/nv-embedqa-e5-v5` | ~2 GB | **8001** | `embed-e5` |
| Reranker (for RAG) | `nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3` | ~8 GB | **8002** | `rerank` |
| Vision-language | `nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | ~16 GB | 8000 | `nemotron-vl` |

Version tags on `catalog.ngc.nvidia.com` change over time; using `:latest` is fine
for learning. Pin to an explicit tag for production.

## Example — the RAG trio on one machine

If you have 24+ GB VRAM you can run chat + embed + rerank concurrently. Split
the GPU among them (or give the reranker its own if you have a second GPU).

### Terminal 1 — chat NIM on :8000

```bash
docker run --rm --gpus all --shm-size=16G \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u "$(id -u):$(id -g)" \
  -p 8000:8000 --name llama31-8b \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Terminal 2 — embedding NIM on :8001

```bash
docker run --rm --gpus all --shm-size=8G \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u "$(id -u):$(id -g)" \
  -p 8001:8000 --name embed-e5 \
  nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest
```

### Terminal 3 — reranker NIM on :8002

```bash
docker run --rm --gpus all --shm-size=8G \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u "$(id -u):$(id -g)" \
  -p 8002:8000 --name rerank \
  nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:latest
```

Once all three health-checks pass:

```bash
curl -s http://localhost:8000/v1/health/ready
curl -s http://localhost:8001/v1/health/ready
curl -s http://localhost:8002/v1/health/ready
```

You have a **fully local RAG backend**. Step 4's RAG code will point at
these URLs instead of the hosted ones — same code, different `base_url`s.

## Pinning a specific GPU per container

If you have multiple GPUs and want to isolate:

```bash
docker run ... --gpus '"device=0"' ... nvcr.io/nim/meta/llama-3.1-8b-instruct:latest   # chat on GPU 0
docker run ... --gpus '"device=1"' ... nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest      # embed on GPU 1
```

## Housekeeping

```bash
docker ps                           # who's running
docker stop llama31-8b embed-e5     # stop by name
docker system prune -f              # reclaim disk from dead containers/images
du -sh ~/.cache/nim                 # see how much cache you're using
```

---

**End of Step 3.** With this runbook you can, on any GPU box:
- launch chat / embed / rerank / VLM NIMs,
- verify readiness,
- serve OpenAI-compatible traffic locally,
- and reuse ALL of your Step 2 client code by changing one `base_url` string.
