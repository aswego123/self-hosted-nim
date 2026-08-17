# 01 — First launch: pull & run your first NIM

Prerequisite: `00-preflight.sh` prints **"Ready to launch a NIM."**
If any check failed, fix it first (see the *How to fix* section it prints).

This runbook has you launching **meta/llama-3.1-8b-instruct** — the smallest
general-purpose LLM NIM. It fits comfortably on a single 24 GB GPU (A10,
RTX 3090/4090, L4, A100). Smaller alternatives are listed in `05-swap-models.md`.

---

## Step 1 — Load your NGC API key into the shell

```bash
cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
set -a; source "Self-hosted NIM in Docker/.env"; set +a
echo "key length: ${#NGC_API_KEY}"   # sanity check — should be > 30
```

`set -a` / `set +a` = auto-export every var loaded from `.env` so `docker`
inherits them.

## Step 2 — Log in to NVIDIA's container registry

```bash
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Expected: `Login Succeeded`. `'$oauthtoken'` is a literal string (single-quoted
so the shell doesn't expand it) — that's the actual username NGC expects for
API-key auth.

If you see `401 unauthorized`:
- Regenerate the NGC key with the **NGC Catalog** service enabled.
- Re-run the `docker login` command.

## Step 3 — Create the model-cache directory

Models are 15–40 GB and take minutes to download. Cache them on the host so
re-launches are instant.

```bash
mkdir -p ~/.cache/nim
```

We'll mount `~/.cache/nim` into the container. NIM will find any previously
downloaded weights + engines there.

## Step 4 — Pull the image (optional but nice to do explicitly)

```bash
docker pull nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

The image itself is ~10 GB. Model weights are pulled separately at container
start (see Step 5).

## Step 5 — Run the NIM

```bash
docker run --rm --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u "$(id -u):$(id -g)" \
  -p 8000:8000 \
  --name llama31-8b \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Anatomy of this command

| Flag | Why |
|---|---|
| `--rm` | Auto-remove the container when it exits (keep host clean). |
| `--gpus all` | Give the container **all** NVIDIA GPUs. Restrict with `--gpus '"device=0"'` if you have several. |
| `--shm-size=16G` | Some NIMs need large shared memory for inter-process comms. 16G is safe. |
| `-e NGC_API_KEY` | Pass the env var into the container so it can download weights. |
| `-v ~/.cache/nim:/opt/nim/.cache` | Persist model weights & compiled TensorRT engines across runs. |
| `-u "$(id -u):$(id -g)"` | Run as your user so files under the cache mount aren't owned by root. |
| `-p 8000:8000` | Expose the OpenAI-compatible API on host port 8000. |
| `--name llama31-8b` | Named container so you can `docker logs -f llama31-8b`. |

### What you'll see in the logs (roughly)

```
Detected GPU: NVIDIA A100-SXM4-40GB
Selected profile: tensorrt_llm-h100-fp16-tp1-throughput
Downloading model weights ...  ██████████ 15.3 GB / 15.3 GB
Building TensorRT-LLM engine ...   (takes 3-8 minutes the first time)
Loading model ...
Uvicorn running on http://0.0.0.0:8000
```

**First launch takes 5–15 minutes** (download + engine build). Subsequent
launches with the same GPU are **~30 seconds** (engine cached).

Leave this terminal running. Open a **second terminal** for the next steps.

## Step 6 — Verify readiness

In a new terminal:

```bash
# Liveness (is the process alive at all?)
curl -s http://localhost:8000/v1/health/live | jq

# Readiness (is the model loaded & ready to serve?)
curl -s http://localhost:8000/v1/health/ready | jq

# What models are being served?
curl -s http://localhost:8000/v1/models | jq
```

`/v1/health/ready` returning `{"object":"health.response","message":"Service is ready."}`
means: **weights loaded, engine ready, you can send inference requests**. This
is the health probe Kubernetes will use in Step 5.

## Step 7 — Send your first local request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role":"user","content":"Reply with exactly: local NIM is live."}],
        "max_tokens": 20
      }' | jq
```

The response shape is **identical** to the hosted API you saw in Step 2 —
`choices[0].message.content`, `usage`, `nvext` telemetry. That's the whole
point.

## Common failures & fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-container-cli: initialization error` | Container Toolkit not configured | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Container exits with `CUDA out of memory` | GPU too small for FP16 | Try a smaller model — see `05-swap-models.md`. |
| Stuck at "Building engine" for >20 min | First-time build for your GPU | Normal on H100/A100 fresh builds. Be patient. |
| `port already in use` | Something else on 8000 | Use `-p 8001:8000` and adjust the client. |
| `docker: permission denied` | User not in `docker` group | `sudo usermod -aG docker $USER` → log out → back in. |

## Step 8 — When done

```bash
# In the terminal running the container:
Ctrl+C           # gracefully stop

# Or from another terminal:
docker stop llama31-8b
```

Model cache stays under `~/.cache/nim`, so next launch is fast.

---

**Next:** open `02-call-local-nim.py` — reuses Step 2 client code, pointed at
`localhost:8000`. The "code is identical" moment.
