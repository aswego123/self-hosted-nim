# Self-hosted NIM in Docker — Step 3 of the NVIDIA NIM learning path

> **⚠️ GPU REQUIRED.** All lessons in this folder run only on a machine with a
> supported NVIDIA GPU + driver + Container Toolkit. If you're on a CPU-only
> machine, treat this folder as a **turnkey runbook** you can execute later
> on a cloud GPU (Lambda / RunPod / Vast / AWS / GCP) or on new hardware.

Same code and API as Step 2 — the only difference is where the model runs:

| | Hosted (Step 2) | **Self-hosted (Step 3)** |
|---|---|---|
| `base_url` | `https://integrate.api.nvidia.com/v1` | `http://localhost:8000/v1` |
| Where GPU is | NVIDIA's cluster | Your machine |
| Data privacy | Sent to NVIDIA | Stays on your box |
| Cold start | Zero | 5–15 min first run (weights + engine build) |

## Files, in the order to run them

| # | File | What you do |
|---|---|---|
| 00 | [`00-preflight.sh`](00-preflight.sh) | Diagnostic. Verifies GPU, Docker, Container Toolkit, disk, NGC key. Nothing installed. |
| 01 | [`01-first-launch.md`](01-first-launch.md) | `docker login nvcr.io` → `docker pull` → `docker run` the NIM → watch it come up → curl `/v1/health/ready`. |
| 02 | [`02-call-local-nim.py`](02-call-local-nim.py) | Reuse the Step 2 client code, pointed at `localhost:8000`. Streaming + non-streaming. |
| 03 | [`03-inspect-container.md`](03-inspect-container.md) | Explore the running NIM: `/v1/health/*`, `/v1/models`, `/metrics`, `/docs`, logs, host cache. |
| 04 | [`04-compare-hosted-vs-local.py`](04-compare-hosted-vs-local.py) | Same prompt at hosted + local endpoints. Median TTFT + total wall time. |
| 05 | [`05-swap-models.md`](05-swap-models.md) | Universal launch pattern; cheat-sheet for chat / embed / rerank / VLM NIMs; running the RAG trio side-by-side. |

## The two API keys (yes, two)

| Variable | Purpose | File |
|---|---|---|
| `NVIDIA_API_KEY` (Step 2) | Call **hosted** NIMs at `integrate.api.nvidia.com` | `Hosted-NIM-API/.env` |
| `NGC_API_KEY` (Step 3)   | Pull **container images** from `nvcr.io/nim/*`   | `Self-hosted NIM in Docker/.env` |

Different pages of nvidia.com, different scopes — keep them separate.
See "NGC API key — how to get it" below.

## Phase 0 — Preflight (do this first!)

```bash
cd /home/anjali/Downloads/AgenticIQ_ai/NVDIA-NIM
bash "Self-hosted NIM in Docker/00-preflight.sh"
```

You want to see:

```
Ready to launch a NIM.  Next: follow 01-first-launch.md.
```

If preflight fails, its **How to fix** section prints the exact commands you
need. Don't skip this step — the toolkit setup catches most surprises.

## NGC API key — how to get it

1. Go to https://ngc.nvidia.com (same NVIDIA account as build.nvidia.com).
2. Click your profile → **Setup** → **Personal API Key** → **Generate**.
3. Under services/scopes, include **NGC Catalog** so it can pull `nvcr.io/nim/*`.
4. Copy the key (starts with `nvapi-`) and paste it into `.env`:

```bash
cp "Self-hosted NIM in Docker/.env.example" "Self-hosted NIM in Docker/.env"
$EDITOR "Self-hosted NIM in Docker/.env"     # paste NGC_API_KEY=nvapi-...
```

## What preflight checks (in one glance)

| # | Check | Why it matters |
|---|---|---|
| 1 | NVIDIA driver + GPU + VRAM | NIMs require an NVIDIA GPU with driver ≥ 535. |
| 2 | Docker installed + reachable without sudo | You'll `docker run` a lot. |
| 3 | NVIDIA Container Toolkit | Lets the GPU be visible inside `--gpus all`. |
| 4 | Free disk in `$HOME` | Models + engines are 15–40 GB each, cached under `~/.cache/nim`. |
| 5 | NGC API key in `.env` | Needed to pull `nvcr.io/nim/*`. |

## Troubleshooting quick reference

- **`nvidia-smi` command not found** → NVIDIA driver isn't installed.
- **`lspci | grep -Ei 'vga|3d'` shows only Intel/AMD** → No NVIDIA hardware. Steps 3 and 5 can't run here; do Step 4 (RAG) against hosted NIMs instead.
- **`docker info` says permission denied** →
  `sudo usermod -aG docker $USER && newgrp docker`
- **`--gpus all` fails but `nvidia-smi` works on host** → Container Toolkit
  missing. See NVIDIA's install guide, then
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
- **`docker pull nvcr.io/nim/...` returns 401** → NGC login didn't stick:
  `echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin`

## Not in Step 3

- ❌ Kubernetes / Helm / NIM Operator → **Step 5**
- ❌ Multi-GPU tensor-parallel setup → advanced, requires ≥ 2 GPUs
- ❌ Custom fine-tuned models → catalog NIMs only
- ❌ RAG orchestration → **Step 4** (which will happily use these local NIMs
  once you have GPU access — just point at `localhost:8000/8001/8002`)
