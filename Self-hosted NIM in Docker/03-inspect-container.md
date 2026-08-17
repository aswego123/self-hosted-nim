# 03 — Inspect a running NIM (endpoints, logs, metrics, cache)

Prerequisite: NIM container is running (`01-first-launch.md`).

This lesson is about knowing *what's inside the box* — the same knowledge you'd
use to operate NIM in production.

---

## 1. Container-level introspection

```bash
docker ps                                     # is the container up?
docker logs -f llama31-8b                     # follow the container logs
docker top llama31-8b                         # processes inside
docker stats llama31-8b                       # live CPU / RAM / GPU (limited)
nvidia-smi                                    # host-side GPU utilization
```

The **`nvidia-smi`** view is more useful than `docker stats` for GPU workloads
— you'll see the container's PID under the "Processes" section.

## 2. NIM HTTP endpoints (the operator's toolkit)

All of these are exposed on the same port as the OpenAI API (default 8000).

### Liveness — "is the process alive?"

```bash
curl -s http://localhost:8000/v1/health/live | jq
```

Returns 200 as soon as the HTTP server is up. **Doesn't mean model is loaded.**
This is the probe Kubernetes uses to decide whether to restart the pod.

### Readiness — "can it serve inference right now?"

```bash
curl -s http://localhost:8000/v1/health/ready | jq
```

Returns 200 only after weights are loaded and the engine is compiled. This is
the probe Kubernetes uses to route traffic. Waits until this passes before
sending inference.

### `/v1/models` — what's being served

```bash
curl -s http://localhost:8000/v1/models | jq
```

Returns an OpenAI-compatible model list. In self-hosted NIMs this is usually
one model per container — the one you launched.

### `/v1/chat/completions` — inference (already used)

Same as Step 2. Streaming, non-streaming, JSON mode, function calling — all
supported when the underlying model supports them.

### `/v1/embeddings` — for embedding NIMs

Same schema as OpenAI. Only present when the container is an embedding NIM
(see `05-swap-models.md`).

### `/metrics` — Prometheus scrape target

```bash
curl -s http://localhost:8000/metrics | head -40
```

You'll see metrics like:
- `nim_request_count_total{model=...,status=...}`
- `nim_request_duration_seconds_bucket{...}`
- `nim_active_requests`
- `nim_gpu_kv_cache_usage_perc`

This is what you point Prometheus at in production. Grafana dashboards for NIM
are on NVIDIA's GitHub.

### `/docs` — Swagger UI (auto-generated OpenAPI spec)

Open `http://localhost:8000/docs` in a browser. It's the interactive API
documentation NIM generates from its own OpenAPI schema — a great way to
explore every endpoint.

## 3. Look inside the container

```bash
# open a shell inside the running container
docker exec -it llama31-8b bash

# once inside:
ls /opt/nim                                   # NIM install dir
ls /opt/nim/.cache                            # your mounted host cache
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
env | grep -i nim                             # NIM-specific env vars
exit                                          # or Ctrl+D
```

## 4. Inspect the model cache on the HOST

Because we mounted `~/.cache/nim` into the container, everything's on your host FS:

```bash
du -sh ~/.cache/nim
find ~/.cache/nim -maxdepth 3 -type d
```

You'll see subdirs for downloaded weights and for the compiled TensorRT-LLM
engine specific to your GPU. **Do not delete this** unless disk-pressured — a
fresh build takes minutes.

## 5. Useful log lines to search for

```bash
docker logs llama31-8b 2>&1 | grep -E 'Selected profile|GPU|Uvicorn|error' -i
```

Key lines:
- `Detected GPU: ...` — what NIM saw.
- `Selected profile: ...` — which engine profile (dtype, batching strategy) it chose.
- `Building TensorRT-LLM engine ...` — first-time engine build.
- `Uvicorn running on ...` — HTTP server bound; safe to hit `/v1/health/live` now.
- Anything with `ERROR` or `Traceback`.

## 6. Kill it cleanly

```bash
docker stop llama31-8b
# cache stays on host; next launch skips the download + build.
```

---

**Next:** `04-compare-hosted-vs-local.py` — side-by-side benchmark of the
same prompt against your local NIM and the hosted NIM.
