#!/usr/bin/env bash
# =====================================================================
# NIM Docker preflight — checks every prerequisite for self-hosting a NIM
# =====================================================================
#
# Runs a series of read-only diagnostic checks and prints a green/red
# report card. Nothing is installed, downloaded, or modified.
#
# Exit 0 = ready to launch a NIM.
# Exit 1 = one or more prereqs missing; see fix hints printed below.
#
# Usage: bash "Self-hosted NIM in Docker/00-preflight.sh"

set -u  # trap use of undefined variables

# ---- pretty-print helpers -------------------------------------------------
if [[ -t 1 ]]; then
  GREEN=$'\e[32m'; RED=$'\e[31m'; YEL=$'\e[33m'; BLD=$'\e[1m'; DIM=$'\e[2m'; RST=$'\e[0m'
else
  GREEN=""; RED=""; YEL=""; BLD=""; DIM=""; RST=""
fi

PASS=0; FAIL=0; WARN=0
declare -a FIXES=()

pass()  { printf "  ${GREEN}✓${RST} %s\n" "$1"; PASS=$((PASS+1)); }
fail()  { printf "  ${RED}✗${RST} %s\n" "$1"; FAIL=$((FAIL+1)); FIXES+=("$2"); }
warn()  { printf "  ${YEL}⚠${RST} %s\n" "$1"; WARN=$((WARN+1)); }
info()  { printf "  ${DIM}·${RST} %s\n" "$1"; }
head1() { printf "\n${BLD}%s${RST}\n" "$1"; }

# ---- 1. NVIDIA driver & GPU -----------------------------------------------
head1 "1) NVIDIA driver & GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$GPU" && -n "$DRIVER" ]]; then
    pass "GPU detected: $GPU"
    info "driver: $DRIVER"
    VRAM_GB=$(( VRAM_MIB / 1024 ))
    if   (( VRAM_GB >= 24 )); then pass "VRAM: ${VRAM_GB} GB (comfortable for 8B-class NIMs)"
    elif (( VRAM_GB >= 16 )); then warn "VRAM: ${VRAM_GB} GB (tight for 8B FP16 — pick a smaller model or quantized profile)"
    elif (( VRAM_GB >= 12 )); then warn "VRAM: ${VRAM_GB} GB (Phi-3-mini or 4-bit models only)"
    else                            fail "VRAM: ${VRAM_GB} GB (too small for typical NIMs)" \
      "Consider running NIMs on a workstation with >=16 GB VRAM, or use hosted NIMs (Step 2)."
    fi
    # Major version check on driver.
    DRIVER_MAJ=${DRIVER%%.*}
    if (( DRIVER_MAJ >= 535 )); then pass "driver major $DRIVER_MAJ (>=535, OK for CUDA 12 NIMs)"
    else                             fail "driver major $DRIVER_MAJ (<535)" \
      "Upgrade NVIDIA driver to 535+ (check your distro or download from https://www.nvidia.com/drivers)."
    fi
  else
    fail "nvidia-smi found but returned no GPU info" \
      "Reinstall/repair NVIDIA driver so nvidia-smi lists a GPU."
  fi
else
  fail "nvidia-smi not found" \
    "Install NVIDIA driver: https://www.nvidia.com/drivers  or your distro's package (e.g. sudo apt install nvidia-driver-550)."
fi

# ---- 2. Docker --------------------------------------------------------------
head1 "2) Docker daemon"
if command -v docker >/dev/null 2>&1; then
  DVER=$(docker --version 2>/dev/null | head -1)
  pass "docker CLI: $DVER"
  if docker info >/dev/null 2>&1; then
    pass "docker daemon reachable without sudo"
  else
    fail "docker daemon not reachable (permission denied or not running)" \
      "Fix: sudo systemctl start docker && sudo usermod -aG docker \$USER  then log out & back in."
  fi
else
  fail "docker not installed" \
    "Install Docker Engine: https://docs.docker.com/engine/install/"
fi

# ---- 3. NVIDIA Container Toolkit -------------------------------------------
head1 "3) NVIDIA Container Toolkit (GPU access from inside containers)"
if command -v nvidia-ctk >/dev/null 2>&1; then
  CTK=$(nvidia-ctk --version 2>&1 | head -1)
  pass "nvidia-ctk installed: $CTK"
else
  warn "nvidia-ctk CLI not found (may still work if runtime is configured)"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  info "smoke test: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
  if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    pass "GPU is visible inside a Docker container"
  else
    fail "GPU NOT visible inside container (--gpus all failed)" \
      "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   Then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
else
  warn "skipping container GPU test (Docker not usable)"
fi

# ---- 4. Free disk ----------------------------------------------------------
head1 "4) Free disk space (models are big)"
FREE_HOME_KB=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
FREE_HOME_GB=$(( FREE_HOME_KB / 1024 / 1024 ))
if   (( FREE_HOME_GB >= 100 )); then pass "\$HOME has ${FREE_HOME_GB} GB free"
elif (( FREE_HOME_GB >= 50  )); then warn "\$HOME has ${FREE_HOME_GB} GB free (fine for one 8B model, cramped for two)"
else                                 fail "\$HOME has only ${FREE_HOME_GB} GB free" \
  "Free at least 50 GB — models & engines are 15-40 GB each. NIM cache lives at ~/.cache/nim."
fi

# ---- 5. NGC API key --------------------------------------------------------
head1 "5) NGC API key (for pulling nvcr.io/nim/* images)"
ENV_FILE="Self-hosted NIM in Docker/.env"
if [[ -f "$ENV_FILE" ]]; then
  if grep -qE '^NGC_API_KEY=nvapi-[A-Za-z0-9_-]{20,}' "$ENV_FILE"; then
    pass ".env has NGC_API_KEY set"
  else
    fail ".env exists but NGC_API_KEY is not set correctly" \
      "Edit '$ENV_FILE' and add:  NGC_API_KEY=nvapi-...  (create key at https://ngc.nvidia.com  ->  Setup -> API Keys)"
  fi
else
  fail ".env not found at '$ENV_FILE'" \
    "cp 'Self-hosted NIM in Docker/.env.example' '$ENV_FILE'  then paste NGC_API_KEY=nvapi-..."
fi

# ---- Summary ---------------------------------------------------------------
head1 "Summary"
printf "  pass=%d  warn=%d  fail=%d\n" "$PASS" "$WARN" "$FAIL"

if (( FAIL > 0 )); then
  head1 "How to fix"
  n=1
  for f in "${FIXES[@]}"; do
    printf "  ${BLD}%d.${RST} %s\n" "$n" "$f"
    n=$((n+1))
  done
  printf "\n${RED}Not ready to launch a NIM yet.${RST} Fix the items above, then re-run this script.\n"
  exit 1
fi

if (( WARN > 0 )); then
  printf "\n${YEL}Warnings present but no hard failures — proceed with a small model.${RST}\n"
fi

printf "\n${GREEN}Ready to launch a NIM.${RST}  Next: follow ${BLD}01-first-launch.md${RST}.\n"
exit 0
