#!/usr/bin/env bash
# Rebuild everything from scratch on a CPU-only Linux box. No sign-in, no token, no API keys.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision torchcodec
  .venv/bin/pip install -q -r requirements.txt
fi
export HF_HOME="$PWD/.hf-cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1; unset HF_TOKEN || true
# 1) weights: google/embeddinggemma-2 is NOT gated (Apache 2.0) -> anonymous download, ~1.5 GB
.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/embeddinggemma-2', token=False))"
# 2) data: Wikipedia intros + lead photos (CC licences recorded) + Google's cookbook sample media
[ -f data/corpus.json ] || .venv/bin/python fetch_data.py
export HF_HUB_OFFLINE=1   # everything below runs fully offline
.venv/bin/python measure_configs.py   | tee results/configs.log
.venv/bin/python search_text.py       | tee results/text_search.log
.venv/bin/python multimodal.py        | tee results/multimodal.log
.venv/bin/python precision_and_video.py | tee results/precision_and_video.log
