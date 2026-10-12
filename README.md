# EmbeddingGemma 2 on a plain CPU: local, offline search over text, photos, speech and video

▶ Video: https://www.youtube.com/watch?v=NlwMFVB0ibw (ProAI Clips)

Video timestamps below come from the final cut.

Google DeepMind's **EmbeddingGemma 2** (released 6 Oct 2026, Apache 2.0) puts text, code, images, video and audio into one 768-number vector space.
This repo is the exact code from the video. It runs on an ordinary computer: we used Linux, 8 CPU cores, ~15 GB RAM and **no GPU**.
No sign-in, no token and no API key are needed, because `google/embeddinggemma-2` on Hugging Face is not gated.

## Quick start

Everything at once (creates `.venv` with CPU PyTorch, downloads the weights and data, runs all four demos):
```bash
git clone https://github.com/schoolofai/embeddinggemma2-cpu-search
cd embeddinggemma2-cpu-search
./run_all.sh
```
Needs Python 3.10+ (we used 3.13), about 3 GB of disk, and about 6 GB of free RAM for the multimodal step.
Works on Linux and macOS. On Windows, use WSL. Your timings will differ from ours; the scores should match closely.

Or step by step:

### Step 1: install (CPU-only PyTorch + sentence-transformers 6.1+)
Video: [3:32](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=212s)
```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchcodec
.venv/bin/pip install -r requirements.txt
```
Expected: pip ends with `Successfully installed ... sentence-transformers-6.1.x ...`.

### Step 2: download the weights anonymously (~1.5 GB)
Video: [3:21](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=201s)
```bash
export HF_HOME="$PWD/.hf-cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/embeddinggemma-2', token=False))"
```
Expected: a path ending in `.hf-cache/hub/models--google--embeddinggemma-2/snapshots/...`. No login prompt.

### Step 3: fetch the demo data
```bash
.venv/bin/python fetch_data.py
```
Implementation: `fetch_data.py` pulls 35 English Wikipedia intros (CC BY-SA 4.0), 10 lead photos from Wikimedia Commons (licence and author saved to `data/images.json`), and Google's sample photo, speech clip and cat video from the official EmbeddingGemma 2 notebook.
Expected: `data/corpus.json`, `data/images/*.jpg`, `data/media/roses-are.wav`, `data/media/cat_dozing.mp4`.

### Step 4: modular loading (text-only 270M up to full 740M)
Video: [1:40](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=100s)
```bash
export HF_HUB_OFFLINE=1
.venv/bin/python measure_configs.py
```
Implementation: loads each configuration in a fresh process with `config_kwargs` (`{"vision_config": None, "audio_config": None}` = text only).
Expected (our box): text only `271.0M` params, about +1.4 GB RAM in float32; full `744.4M`, about +3.3 GB. See `results/configs.log`.

### Step 5: local text search, keyword baseline, other languages, Matryoshka
Video: [3:56](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=236s), [5:50](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=350s)
```bash
.venv/bin/python search_text.py
```
Implementation: documents as `title: ... | text: ...`, queries with `prompt_name="SearchQuery"`, cosine via `model.similarity`; BM25 keyword baseline; 9 non-English questions; `truncate_dim` 768/512/256/128 with `normalize_embeddings=True`.
Expected:
```
Q: why does the night sky glow green near the poles
   0.755  Aurora
english        BM25 hit@1=0.83   EG2 hit@1=1.00
cross_lingual  BM25 hit@1=0.11   EG2 hit@1=1.00
```
Scores may differ in the third decimal place. See `results/text_search.log`.

### Step 6: photos, speech, video and one mixed search box
Video: [4:39](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=279s)
```bash
.venv/bin/python multimodal.py
```
Expected: text to photo 10/10, photo to article 10/10; `roses-are.wav` best match `Roses are red, violets are blue.` (about 0.88); the cat video best match `a cat dozing` (about 0.69); "a sleepy cat" puts `[video] cat_dozing.mp4` first.
On a CPU, expect about 3.4 s per photo at the default 280 vision tokens and about 50 s for the 30 s video at 32 frames. See `results/multimodal.log`.

### Step 7: precision and video frame budget
Video: [5:17](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=317s), [6:32](https://www.youtube.com/watch?v=NlwMFVB0ibw&t=392s)
```bash
.venv/bin/python precision_and_video.py
```
Expected: bfloat16 vs float32 cosine about 0.9996; video `max_frames` 4 is about 10 s with the same ranking as 32. The model card says never use float16. See `results/precision_and_video.log`.

### Optional: Ollama on your PC or Mac (not tested in the video)
```bash
ollama pull embeddinggemma-2
curl http://localhost:11434/api/embed -d '{"model": "embeddinggemma-2", "input": "Why is the sky blue?"}'
```
From Ollama's official library page. Tags: `270m`, `440m`, `570m`, `740m`.

## Files
| File | What it does |
|---|---|
| `run_all.sh` | One command: venv, weights, data, all demos |
| `common.py` | Offline, token-free loading in float32, timing and memory helpers |
| `fetch_data.py` | Wikipedia intros, Commons photos (licences recorded), Google's sample media |
| `measure_configs.py` | Memory and speed of each modular configuration |
| `search_text.py` | Text search, BM25 baseline, cross-lingual, task prefixes, Matryoshka |
| `multimodal.py` | Text↔photo, speech, video, one mixed index |
| `precision_and_video.py` | float32 / bfloat16 / float16, video frame budget |
| `results/` | Our logs and JSON from 8 Oct 2026 |
| `docs/photo_credits.json` | Photo authors and licences |

## Official sources
- Google launch post: https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
- Model card (Hugging Face): https://huggingface.co/google/embeddinggemma-2
- Model card (Google AI for Developers): https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- Developer guide: https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/
- Google AI Edge post: https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/
- Ollama: https://ollama.com/library/embeddinggemma-2

## Credits and licence
Code: MIT (see `LICENSE`). Model: Apache 2.0, and deployments must follow the Gemma Prohibited Use Policy (model card).
Text: Wikipedia, CC BY-SA 4.0. Photos: Wikimedia Commons authors in `docs/photo_credits.json`. Sample photo, speech clip and video: Google's Gemma cookbook.
Benchmarks quoted in the video are Google's. Hands-on numbers are ours, from one shared CPU box, so treat timings as ballpark.
