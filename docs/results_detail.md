# EmbeddingGemma 2 on a plain CPU: local, offline, multimodal search demo

Built and run on Thu 8 Oct 2026 (UK time) on a shared Linux box: 8 vCPU Intel Xeon, ~15 GB RAM, **no GPU**.
Every number below was measured here. Timings are noisy because other jobs share the box, so treat them as ballpark figures.

## Can you get the weights without signing in?
**Yes.** `google/embeddinggemma-2` on Hugging Face is **not gated** (`"gated": false`, licence `apache-2.0`).
It downloaded anonymously, with no token, no login and no licence click. That took about 11 s for ~1.53 GB
(one bf16 `model.safetensors` of 1.49 GB plus the tokenizer). For comparison, EmbeddingGemma 1 (`google/embeddinggemma-300m`) is still gated
(`gated: manual`, Gemma licence).

## Run it
```bash
./run_all.sh            # creates .venv (CPU torch), downloads weights + data, runs all demos
# or step by step:
export HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1
.venv/bin/python measure_configs.py     # memory/latency of each modular config
.venv/bin/python search_text.py         # text search, BM25 baseline, cross-lingual, Matryoshka
.venv/bin/python multimodal.py          # photos, speech, video, "one search box"
.venv/bin/python precision_and_video.py # fp32 vs bf16 vs fp16, video frame budget
```
Results land in `results/*.json` and `results/*.log`.

## Files
| File | What it does |
|---|---|
| `fetch_data.py` | Fetches 35 English Wikipedia intros (CC BY-SA 4.0), 10 lead photos from Wikimedia Commons (licence and author saved in `data/images.json`), and Google's own sample media from the official EmbeddingGemma 2 multimodal notebook (a photo, a 3 s speech clip and a 30 s cat video). |
| `common.py` | Offline, token-free loading (`HF_HUB_DISABLE_IMPLICIT_TOKEN`) plus timing and memory helpers. The model loads in float32, as the model card advises for most CPUs. |
| `measure_configs.py` | Loads text / text+image / text+audio / full, each in a fresh process. |
| `search_text.py` | Query → top-3 with scores, a BM25 keyword baseline, 9 non-English queries, a task-prefix ablation, a 159-sentence fact-level test, and Matryoshka 768/512/256/128. |
| `multimodal.py` | Text→photo, photo→article, speech clip, video clip, and one mixed index. |
| `precision_and_video.py` | fp32 / bf16 / fp16 comparison, plus video `max_frames` 4/8/32. |

## What we measured

### 1. Modular loading (model card's `config_kwargs`), CPU, float32
| Config | Params (counted) | fp32 weights | Load time | RSS increase | Text query (median) |
|---|---|---|---|---|---|
| text only | 271.0M | ~1.08 GB | 3.0 s | +1.42 GB | 77 ms |
| text + image | 438.8M | ~1.76 GB | 3.5 s | +2.10 GB | 128 ms* |
| text + audio | 576.6M | ~2.31 GB | 4.0 s | +2.64 GB | 100 ms* |
| full | 744.4M | ~2.98 GB | 4.3 s | +3.31 GB | 107 ms* |

\* The text path is identical in every config, so the spread in query time is box noise.
In bfloat16 the text-only weights are ~0.54 GB. Google's own phone figures (quantized, Pixel 11 Pro) are ~191 MB text-only and ~567 MB full.

### 2. Text search over 35 real Wikipedia intros (text-only, 768d)
- Indexing 35 docs (4,591 tokens) took 6.5 s, about 185 ms per doc batched. One query embeds in a median **~97 ms**.
- Example: *"why does the night sky glow green near the poles"* → **Aurora 0.755**, Rainbow 0.631, Solar eclipse 0.622.
- Example: *"instrument that needs an amplifier to be heard"* → **Electric guitar 0.749**, Jazz 0.657.
- Example (Bengali): *"বাঙালি কবি যিনি নোবেল পুরস্কার পেয়েছিলেন"* → **Rabindranath Tagore 0.781**, Durga Puja 0.594.
- **Keyword search vs. embeddings (top-1 accuracy):**
  - 35 English paraphrase queries: BM25 0.83, EmbeddingGemma 2 **1.00**.
  - 9 non-English queries (fr, es, de, bn×2, hi, ja, it, pt) against English docs: BM25 **0.11**, EmbeddingGemma 2 **1.00**.
  - 24 fact-level questions over 159 single sentences: BM25 0.83, EmbeddingGemma 2 **0.96** (MRR 0.98). The one miss was "how many pieces each player starts with": it picked Chess sentence 0 over sentence 1.
- Task prefixes: dropping them still got 35/35 here. The average gap between the right answer and the best wrong answer shrank from 0.117 to 0.106.

### 3. Matryoshka truncation (same embeddings, keep the first *d* dims, re-normalise)
| Dims | Bytes/vector (fp32) | 1M vectors (fp32 / bf16) | EN top-1 | Fact top-1 | Avg right-vs-wrong gap (fact) | Brute-force top-10 over 200k vectors |
|---|---|---|---|---|---|---|
| 768 | 3,072 | 3.07 GB / 1.54 GB | 1.00 | 0.96 | 0.077 | 10.0 ms |
| 512 | 2,048 | 2.05 GB / 1.02 GB | 1.00 | 0.96 | 0.076 | 7.2 ms |
| 256 | 1,024 | 1.02 GB / 0.51 GB | 1.00 | 0.96 | 0.065 | 3.7 ms |
| 128 | 512 | 0.51 GB / 0.26 GB | 1.00 | 0.96 | 0.048 | 1.1 ms |

On a corpus this small, even 128d keeps every top-1 answer. **The gap between right and wrong shrinks, though: −38% at 128d.** Google's own table shows real drops at scale, especially for images, video and audio at 128d.
Truncation does **not** speed up embedding. A query took 100 ms with `truncate_dim=128` vs 97 ms at 768. It only saves storage and search time.
(The 200k-vector timing uses random unit vectors purely to time the matrix maths.)

### 4. Multimodal (full 740M config, CPU fp32)
- **Image embedding time per photo** by vision token budget: 70 tokens → **0.69 s**, 280 (default) → **3.4 s**, 560 → **7.4 s**.
  Text→photo top-1 accuracy was 10/10 at all three budgets.
- **Text → photo** (10 Wikipedia lead photos): 10/10. Example: "an old black bike parked on a street" → Bicycle.jpg 0.782 (next best 0.511).
- **Photo → which of 35 articles?** 10/10. Example: Red_panda.jpg → "Red panda" 0.814 (2nd: Bengal tiger 0.640).
- **Google's sample photo** GoldenGate.png: "San Francisco" 0.641 > "a cute cat walking" 0.535 > "a friendly dog…" 0.509.
- **Speech** (Google's 3.1 s `roses-are.wav`, 0.95 s to embed): "Roses are red, violets are blue." **0.876**, "a person speaking English" 0.679, "white noise" 0.651, "a dog barking" 0.650. It matches *what was said* without any speech-to-text step.
- **Video** (Google's 29.7 s `cat_dozing.mp4`): "a cat dozing" **0.688** > "a cat playing with a small lizard" 0.603 > "a red panda on a branch" 0.545 > "a sports car…" 0.440.
  CPU cost depends on frames: max_frames 4 → 10 s, 8 → 15 s, 32 (default) → 47–55 s. The ranking stayed the same.
- **One search box** (35 texts + 10 photos + 1 speech clip + 1 video):
  - "a sleepy cat" → **[video] cat_dozing.mp4 0.700**, then [photo] Red_panda 0.651.
  - "a lighthouse in the sea" → **[photo] Lighthouse 0.775**, then [text] Lighthouse 0.764.
  - "someone reciting a short poem" → [text] Tagore 0.625, then **[audio] roses-are.wav 0.607**. Text matches often score a little higher than media matches, which is a mixed-index quirk to know about.
- Peak RSS for the full multimodal run: 5.7 GB (fp32).

### 5. Precision (text-only, 35 docs)
| dtype | NaN rows | Cosine vs fp32 (mean / min) | 35 docs | Query |
|---|---|---|---|---|
| float32 | 0 | 1 / 1 | 9.0 s | 104 ms |
| bfloat16 | 0 | 0.99958 / 0.99849 | 5.5 s | 74 ms |
| float16 | 0 | 0.999999 / 0.999998 | 6.3 s | 85 ms |

We did **not** reproduce the float16 NaN problem on these short CPU inputs. Google's model card still says never use float16, because activations can overflow and fail silently. Follow the card: use bf16 or fp32.

## Not done or caveats
- Audio search over a *collection* wasn't built. The only clip is Google's single 3 s sample; we avoided pulling random audio with unclear licences.
- No GPU numbers. No quantized runs (GGUF, Ollama, LiteRT) on this box.
- The tiny corpus saturates accuracy, so the MRL quality differences show up as margins, not misses.
- To publish as a public repo: keep `fetch_data.py` downloading the media at run time (see `.gitignore`), and credit Wikipedia (CC BY-SA 4.0) and the photo authors listed in `data/images.json`.
