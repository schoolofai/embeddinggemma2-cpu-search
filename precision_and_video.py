"""Demo 3 - two practical knobs: numeric precision (fp32 / bf16 / fp16) and video frame budget.

Model card: run in bfloat16 or float32, never float16 (NaN / silently degraded embeddings).
"""
import subprocess, sys, json, pathlib, os
HERE = pathlib.Path(__file__).parent
CHILD = r'''
import sys, json, numpy as np, torch
from sentence_transformers import SentenceTransformer
from common import MODEL_ID, CONFIGS, timed, ms, rss_mb, corpus
dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
before = rss_mb()
m = SentenceTransformer(MODEL_ID, device="cpu", config_kwargs=CONFIGS["text"], model_kwargs={"torch_dtype": dt})
after = rss_mb()
docs = corpus()
texts = [f"title: {d['title']} | text: {d['text']}" for d in docs]
m.encode("warm up", prompt_name="SearchQuery")
E, t = timed(lambda: m.encode(texts, batch_size=8, convert_to_numpy=True), repeats=1)
E = np.asarray(E, dtype=np.float32)
_, tq = timed(lambda: m.encode("why does the night sky glow green near the poles", prompt_name="SearchQuery"), repeats=10)
np.save(f"/tmp/eg2_{sys.argv[1]}.npy", E)
print("JSON" + json.dumps({"dtype": sys.argv[1], "rss_delta_mb": round(after - before), "nan_rows": int(np.isnan(E).any(1).sum()),
      "index_35_docs_s": round(t[0], 2), "query": ms(tq)}))
'''
env = {**os.environ, "HF_HUB_OFFLINE": "1"}
rows = []
for dt in ["float32", "bfloat16", "float16"]:
    o = subprocess.run([sys.executable, "-c", CHILD, dt], cwd=HERE, capture_output=True, text=True, env=env)
    line = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        rows.append({"dtype": dt, "error": o.stderr.strip().splitlines()[-1][:300]}); print(dt, "ERROR", rows[-1]["error"]); continue
    rows.append(json.loads(line[0][4:]))
import numpy as np
ref = np.load("/tmp/eg2_float32.npy"); ref /= np.linalg.norm(ref, axis=1, keepdims=True)
for r in rows:
    p = pathlib.Path(f"/tmp/eg2_{r['dtype']}.npy")
    if "error" in r or not p.exists(): continue
    E = np.load(p); n = np.linalg.norm(E, axis=1, keepdims=True); E = E / np.where(n == 0, 1, n)
    cos = np.nansum(E * ref, 1)
    r["cosine_vs_fp32_mean"] = round(float(np.nanmean(cos)), 5); r["cosine_vs_fp32_min"] = round(float(np.nanmin(cos)), 5)
    print(f"{r['dtype']:9s} RSS +{r['rss_delta_mb']} MB | NaN rows {r['nan_rows']}/35 | 35 docs {r['index_35_docs_s']}s | "
          f"query median {r['query']['median_ms']} ms | cosine vs fp32 mean {r['cosine_vs_fp32_mean']} min {r['cosine_vs_fp32_min']}")

VID = r'''
import sys, json
from common import load, timed, ms
m, info = load("text+image")
cands = ["a cat dozing", "a cat playing with a small lizard", "a sports car racing on a highway"]
C = m.encode(cands, normalize_embeddings=True)
out = {}
for mf in [4, 8, 32]:
    V, t = timed(lambda: m.encode({"video": "data/media/cat_dozing.mp4"}, normalize_embeddings=True,
                                   processing_kwargs={"video": {"max_frames": mf}}))
    out[mf] = {"seconds": round(t[0], 1), "scores": dict(zip(cands, [round(float(x), 3) for x in (V @ C.T)]))}
print("JSON" + json.dumps(out))
'''
o = subprocess.run([sys.executable, "-c", VID], cwd=HERE, capture_output=True, text=True, env=env)
line = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
vid = json.loads(line[0][4:]) if line else {"error": o.stderr[-1500:]}
print("\nVideo (29.7 s clip, text+image config) by max_frames:")
for k, v in vid.items(): print(f"  max_frames={k}: {v}")
(HERE / "results" / "precision_and_video.json").write_text(json.dumps({"precision": rows, "video_frames": vid}, indent=1))
print("[saved results/precision_and_video.json]")
