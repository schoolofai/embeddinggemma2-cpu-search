"""Shared helpers: offline, token-free model loading + timing/memory probes."""
import os, json, time, resource, pathlib, statistics

HERE = pathlib.Path(__file__).parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))   # weights live inside the demo folder
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # never send a token
os.environ.pop("HF_TOKEN", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil, torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL_ID = "google/embeddinggemma-2"
CONFIGS = {  # straight from the model card's "Selective Encoder Loading" table
    "text":       {"vision_config": None, "audio_config": None},  # 270M
    "text+image": {"audio_config": None},                          # 440M
    "text+audio": {"vision_config": None},                         # 570M
    "full":       {},                                              # 740M
}

def rss_mb():
    return psutil.Process().memory_info().rss / 1e6

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # Linux: KiB -> MB

def load(config="text"):
    """Load EmbeddingGemma 2 on CPU in float32 (model card: use fp32 on most CPUs, never fp16)."""
    before = rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, device="cpu", config_kwargs=CONFIGS[config],
                                model_kwargs={"torch_dtype": torch.float32})
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    info = {"config": config, "params": params, "param_mb_fp32": round(params * 4 / 1e6),
            "load_seconds": round(load_s, 2), "rss_before_load_mb": round(before),
            "rss_after_load_mb": round(rss_mb()), "threads": torch.get_num_threads(),
            "torch": torch.__version__}
    return model, info

def timed(fn, repeats=1):
    """Run fn() `repeats` times; return (last result, list of seconds)."""
    out, ts = None, []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, ts

def ms(ts):
    return {"median_ms": round(statistics.median(ts) * 1000, 1), "min_ms": round(min(ts) * 1000, 1),
            "max_ms": round(max(ts) * 1000, 1), "n": len(ts)}

def save(name, obj):
    p = HERE / "results" / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))
    print(f"\n[saved {p.relative_to(HERE)}]")

def corpus():
    return json.loads((HERE / "data" / "corpus.json").read_text())
