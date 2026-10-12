"""Demo 0 - what each modular configuration costs on this CPU box.

Each config is loaded in a fresh Python process so memory numbers don't leak
between runs. Reports params, load time, RSS increase, and one text query latency.
"""
import json, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).parent
CHILD = r'''
import json, sys
from common import load, timed, ms, rss_mb, peak_rss_mb
m, info = load(sys.argv[1])
m.encode("warm up", prompt_name="SearchQuery")
_, t = timed(lambda: m.encode("why does the night sky glow green near the poles", prompt_name="SearchQuery"), repeats=15)
info.update({"rss_delta_mb": info["rss_after_load_mb"] - info["rss_before_load_mb"], "text_query": ms(t),
             "peak_rss_mb": round(peak_rss_mb())})
print("JSON" + json.dumps(info))
'''
rows = []
for cfg in ["text", "text+image", "text+audio", "full"]:
    out = subprocess.run([sys.executable, "-c", CHILD, cfg], cwd=HERE, capture_output=True, text=True,
                         env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    line = [l for l in out.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print(cfg, "FAILED", out.stderr[-2000:]); continue
    r = json.loads(line[0][4:]); rows.append(r)
    print(f"{cfg:11s} params={r['params']/1e6:6.1f}M  fp32 weights≈{r['param_mb_fp32']:5d} MB  load={r['load_seconds']:5.2f}s  "
          f"RSS +{r['rss_delta_mb']:5d} MB (peak {r['peak_rss_mb']} MB)  text query median {r['text_query']['median_ms']} ms")
(HERE / "results" / "configs.json").write_text(json.dumps(rows, indent=1))
print("[saved results/configs.json]")
