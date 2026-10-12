"""Demo 2 - one vector space for text, photos, speech and video (full 740M config, CPU).

Uses 10 real Wikipedia lead photos + 35 Wikipedia intros, plus Google's own
sample speech clip and video from the official EmbeddingGemma 2 notebook.
"""
import json, time
import numpy as np
from PIL import Image
from common import HERE, load, timed, ms, save, corpus, rss_mb, peak_rss_mb

IMG = HERE / "data" / "images"; MEDIA = HERE / "data" / "media"
IMAGE_QUERIES = [  # describe the picture, not the article
    ("green light curtains in a winter night sky", "Aurora.jpg"),
    ("a big striped cat walking through dry grass", "Bengal_tiger.jpg"),
    ("an old black bike parked on a street", "Bicycle.jpg"),
    ("a red suspension bridge over the water", "Golden_Gate_Bridge.jpg"),
    ("a red balloon floating in a clear blue sky", "Hot_air_balloon.jpg"),
    ("a red and white striped tower standing in the sea", "Lighthouse.jpg"),
    ("a sea creature with tentacles on the seabed", "Octopus.jpg"),
    ("a bowl of noodle soup with sliced pork", "Ramen.jpg"),
    ("a fluffy reddish animal resting on a branch", "Red_panda.jpg"),
    ("a snowy mountain with smoke rising from the top", "Volcano.jpg"),
]
DIMS = [768, 512, 256, 128]

def trunc(E, d):
    E = E[:, :d]; return E / np.linalg.norm(E, axis=1, keepdims=True)

def hit1(S, gold):
    return round(float(np.mean(np.argmax(S, 1) == np.array(gold))), 3)

def main():
    model, info = load("full")
    print(f"Loaded {info['params']/1e6:.0f}M params (full multimodal) in {info['load_seconds']}s; RSS {info['rss_before_load_mb']} -> {info['rss_after_load_mb']} MB")
    docs = corpus(); titles = [d["title"] for d in docs]
    D = model.encode([f"title: {d['title']} | text: {d['text']}" for d in docs], batch_size=8, normalize_embeddings=True)
    files = sorted(p.name for p in IMG.glob("*.jpg"))
    pil = {f: Image.open(IMG / f).convert("RGB") for f in files}
    out = {"model_info": info}

    # ---- images: timing at three vision token budgets ----
    model.encode({"image": pil[files[0]]})  # warm-up
    per_budget = {}
    for budget in [70, 280, 560]:
        kw = {"processing_kwargs": {"image": {"max_soft_tokens": budget}}}
        ts, embs = [], []
        for f in files:
            e, t = timed(lambda: model.encode({"image": pil[f]}, normalize_embeddings=True, **kw)); ts += t; embs.append(e)
        per_budget[budget] = (np.vstack(embs), ms(ts))
        print(f"image embed @ {budget:4d} vision tokens: median {ms(ts)['median_ms']} ms/image (n={len(ts)})")
    I = per_budget[280][0]  # 280 = model-card default

    # ---- text -> image ----
    qs = [q for q, _ in IMAGE_QUERIES]; gold = [files.index(f) for _, f in IMAGE_QUERIES]
    Q = model.encode(qs, prompt_name="SearchQuery", normalize_embeddings=True)
    print("\n=== Text -> image (10 photos) ===")
    for q, s in list(zip(qs, Q @ I.T))[:4]:
        top = np.argsort(-s)[:2]; print(f"Q: {q}\n   " + "   ".join(f"{s[i]:.3f} {files[i]}" for i in top))
    t2i = {b: hit1(Q @ E.T, gold) for b, (E, _) in per_budget.items()}
    print("text->image hit@1 by vision budget:", t2i)

    # ---- image -> Wikipedia article (cross-modal, 35 candidates) ----
    art_gold = [titles.index(json.loads((HERE / "data" / "images.json").read_text())[[x["file"] for x in json.loads((HERE / "data" / "images.json").read_text())].index(f)]["article"]) for f in files]
    S_ia = I @ D.T
    i2a = hit1(S_ia, art_gold)
    print("\n=== Photo -> which of 35 Wikipedia articles? ===")
    for f, s, g in zip(files, S_ia, art_gold):
        top = np.argsort(-s)[:2]; print(f"{f:24s} -> {titles[top[0]]:20s} {s[top[0]]:.3f}  (2nd {titles[top[1]]} {s[top[1]]:.3f}){'' if top[0]==g else '  MISS'}")
    print("image->article hit@1:", i2a)

    # ---- Google's own sample media (from the official notebook) ----
    gg = model.encode({"image": str(MEDIA / "GoldenGate.png")}, normalize_embeddings=True)
    cands = ["San Francisco", "a cute cat walking", "a friendly dog sitting in the grass"]
    gg_s = (gg @ model.encode(cands, normalize_embeddings=True).T).tolist()
    print("\nGoogle sample GoldenGate.png vs", dict(zip(cands, np.round(gg_s, 3))))

    model.encode({"audio": str(MEDIA / "roses-are.wav")})  # warm-up
    A, t_a = timed(lambda: model.encode({"audio": str(MEDIA / "roses-are.wav")}, normalize_embeddings=True), repeats=3)
    a_c = ["a person speaking English", "random static white noise", "Roses are red, violets are blue.", "a dog barking", "a jazz saxophone solo"]
    a_s = (A @ model.encode(a_c, normalize_embeddings=True).T).tolist()
    import soundfile as sf; info_wav = sf.info(str(MEDIA / "roses-are.wav"))
    print(f"\nAudio roses-are.wav ({info_wav.duration:.1f}s) embed median {ms(t_a)['median_ms']} ms:")
    for c, s in sorted(zip(a_c, a_s), key=lambda x: -x[1]): print(f"   {s:.3f}  {c}")

    V, t_v = timed(lambda: model.encode({"video": str(MEDIA / "cat_dozing.mp4")}, normalize_embeddings=True), repeats=1)
    v_c = ["a cat dozing", "a cat playing with a small lizard", "a sports car racing on a highway", "a red panda on a branch"]
    v_s = (V @ model.encode(v_c, normalize_embeddings=True).T).tolist()
    print(f"\nVideo cat_dozing.mp4 embed median {ms(t_v)['median_ms']} ms:")
    for c, s in sorted(zip(v_c, v_s), key=lambda x: -x[1]): print(f"   {s:.3f}  {c}")

    # ---- one search box over everything ----
    items = [("text", t) for t in titles] + [("photo", f) for f in files] + [("audio", "roses-are.wav (speech)"), ("video", "cat_dozing.mp4")]
    ALL = np.vstack([D, I, A, V])
    mixed_q = ["a sleepy cat", "someone reciting a short poem", "red panda", "the tallest mountain on Earth", "a lighthouse in the sea"]
    MQ = model.encode(mixed_q, prompt_name="SearchQuery", normalize_embeddings=True)
    print("\n=== One search box: 35 texts + 10 photos + 1 speech clip + 1 video ===")
    mixed = []
    for q, s in zip(mixed_q, MQ @ ALL.T):
        top = np.argsort(-s)[:4]
        print(f"Q: {q}\n   " + " | ".join(f"{s[i]:.3f} [{items[i][0]}] {items[i][1]}" for i in top))
        mixed.append({"query": q, "top4": [(items[i][0], items[i][1], round(float(s[i]), 3)) for i in top]})

    # ---- Matryoshka on the cross-modal tasks ----
    mrl = {d: {"text_to_image_hit@1": hit1(trunc(Q, d) @ trunc(I, d).T, gold), "image_to_article_hit@1": hit1(trunc(I, d) @ trunc(D, d).T, art_gold)} for d in DIMS}
    print("\nMatryoshka on cross-modal tasks:", mrl)

    out.update({"image_ms_by_budget": {b: v[1] for b, v in per_budget.items()}, "text_to_image_hit@1_by_budget": t2i,
                "image_to_article_hit@1": i2a, "google_goldengate": dict(zip(cands, gg_s)),
                "audio": {"seconds": info_wav.duration, "embed": ms(t_a), "scores": dict(zip(a_c, a_s))},
                "video": {"embed": ms(t_v), "scores": dict(zip(v_c, v_s))}, "mixed_search": mixed, "matryoshka": mrl,
                "rss_end_mb": round(rss_mb()), "peak_rss_mb": round(peak_rss_mb())})
    print(f"\nRSS end {out['rss_end_mb']} MB, peak {out['peak_rss_mb']} MB")
    save("multimodal", out)

if __name__ == "__main__":
    main()
