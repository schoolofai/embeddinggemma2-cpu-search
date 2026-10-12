"""Demo 1 - local semantic search over 35 real Wikipedia intros (text-only, 270M config, CPU).

Shows: query -> top results with scores, per-embedding timing, memory,
a keyword (BM25) baseline for contrast, cross-lingual queries, task-prefix
ablation, and Matryoshka truncation (768/512/256/128) quality + speed + storage.
"""
import math, re, time, collections
import numpy as np
from common import load, timed, ms, save, corpus, rss_mb, peak_rss_mb

# (query, gold article title). Written to avoid re-using the article's own title words.
QUERIES_EN = [
    ("why does the night sky glow green near the poles", "Aurora"),
    ("how do plants turn sunlight into food", "Photosynthesis"),
    ("a place where gravity is so strong not even light escapes", "Black hole"),
    ("mountain that spews molten rock and ash", "Volcano"),
    ("insect that makes honey and lives in hives", "Western honey bee"),
    ("small reddish tree-dwelling mammal from the Himalayas", "Red panda"),
    ("Japanese noodle soup in a rich broth", "Ramen"),
    ("two-wheeled vehicle you pedal", "Bicycle"),
    ("tower with a bright lamp that warns ships", "Lighthouse"),
    ("wrought-iron landmark in Paris", "Eiffel Tower"),
    ("biggest coral system off the coast of Queensland", "Great Barrier Reef"),
    ("colourful arc in the sky after rain", "Rainbow"),
    ("drink brewed from roasted beans", "Coffee"),
    ("improvised music that grew up in New Orleans", "Jazz"),
    ("board game where you try to checkmate the king", "Chess"),
    ("Japanese craft of making shapes from a single sheet", "Origami"),
    ("a jab that trains your immune system", "Vaccine"),
    ("antibiotic Fleming found in mould", "Penicillin"),
    ("when the moon blocks out the sun", "Solar eclipse"),
    ("slow-moving river of ice", "Glacier"),
    ("striped big cat of India and Bangladesh", "Bengal tiger"),
    ("Nobel prize winning poet from Bengal", "Rabindranath Tagore"),
    ("autumn festival worshipping the goddess in Kolkata", "Durga Puja"),
    ("bat-and-ball sport with wickets and overs", "Cricket"),
    ("bread raised with a wild yeast starter", "Sourdough"),
    ("instrument that needs an amplifier to be heard", "Electric guitar"),
    ("flying in a basket under a bag of heated air", "Hot air balloon"),
    ("highest peak on Earth", "Mount Everest"),
    ("eight-armed sea creature that changes colour", "Octopus"),
    ("turning kitchen scraps into fertiliser", "Compost"),
    ("toy you fly on a string in the wind", "Kite"),
    ("boat that travels underwater", "Submarine"),
    ("grain that is the staple food of billions", "Rice"),
    ("orange suspension bridge in San Francisco", "Golden Gate Bridge"),
    ("dried leaves steeped in hot water", "Tea"),
]
QUERIES_X = [  # cross-lingual: non-English query, English documents
    ("fr", "pourquoi le ciel nocturne brille-t-il en vert près des pôles", "Aurora"),
    ("es", "el animal marino de ocho brazos que cambia de color", "Octopus"),
    ("de", "der höchste Berg der Welt", "Mount Everest"),
    ("bn", "বাঙালি কবি যিনি নোবেল পুরস্কার পেয়েছিলেন", "Rabindranath Tagore"),
    ("bn", "কলকাতার শরৎকালের সবচেয়ে বড় উৎসব", "Durga Puja"),
    ("hi", "पानी के नीचे चलने वाली नाव", "Submarine"),
    ("ja", "醤油味のスープに入った麺料理", "Ramen"),
    ("it", "bevanda preparata con chicchi tostati", "Coffee"),
    ("pt", "brinquedo que voa preso a uma linha", "Kite"),
]
SHOWCASE = ["why does the night sky glow green near the poles", "instrument that needs an amplifier to be heard",
            "a jab that trains your immune system", "বাঙালি কবি যিনি নোবেল পুরস্কার পেয়েছিলেন"]
DIMS = [768, 512, 256, 128]

# Harder, fact-level test: the corpus split into ~159 single sentences; each query targets ONE sentence.
def sentences(docs):
    out = []
    for d in docs:
        ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", d["text"].replace("\n", " ")) if len(x.strip()) > 30]
        out += [(d["title"], i, x) for i, x in enumerate(ss)]
    return out
FACT_QUERIES = [
    ("what gas does oxygenic photosynthesis give off", "Photosynthesis", 1),
    ("what is the point of no return around one called", "Black hole", 2),
    ("where are most volcanoes on Earth found", "Volcano", 1),
    ("what does the honey bee's Latin species name mean", "Western honey bee", 1),
    ("how heavy is a red panda", "Red panda", 2),
    ("why does tea make you feel alert", "Tea", 1),
    ("regional styles like the pork-bone broth from Kyushu", "Ramen", 3),
    ("who built the Paris iron tower and when", "Eiffel Tower", 1),
    ("can the reef be seen from space", "Great Barrier Reef", 3),
    ("which part of the sky a rainbow shows up in", "Rainbow", 2),
    ("how many pieces each player starts with", "Chess", 1),
    ("first non-European to win a Nobel prize", "Rabindranath Tagore", 1),
    ("poet who wrote two countries' national anthems", "Rabindranath Tagore", 2),
    ("UNESCO heritage listing for the Kolkata festival", "Durga Puja", 2),
    ("what gives the loaf its tangy taste", "Sourdough", 2),
    ("how pickups turn string vibrations into a signal", "Electric guitar", 1),
    ("what fuel heats the air for the basket ride", "Hot air balloon", 1),
    ("official height of the tallest mountain measured in 2020", "Mount Everest", 2),
    ("how clever are octopuses", "Octopus", 6),
    ("where the flying toy's name comes from", "Kite", 5),
    ("why underwater vessels are called boats not ships", "Submarine", 2),
    ("how much of the grain is traded between countries", "Rice", 4),
    ("which colours excited oxygen and nitrogen give off", "Aurora", 1),
    ("how ice rivers carve fjords and moraines", "Glacier", 3),
]

# ---------- tiny BM25 keyword baseline (what "Ctrl+F with extra steps" gets you) ----------
def tok(s): return re.findall(r"\w+", s.lower())
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]; self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        df = collections.Counter(w for d in self.docs for w in set(d)); N = len(self.docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]
    def scores(self, q):
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for w in tok(q):
                if w in tf:
                    f = tf[w]; s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return np.array(out)

def rank_metrics(S, gold_idx):
    """S: (n_queries, n_docs) scores. Returns hit@1, hit@3, MRR, mean margin (gold - best other)."""
    hits1 = hits3 = rr = margin = 0.0
    for s, g in zip(S, gold_idx):
        order = np.argsort(-s); r = int(np.where(order == g)[0][0]) + 1
        hits1 += r == 1; hits3 += r <= 3; rr += 1 / r
        margin += s[g] - np.max(np.delete(s, g))
    n = len(gold_idx)
    return {"hit@1": round(hits1 / n, 3), "hit@3": round(hits3 / n, 3), "mrr": round(rr / n, 3), "mean_margin": round(margin / n, 4)}

def trunc(E, d):
    E = E[:, :d]; return E / np.linalg.norm(E, axis=1, keepdims=True)

def main():
    docs = corpus(); titles = [d["title"] for d in docs]
    doc_inputs = [f"title: {d['title']} | text: {d['text']}" for d in docs]  # model card: titled doc format
    model, info = load("text")
    print(f"Loaded {info['params']/1e6:.0f}M params (text-only config) in {info['load_seconds']}s; "
          f"RSS {info['rss_before_load_mb']} -> {info['rss_after_load_mb']} MB")

    # warm-up (first call pays one-off costs)
    model.encode("warm up", prompt_name="SearchQuery")

    # ---- index the corpus ----
    D, t_idx = timed(lambda: model.encode(doc_inputs, batch_size=8, normalize_embeddings=True), repeats=1)
    n_tok = sum(len(model.tokenizer(x)["input_ids"]) for x in doc_inputs)
    _, t_one_doc = timed(lambda: model.encode(doc_inputs[0], normalize_embeddings=True), repeats=10)
    print(f"Indexed {len(docs)} docs ({n_tok} tokens) in {t_idx[0]:.2f}s "
          f"= {t_idx[0]/len(docs)*1000:.0f} ms/doc batched; single doc median {ms(t_one_doc)['median_ms']} ms")

    # ---- queries ----
    qs = [q for q, _ in QUERIES_EN]; gold = [titles.index(g) for _, g in QUERIES_EN]
    qx = [q for _, q, _ in QUERIES_X]; goldx = [titles.index(g) for _, _, g in QUERIES_X]
    Q, _ = timed(lambda: model.encode(qs, prompt_name="SearchQuery", normalize_embeddings=True))
    QX = model.encode(qx, prompt_name="SearchQuery", normalize_embeddings=True)
    _, t_q = timed(lambda: model.encode(qs[0], prompt_name="SearchQuery", normalize_embeddings=True), repeats=20)
    print(f"Single query embedding: median {ms(t_q)['median_ms']} ms (n=20)")

    # ---- showcase ----
    all_q = qs + qx; all_Q = np.vstack([Q, QX])
    showcase = []
    print("\n=== Query -> top 3 (cosine) ===")
    for q in SHOWCASE:
        s = all_Q[all_q.index(q)] @ D.T; top = np.argsort(-s)[:3]
        print(f"\nQ: {q}")
        for i in top: print(f"   {s[i]:.3f}  {titles[i]}")
        showcase.append({"query": q, "top3": [(titles[i], round(float(s[i]), 3)) for i in top]})

    # ---- keyword baseline vs embeddings ----
    bm = BM25([f"{d['title']} {d['text']}" for d in docs])
    B = np.vstack([bm.scores(q) for q in qs]); BX = np.vstack([bm.scores(q) for q in qx])
    res = {"english": {"bm25": rank_metrics(B, gold), "embeddinggemma2_768": rank_metrics(Q @ D.T, gold)},
           "cross_lingual": {"bm25": rank_metrics(BX, goldx), "embeddinggemma2_768": rank_metrics(QX @ D.T, goldx)}}
    misses = [(q, titles[g], titles[int(np.argmax(s))]) for q, g, s in zip(all_q, gold + goldx, all_Q @ D.T) if int(np.argmax(s)) != g]
    print("\n=== Keyword (BM25) vs EmbeddingGemma 2 ===")
    for k, v in res.items(): print(f"{k:14s} BM25 hit@1={v['bm25']['hit@1']:.2f}   EG2 hit@1={v['embeddinggemma2_768']['hit@1']:.2f}  (MRR {v['embeddinggemma2_768']['mrr']:.2f})")
    print("EG2 top-1 misses:", misses or "none")

    # ---- task prefix ablation ----
    Qn = model.encode(qs, normalize_embeddings=True)  # no prefix on queries
    Dn = model.encode([d["text"] for d in docs], normalize_embeddings=True)  # no title/prefix on docs
    res["prefix_ablation"] = {"with_prefixes": rank_metrics(Q @ D.T, gold), "no_prefixes": rank_metrics(Qn @ Dn.T, gold)}
    print("\n=== Task prefixes ===", res["prefix_ablation"])

    # ---- fact-level index (sentences) for a harder MRL test ----
    sents = sentences(docs)
    S_in = [f"title: {t} | text: {x}" for t, _, x in sents]
    SE, t_sidx = timed(lambda: model.encode(S_in, batch_size=16, normalize_embeddings=True))
    keys = [(t, i) for t, i, _ in sents]
    fq = [q for q, _, _ in FACT_QUERIES]; fgold = [keys.index((t, i)) for _, t, i in FACT_QUERIES]
    FQ = model.encode(fq, prompt_name="QuestionAnswering", normalize_embeddings=True)
    print(f"\nFact-level index: {len(sents)} sentences embedded in {t_sidx[0]:.1f}s ({t_sidx[0]/len(sents)*1000:.0f} ms/sentence)")
    fmiss = [(q, sents[g][2][:70], sents[int(np.argmax(s))][2][:70]) for q, g, s in zip(fq, fgold, FQ @ SE.T) if int(np.argmax(s)) != g]
    res["fact_level"] = {"n_sentences": len(sents), "bm25": rank_metrics(np.vstack([BM25([x for x in S_in]).scores(q) for q in fq]), fgold),
                         "embeddinggemma2_768": rank_metrics(FQ @ SE.T, fgold), "misses_768": fmiss}
    print("fact-level BM25", res["fact_level"]["bm25"], "\nfact-level EG2 ", res["fact_level"]["embeddinggemma2_768"])
    for m in fmiss: print("  miss:", m)

    # ---- Matryoshka truncation ----
    print("\n=== Matryoshka truncation (same model, keep first d dims, re-normalise) ===")
    mrl = {}
    _, t_q128 = timed(lambda: model.encode(qs[0], prompt_name="SearchQuery", truncate_dim=128, normalize_embeddings=True), repeats=20)
    rng = np.random.default_rng(0); N = 200_000
    big = rng.standard_normal((N, 768), dtype=np.float32); big /= np.linalg.norm(big, axis=1, keepdims=True)
    for d in DIMS:
        Qd, Dd, QXd = trunc(Q, d), trunc(D, d), trunc(QX, d)
        bigd = np.ascontiguousarray(big[:, :d]); qd = Qd[0]
        _, t_s = timed(lambda: np.argpartition(-(bigd @ qd), 10)[:10], repeats=20)
        agree = float(np.mean(np.argmax(Qd @ Dd.T, 1) == np.argmax(Q @ D.T, 1)))
        mrl[d] = {"english": rank_metrics(Qd @ Dd.T, gold), "cross_lingual": rank_metrics(QXd @ Dd.T, goldx),
                  "fact_level": rank_metrics(trunc(FQ, d) @ trunc(SE, d).T, fgold),
                  "top1_agrees_with_768": round(agree, 3),
                  "bytes_per_vector_fp32": d * 4, "storage_1M_vectors_fp32_mb": d * 4, "storage_1M_vectors_bf16_mb": d * 2,
                  "brute_force_top10_over_200k_vectors": ms(t_s)}
        print(f"{d:4d}d  EN hit@1={mrl[d]['english']['hit@1']:.2f} MRR={mrl[d]['english']['mrr']:.3f} margin={mrl[d]['english']['mean_margin']:.3f} | "
              f"X-lingual hit@1={mrl[d]['cross_lingual']['hit@1']:.2f} | FACT hit@1={mrl[d]['fact_level']['hit@1']:.2f} MRR={mrl[d]['fact_level']['mrr']:.3f} | {d*4} B/vec | "
              f"search 200k vecs {ms(t_s)['median_ms']} ms")
    print(f"Query embed time with truncate_dim=128: median {ms(t_q128)['median_ms']} ms (vs {ms(t_q)['median_ms']} ms at 768) "
          "-> truncation saves storage + search time, not embedding time")

    out = {"model_info": info, "n_docs": len(docs), "doc_tokens": n_tok,
           "index_seconds": round(t_idx[0], 2), "ms_per_doc_batched": round(t_idx[0] / len(docs) * 1000, 1),
           "single_doc": ms(t_one_doc), "single_query": ms(t_q), "single_query_trunc128": ms(t_q128),
           "showcase": showcase, "retrieval": res, "misses_768": misses, "matryoshka": mrl,
           "rss_end_mb": round(rss_mb()), "peak_rss_mb": round(peak_rss_mb()),
           "note": "200k-vector search uses random unit vectors purely to time brute-force dot products; quality numbers use the real corpus."}
    print(f"\nRSS end {out['rss_end_mb']} MB, peak {out['peak_rss_mb']} MB (incl. the 200k x 768 timing matrix ≈ 614 MB)")
    save("text_search", out)

if __name__ == "__main__":
    main()
