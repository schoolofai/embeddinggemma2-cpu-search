"""Fetch a small, real, public corpus for the demo.

Text:   the intro paragraph of ~36 English Wikipedia articles (CC BY-SA 4.0),
        via the public Wikipedia REST summary API.
Images: the lead image (thumbnail) of 12 of those articles, from Wikimedia
        Commons, with licence + author metadata recorded for attribution.
Media:  Google's own sample files used in the official EmbeddingGemma 2
        multimodal notebook (Gemma cookbook): a photo, a speech clip, a video.
No sign-in, no API keys.
"""
import json, re, time, urllib.error, urllib.parse, urllib.request, pathlib, html

ROOT = pathlib.Path(__file__).parent / "data"
UA = {"User-Agent": "ProAIClips-EmbeddingGemma2-demo/1.0 (research; contact via YouTube channel)"}

TOPICS = [
    "Aurora", "Photosynthesis", "Black hole", "Volcano", "Western honey bee", "Red panda",
    "Tea", "Ramen", "Bicycle", "Lighthouse", "Eiffel Tower", "Great Barrier Reef",
    "Rainbow", "Coffee", "Jazz", "Chess", "Origami", "Vaccine", "Penicillin",
    "Solar eclipse", "Glacier", "Bengal tiger", "Rabindranath Tagore", "Durga Puja",
    "Cricket", "Sourdough", "Electric guitar", "Hot air balloon", "Mount Everest",
    "Octopus", "Compost", "Kite", "Submarine", "Rice", "Golden Gate Bridge", "Northern Lights (disambiguation)",
]
TOPICS.remove("Northern Lights (disambiguation)")  # keep list honest: 35 topics
IMAGE_TOPICS = ["Aurora", "Red panda", "Bicycle", "Lighthouse", "Ramen",
                "Bengal tiger", "Hot air balloon", "Octopus", "Electric guitar", "Volcano", "Golden Gate Bridge"]

GOOGLE_MEDIA = {  # from the official multimodal notebook (google-gemma/cookbook, Apache 2.0)
    "GoldenGate.png": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
    "kitchen_painting.jpg": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/kitchen_painting.jpg",
    "roses-are.wav": "https://ai.google.dev/gemma/docs/audio/roses-are.wav",
    "cat_dozing.mp4": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/cat_dozing.mp4",
}

def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 2 ** (i + 1)
            print(f"  429, waiting {wait}s"); time.sleep(wait)

def strip(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()

def main():
    corpus = []
    cached = ROOT / "corpus.json"
    for t in ([] if cached.exists() else TOPICS):
        d = json.loads(get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(t.replace(" ", "_"))))
        corpus.append({"id": d["titles"]["canonical"], "title": d["title"], "text": d["extract"],
                       "url": d["content_urls"]["desktop"]["page"], "licence": "CC BY-SA 4.0 (Wikipedia)",
                       "thumb": (d.get("thumbnail") or {}).get("source"),
                       "image_file": (d.get("originalimage") or {}).get("source")})
        time.sleep(1.0)
    if corpus:
        cached.write_text(json.dumps(corpus, indent=1, ensure_ascii=False))
    corpus = json.loads(cached.read_text())
    print("texts:", len(corpus))

    images = []
    for doc in corpus:
        if doc["title"] not in IMAGE_TOPICS or not doc["thumb"]:
            continue
        # ask for a 500px rendition (a Wikimedia standard thumbnail size)
        src = re.sub(r"/(\d+)px-", "/500px-", doc["thumb"])
        fname = re.sub(r"[^A-Za-z0-9]+", "_", doc["title"]).strip("_") + ".jpg"
        if not (ROOT / "images" / fname).exists():
            (ROOT / "images" / fname).write_bytes(get(src))
        # Commons file name = path segment after /thumb/x/xy/ in the thumbnail URL
        m = re.search(r"/commons/thumb/[0-9a-f]/[0-9a-f]{2}/([^/?]+)/", doc["thumb"])
        filetitle = "File:" + urllib.parse.unquote(m.group(1)) if m else "File:" + urllib.parse.unquote(doc["image_file"].split("/")[-1].split("?")[0])
        meta = json.loads(get("https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=extmetadata&format=json&titles="
                              + urllib.parse.quote(filetitle)))
        page = next(iter(meta["query"]["pages"].values()))
        em = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
        images.append({"file": fname, "article": doc["title"], "commons_page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(filetitle.replace(" ", "_")),
                       "licence": strip(em.get("LicenseShortName", {}).get("value")), "artist": strip(em.get("Artist", {}).get("value"))[:120]})
        time.sleep(1.5)
    (ROOT / "images.json").write_text(json.dumps(images, indent=1, ensure_ascii=False))
    print("images:", len(images))

    for name, url in GOOGLE_MEDIA.items():
        if not (ROOT / "media" / name).exists():
            (ROOT / "media" / name).write_bytes(get(url))
    (ROOT / "media" / "SOURCES.json").write_text(json.dumps(GOOGLE_MEDIA, indent=1))
    print("google sample media:", len(GOOGLE_MEDIA))

if __name__ == "__main__":
    main()
