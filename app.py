# app.py
import os
import re
import time
import collections
from flask import Flask, request, jsonify
from flask_cors import CORS
import feedparser
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# --- small utilities ---
def text_first_sentence(text, max_chars=300):
    if not text:
        return ""
    txt = re.sub(r'\s+', ' ', text).strip()
    # split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', txt)
    first = parts[0] if parts else txt
    return first[:max_chars].strip()

def extract_text_from_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # prefer article/main
    article = soup.find("article") or soup.find("main")
    if article:
        ps = article.find_all("p")
    else:
        ps = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in ps)
    # fallback to raw text if no p tags
    if not text:
        text = soup.get_text(separator=" ", strip=True)
    return text

def safe_fetch(url, timeout=10):
    try:
        headers = {"User-Agent": "rss-ai-backend/1.0 (+https://example.com)"}
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

# --- endpoints ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "time": int(time.time())})

@app.route("/summarize", methods=["POST"])
def summarize():
    body = request.get_json(silent=True) or {}
    feed_urls = body.get("feedUrls") or []
    max_articles = int(body.get("maxArticles") or 25)

    summaries = []
    seen_urls = set()

    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
            entries = parsed.entries or []
            for e in entries:
                link = e.get("link") or e.get("id") or ""
                if not link or link in seen_urls:
                    continue
                seen_urls.add(link)

                title = (e.get("title") or "").strip()
                # prefer content -> summary -> description
                content = ""
                if "content" in e and e.content:
                    content = " ".join([c.value for c in e.content if c.value])
                content = content or e.get("summary") or e.get("description") or ""
                # extract text and make a lightweight summary
                text = extract_text_from_html(content)
                if not text and link:
                    page_html = safe_fetch(link)
                    text = extract_text_from_html(page_html) if page_html else ""

                short_summary = text_first_sentence(text or title, max_chars=300)
                summaries.append({
                    "title": title or link,
                    "url": link,
                    "summary": short_summary
                })
                if len(summaries) >= max_articles:
                    break
        except Exception:
            # skip bad feed
            continue
        if len(summaries) >= max_articles:
            break

    # Build a simple digest: top frequent words across summaries (excluding stopwords)
    combined = " ".join(s["summary"] for s in summaries)
    words = re.findall(r'\b[a-z]{4,}\b', combined.lower())
    stop = {
        "about","which","their","there","these","those","after","before","within",
        "between","should","could","would","please","first","second","third"
    }
    counter = collections.Counter(w for w in words if w not in stop)
    top = [w for w,c in counter.most_common(8)]
    digest = "Top topics: " + ", ".join(top[:5]) if top else "No digest available"

    return jsonify({"digest": digest, "summaries": summaries})

@app.route("/summarize_article", methods=["POST"])
def summarize_article():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"summary": ""})

    page_html = safe_fetch(url, timeout=12)
    if not page_html:
        return jsonify({"summary": ""})

    text = extract_text_from_html(page_html)
    summary = text_first_sentence(text, max_chars=500)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
