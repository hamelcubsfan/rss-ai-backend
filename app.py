# app.py
import os
import re
import time
import collections
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import feedparser
import requests
from bs4 import BeautifulSoup

# Try to import Google GenAI SDK (Gemini). If unavailable, continue without LLM.
USE_GEMINI = False
gemini_client = None
try:
    from google import genai
    # Obtain API key from environment. Accept either GEMINI_API_KEY or GOOGLE_API_KEY.
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            gemini_client = genai.Client(api_key=gemini_key)
            USE_GEMINI = True
            logging.info("Gemini client initialized")
        except Exception as e:
            logging.warning("Failed to initialize Gemini client: %s", e)
            gemini_client = None
            USE_GEMINI = False
    else:
        logging.info("No Gemini API key found; running without LLM.")
except Exception as e:
    gemini_client = None
    USE_GEMINI = False
    logging.info("google-genai not installed or import failed; running without Gemini. %s", e)

app = Flask(__name__)
CORS(app)

# --- small utilities ---
def text_first_sentence(text, max_chars=300):
    if not text:
        return ""
    txt = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(?<=[.!?])\s+', txt)
    first = parts[0] if parts else txt
    return first[:max_chars].strip()

def extract_text_from_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find("main")
    if article:
        ps = article.find_all("p")
    else:
        ps = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in ps)
    if not text:
        text = soup.get_text(separator=" ", strip=True)
    return text

def safe_fetch(url, timeout=10):
    try:
        headers = {"User-Agent": "rss-ai-backend/1.0 (+https://example.com)"}
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.debug("safe_fetch failed for %s: %s", url, e)
        return None

# --- Gemini helpers (if GEMINI enabled) ---
def gemini_summarize_text_one_sentence(text, model="gemini-2.5-flash"):
    """
    Use Gemini to produce a single-sentence summary focussed on recruiting signals.
    Returns empty string on failure or if Gemini unavailable.
    """
    if not USE_GEMINI or not gemini_client or not text:
        return ""
    try:
        prompt = (
            "You are a concise news summarizer focused on recruiting signals. "
            "Given the article content below, produce exactly one sentence that highlights hiring, layoffs, leadership moves, funding, or organizational changes. "
            "Be factual and brief.\n\n"
            "Article content:\n"
            f"{text}\n\n"
            "One-sentence summary:"
        )
        resp = gemini_client.models.generate_content(model=model, contents=prompt)
        # Quicksafe: many GenAI responses expose .text
        out = getattr(resp, "text", None)
        if not out:
            # fallback: try to stringify response
            out = str(resp)
        if out:
            return out.strip().replace("\n", " ")
    except Exception as e:
        logging.warning("Gemini summarization failed: %s", e)
    return ""

def gemini_make_digest_from_summaries(one_sentence_summaries, model="gemini-2.5-flash"):
    """
    Accept list of one-line summaries and ask Gemini for a short digest:
    Trend sentence + up to five short bullets.
    """
    if not USE_GEMINI or not gemini_client:
        return ""
    try:
        joined = "\n".join(f"- {s}" for s in one_sentence_summaries if s)
        prompt = (
            "You are a concise analyst. From the bullet one-line summaries below, "
            "produce: (A) a single one-sentence trend summary, then (B) up to five short bullet points "
            "highlighting the most important recruiting signals. Keep bullets extremely short.\n\n"
            f"Summaries:\n{joined}\n\n"
            "Output format:\nTrend: <one sentence>\n- bullet1\n- bullet2\n"
        )
        resp = gemini_client.models.generate_content(model=model, contents=prompt)
        out = getattr(resp, "text", None) or str(resp)
        return out.strip().replace("\r", "")
    except Exception as e:
        logging.warning("Gemini digest failed: %s", e)
        return ""

# --- endpoints ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "time": int(time.time()), "gemini_enabled": bool(USE_GEMINI)})

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
                content = ""
                if "content" in e and e.content:
                    content = " ".join([c.value for c in e.content if c.value])
                content = content or e.get("summary") or e.get("description") or ""
                text = extract_text_from_html(content)
                if not text and link:
                    page_html = safe_fetch(link)
                    text = extract_text_from_html(page_html) if page_html else ""

                # baseline summary
                short_summary = text_first_sentence(text or title, max_chars=300)

                # attempt Gemini LLM refinement if available
                if USE_GEMINI:
                    refined = gemini_summarize_text_one_sentence(text or title)
                    if refined:
                        short_summary = refined

                summaries.append({
                    "title": title or link,
                    "url": link,
                    "summary": short_summary
                })
                if len(summaries) >= max_articles:
                    break
        except Exception as e:
            logging.debug("Feed parse error for %s: %s", feed_url, e)
            continue
        if len(summaries) >= max_articles:
            break

    # Build digest: prefer Gemini if available
    combined = " ".join(s["summary"] for s in summaries)
    # simple fallback digest (frequency)
    words = re.findall(r'\b[a-z]{4,}\b', combined.lower())
    stop = {
        "about","which","their","there","these","those","after","before","within",
        "between","should","could","would","please","first","second","third"
    }
    counter = collections.Counter(w for w,c in ((w,1) for w in words) if w not in stop)
    # keep simple fallback
    top = [w for w,c in counter.most_common(8)]
    fallback_digest = "Top topics: " + ", ".join(top[:5]) if top else "No digest available"

    digest = fallback_digest
    if USE_GEMINI:
        try:
            one_sentences = [s.get("summary","") for s in summaries]
            llm_digest = gemini_make_digest_from_summaries(one_sentences)
            if llm_digest:
                digest = llm_digest
        except Exception as e:
            logging.warning("Gemini digest generation failed: %s", e)
            digest = fallback_digest

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

    if USE_GEMINI:
        try:
            g = gemini_summarize_text_one_sentence(text)
            if g:
                summary = g
        except Exception:
            pass

    return jsonify({"summary": summary})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
