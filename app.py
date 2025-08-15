# app.py
import os
import re
import logging
import html as html_lib
from typing import List
from urllib.parse import urlparse

import requests
import feedparser
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# google-genai imports
from google import genai
from google.genai import types

# ---- config ----
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
GENAI_MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-pro")

MAX_ARTICLES = int(os.environ.get("MAX_ARTICLES", 40))  # guard for cost/perf

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("rss-ai-backend")

# client init
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = genai.Client()  # will use env-based auth if available

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---- helpers ----
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text_from_html(html: str) -> str:
    """Strip tags, unescape, collapse whitespace and trim."""
    if not html:
        return ""
    # Use BeautifulSoup for nicer extraction if the input is complex
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        text = TAG_RE.sub(" ", html)
    text = html_lib.unescape(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def fetch_article_text(url: str) -> str:
    """Try to fetch page and extract main text. Keep it defensive."""
    try:
        headers = {"User-Agent": "rss-ai-backend/1.0"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return ""
        return clean_text_from_html(r.text)[:10000]
    except Exception:
        return ""


def first_sentence(text: str) -> str:
    """Return a short first-sentence fragment."""
    if not text:
        return ""
    parts = re.split(r"[.?!\n]", text)
    first = parts[0].strip()
    if len(first) < 10 and len(text) > 0:
        first = text.strip().split("\n")[0][:200].strip()
    if not first:
        return ""
    return first.rstrip() + "."


def call_generate(model_name: str, prompt: str, max_tokens: int = 120, temperature: float = 0.0) -> str:
    """Wrapper to call google-genai and return text or an error string."""
    try:
        cfg = types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature)
        resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return getattr(resp, "text", "") or ""
    except Exception as e:
        log.exception("genai.generate_content failure")
        return f"[model error: {e}]"


# ---- prompts ----
ARTICLE_PROMPT_TEMPLATE = (
    "You are a recruiting-market analyst. ONLY use facts contained in the ARTICLE_CONTENT block below. "
    "Do not invent details, dates, numbers, or people. If the article contains no hiring, layoffs, funding, "
    "acquisitions, leadership change, or org-change signals, reply exactly: No relevant recruiting signals found.\n\n"
    "TITLE: {title}\n\n"
    "ARTICLE_CONTENT: {content}\n\n"
    "INSTRUCTIONS: Produce one concise sentence focused on recruiting signals present in ARTICLE_CONTENT. "
    "Keep it short and factual. Do not add context or other facts not present in ARTICLE_CONTENT."
)

DIGEST_PROMPT_TEMPLATE = (
    "You are a recruiting-market analyst. ONLY use the one-sentence summaries provided below. "
    "Do not invent facts. From these sentences, produce:\n"
    "1) One 1-sentence high-level trend summary.\n"
    "2) A short bulleted list (up to 5 bullets) of the most relevant recruiting signals, "
    "each bullet 1-2 short phrases (for example: 'Company X hiring surge', 'Acquisition Y prompts restructuring').\n\n"
    "If the provided summaries are all 'No relevant recruiting signals found' or empty, reply: No recruiting signals across these articles.\n\n"
    "SUMMARIES:\n{summaries}\n\n"
    "OUTPUT:"
)


# ---- ai helpers ----
def ai_summary(title: str, content: str) -> str:
    content_clean = clean_text_from_html(content or "")
    chunk = content_clean[:6000]
    prompt = ARTICLE_PROMPT_TEMPLATE.format(title=title or "", content=chunk)
    text = call_generate(GENAI_MODEL, prompt, max_tokens=120, temperature=0.0).strip()
    text = text.replace("\n", " ").strip()
    if not text:
        return "No relevant recruiting signals found."
    return first_sentence(text)


def ai_digest(summaries: List[str]) -> str:
    cleaned = [s.strip() for s in summaries if s and not s.lower().startswith("[model error:")]
    if not cleaned:
        return "No recruiting signals across these articles."
    cleaned = cleaned[:50]
    summaries_block = "\n".join(f"- {s}" for s in cleaned)
    prompt = DIGEST_PROMPT_TEMPLATE.format(summaries=summaries_block)
    text = call_generate(GENAI_MODEL, prompt, max_tokens=350, temperature=0.0).strip()
    return text


# ---- endpoints ----
@app.errorhandler(404)
def handle_404(e):
    return make_response(jsonify({"error": "not found", "path": request.path}), 404)


@app.errorhandler(Exception)
def handle_exception(e):
    log.exception("Unhandled exception")
    return make_response(jsonify({"error": "internal_server_error", "message": str(e)}), 500)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/summarize_article", methods=["POST"])
def summarize_article():
    payload = request.get_json(force=True, silent=True) or {}
    title = payload.get("title", "")
    content = payload.get("content", "")
    url = payload.get("url", "")
    if not content and url:
        content = fetch_article_text(url)
    if not content:
        return jsonify({"summary": "No relevant recruiting signals found."})
    summary = ai_summary(title, content)
    return jsonify({"summary": summary})


@app.route("/summarize_batch", methods=["POST"])
def summarize_batch():
    payload = request.get_json(force=True, silent=True) or {}
    articles = payload.get("articles") or []
    if not isinstance(articles, list) or not articles:
        return make_response(jsonify({"error": "Provide a non-empty articles list"}), 400)

    results = []
    all_summaries = []
    for art in articles[:MAX_ARTICLES]:
        title = art.get("title") or art.get("headline") or ""
        content = art.get("content") or art.get("summary") or ""
        url = art.get("url") or ""
        if not content and url:
            content = fetch_article_text(url)
        summary = ai_summary(title, content)
        results.append({"title": title, "summary": summary, "url": url})
        all_summaries.append(summary)

    digest = ai_digest(all_summaries)
    return jsonify({"summaries": results, "digest": digest})


@app.route("/summarize", methods=["POST"])
def summarize_feeds():
    """
    New helper endpoint expected by the extension.
    Accepts JSON:
    {
      "feedUrls": ["https://..."],
      "articlePrompt": "...",  # optional, not used now
      "digestPrompt": "..."    # optional
    }
    Returns JSON with summaries and digest.
    """
    payload = request.get_json(force=True, silent=True) or {}
    feed_urls = payload.get("feedUrls") or []
    if not isinstance(feed_urls, list) or not feed_urls:
        return make_response(jsonify({"error": "Provide feedUrls list"}), 400)

    articles = []
    seen = set()
    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            parsed = None
        if not parsed or not getattr(parsed, "entries", None):
            continue
        for entry in parsed.entries:
            if len(articles) >= MAX_ARTICLES:
                break
            title = entry.get("title", "") or entry.get("headline", "")
            url = entry.get("link", "") or entry.get("id", "")
            # dedupe by (hostname + path) when possible
            key = (url or title).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            # extract content preference: content > summary > fetch from url
            content = ""
            if "content" in entry and isinstance(entry.content, list) and entry.content:
                content = entry.content[0].value
            elif entry.get("summary"):
                content = entry.get("summary")
            elif url:
                content = fetch_article_text(url)
            # keep it short on server side
            articles.append({"title": title, "content": content, "url": url})

    if not articles:
        return jsonify({"summaries": [], "digest": "No articles found from provided feeds."})

    # summarize and build digest
    summaries = []
    results = []
    for art in articles:
        s = ai_summary(art.get("title"), art.get("content"))
        results.append({"title": art.get("title"), "summary": s, "url": art.get("url")})
        summaries.append(s)

    digest = ai_digest(summaries)
    return jsonify({"summaries": results, "digest": digest})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
