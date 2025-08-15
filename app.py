# app.py
import os
import re
import json
import logging
import html as html_lib
from typing import List

from flask import Flask, request, jsonify

# Google GenAI SDK imports
from google import genai
from google.genai import types

# ---- Configuration ----
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# prefer GEMINI_API_KEY name used in official examples, but fall back to GENAI_API_KEY if present
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
# default model (adjust if you prefer another)
GENAI_MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-pro")

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("rss-ai-backend")

# initialize client; if API key provided pass it, otherwise client will try environment
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    # If no API key provided, client will attempt to pick from env (GEMINI_API_KEY) or other auth flows
    client = genai.Client()

app = Flask(__name__)

# ---- Helpers ----
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(html: str) -> str:
    """Strip HTML tags, unescape HTML entities, and collapse whitespace."""
    if not html:
        return ""
    text = TAG_RE.sub(" ", html)
    text = html_lib.unescape(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def first_sentence(text: str) -> str:
    """Return the first sentence-like fragment, short and tidy."""
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
    """
    Wrapper around google-genai sync generate_content.
    Returns the text portion of the model response, or empty string on error.
    """
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        # response typically has .text attribute
        return getattr(resp, "text", "") or ""
    except Exception as e:
        log.exception("generate_content call failed")
        return f"[model error: {e}]"


# ---- AI functions and prompts ----
ARTICLE_PROMPT_TEMPLATE = (
    "You are a recruiting-market analyst. ONLY use facts contained in the ARTICLE_CONTENT block below. "
    "Do not invent details, dates, numbers, or people. If the article contains no hiring, layoffs, funding, "
    "acquisitions, leadership change, or org-change signals, reply exactly: No relevant recruiting signals found.\n\n"
    "TITLE: {title}\n\n"
    "ARTICLE_CONTENT: {content}\n\n"
    "INSTRUCTIONS: Produce one concise sentence focused on recruiting, hiring, leadership, funding, "
    "or organizational signals present in the ARTICLE_CONTENT. Keep the sentence short and factual. "
    "Do not add context or other facts not present in ARTICLE_CONTENT."
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


def ai_summary(title: str, content: str) -> str:
    """Generate a short, factual one-sentence summary focused on recruiting signals."""
    content_clean = clean_text(content or "")
    content_chunk = content_clean[:6000]
    prompt = ARTICLE_PROMPT_TEMPLATE.format(title=title or "", content=content_chunk)
    text = call_generate(GENAI_MODEL, prompt, max_tokens=120, temperature=0.0).strip()
    text = text.replace("\n", " ").strip()
    if not text:
        return "No relevant recruiting signals found."
    return first_sentence(text)


def ai_digest(summaries: List[str]) -> str:
    """Aggregate one-sentence summaries into a compact digest with strict constraints."""
    cleaned = [s.strip() for s in summaries if s and not s.lower().startswith("[model error:")]
    if not cleaned:
        return "No recruiting signals across these articles."
    cleaned = cleaned[:50]
    summaries_block = "\n".join(f"- {s}" for s in cleaned)
    prompt = DIGEST_PROMPT_TEMPLATE.format(summaries=summaries_block)
    text = call_generate(GENAI_MODEL, prompt, max_tokens=350, temperature=0.0).strip()
    return text


# ---- Flask endpoints ----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/summarize_article", methods=["POST"])
def summarize_article():
    payload = request.get_json(force=True, silent=True) or {}
    title = payload.get("title", "")
    content = payload.get("content", "")
    if not content and "url" in payload:
        return jsonify({"error": "No content provided. Provide article content or call fetch endpoint first."}), 400
    summary = ai_summary(title, content)
    return jsonify({"summary": summary})


@app.route("/summarize_batch", methods=["POST"])
def summarize_batch():
    payload = request.get_json(force=True, silent=True) or {}
    articles = payload.get("articles") or []
    if not isinstance(articles, list) or not articles:
        return jsonify({"error": "Provide a non-empty articles list"}), 400

    results = []
    all_summaries = []
    for art in articles:
        title = art.get("title") or art.get("headline") or ""
        content = art.get("content") or art.get("summary") or ""
        url = art.get("url") or ""
        summary = ai_summary(title, content)
        results.append({"title": title, "summary": summary, "url": url})
        all_summaries.append(summary)

    digest = ai_digest(all_summaries)
    return jsonify({"summaries": results, "digest": digest})


@app.route("/debug_prompt", methods=["POST"])
def debug_prompt():
    payload = request.get_json(force=True, silent=True) or {}
    prompt = payload.get("prompt", "")
    which = payload.get("which", "article")
    if which == "digest":
        text = call_generate(GENAI_MODEL, prompt, max_tokens=350, temperature=0.0)
        return jsonify({"raw": text})
    else:
        text = call_generate(GENAI_MODEL, prompt, max_tokens=120, temperature=0.0)
        return jsonify({"raw": text})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
