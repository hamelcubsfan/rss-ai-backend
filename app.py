# app.py
import os
import re
import json
import logging
import html as html_lib
from typing import List, Dict

from flask import Flask, request, jsonify
import genai

# ---- Configuration ----
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", None)
GENAI_MODEL = os.environ.get("GENAI_MODEL", "models/gemini-2.5-pro")

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("rss-ai-backend")

if not GENAI_API_KEY:
    log.warning("GENAI_API_KEY not set. Set this in your environment for the model to work.")
else:
    genai.configure(api_key=GENAI_API_KEY)

# Generation configs: deterministic for article summaries to reduce hallucination,
# slightly larger budget for digest generation.
generation_cfg_article = genai.GenerationConfig(temperature=0.0, max_output_tokens=120)
generation_cfg_digest = genai.GenerationConfig(temperature=0.0, max_output_tokens=350)

# Models (two instances share same model name but different generation configs)
model_article = genai.GenerativeModel(GENAI_MODEL, generation_config=generation_cfg_article)
model_digest = genai.GenerativeModel(GENAI_MODEL, generation_config=generation_cfg_digest)

app = Flask(__name__)

# ---- Helpers ----
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(html: str) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace."""
    if not html:
        return ""
    text = TAG_RE.sub(" ", html)
    text = html_lib.unescape(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def first_sentence(text: str) -> str:
    """Return the first sentence-like fragment. Keep it short."""
    if not text:
        return ""
    # Split on period, question mark, exclamation, or newline
    parts = re.split(r"[.?!\n]", text)
    first = parts[0].strip()
    # If too short, fallback to a trimmed substring
    if len(first) < 10 and len(text) > 0:
        first = text.strip().split("\n")[0][:200].strip()
    if not first:
        return ""
    # Ensure it ends with a period for UI consistency
    return first.rstrip() + "."


def safe_extract_text_from_response(resp) -> str:
    """
    Extract the generated text from the SDK response in a robust way.
    The SDK may return different shapes; handle common ones.
    """
    if resp is None:
        return ""
    # try common attributes in order
    if hasattr(resp, "text"):
        return getattr(resp, "text") or ""
    if isinstance(resp, dict):
        # Attempt to navigate possible dict shape
        # e.g., {"generations": [{"text": "..."}, ...]}
        gens = resp.get("generations") or resp.get("choices") or []
        if gens and isinstance(gens, list) and "text" in gens[0]:
            return gens[0]["text"] or ""
        # fallback to 'text' key
        return resp.get("text", "") or ""
    # Last-resort string conversion
    return str(resp)


# ---- AI functions ----
ARTICLE_PROMPT_TEMPLATE = (
    "You are a recruiting-market analyst. ONLY use facts contained in the ARTICLE_CONTENT block below. "
    "Do not invent details, dates, numbers, or persons. If the article contains no hiring, layoffs, funding, "
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
    "each bullet 1-2 short phrases (e.g., 'Company X hiring surge', 'Acquisition Y prompts restructuring').\n\n"
    "If the provided summaries are all 'No relevant recruiting signals found' or empty, reply: No recruiting signals across these articles.\n\n"
    "SUMMARIES:\n{summaries}\n\n"
    "OUTPUT:"
)


def ai_summary(title: str, content: str) -> str:
    """Generate a short, factual one-sentence summary focused on recruiting signals."""
    content_clean = clean_text(content)
    # keep a reasonable chunk of the article but not infinite
    content_chunk = content_clean[:6000]  # adjust as needed for token budget
    prompt = ARTICLE_PROMPT_TEMPLATE.format(title=title or "", content=content_chunk)
    try:
        resp = model_article.generate_content(prompt)
        text = safe_extract_text_from_response(resp).strip()
        text = text.replace("\n", " ").strip()
        if not text:
            return "No relevant recruiting signals found."
        # Normalize and return only the first sentence to avoid long outputs
        return first_sentence(text)
    except Exception as e:
        log.exception("Error calling model_article.generate_content")
        return f"[model error: {e}]"


def ai_digest(summaries: List[str]) -> str:
    """Aggregate short summaries into a compact digest with strict constraints."""
    # filter and normalize summaries
    cleaned = [s.strip() for s in summaries if s and not s.lower().startswith("[model error:")]
    if not cleaned:
        return "No recruiting signals across these articles."
    # If many, limit to first 50 to keep digest focused
    cleaned = cleaned[:50]
    summaries_block = "\n".join(f"- {s}" for s in cleaned)
    prompt = DIGEST_PROMPT_TEMPLATE.format(summaries=summaries_block)
    try:
        resp = model_digest.generate_content(prompt)
        text = safe_extract_text_from_response(resp).strip()
        return text
    except Exception as e:
        log.exception("Error calling model_digest.generate_content")
        return f"[model error: {e}]"


# ---- Flask endpoints ----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/summarize_article", methods=["POST"])
def summarize_article():
    """
    Expect JSON: { "title": "...", "content": "..." }
    Returns: { "summary": "..." }
    """
    payload = request.get_json(force=True, silent=True) or {}
    title = payload.get("title", "")
    content = payload.get("content", "")
    if not content and "url" in payload:
        # optional: if frontend sends url only, respond with an error so frontend can fetch content
        return jsonify({"error": "No content provided. Provide article content or call fetch endpoint first."}), 400
    summary = ai_summary(title, content)
    return jsonify({"summary": summary})


@app.route("/summarize_batch", methods=["POST"])
def summarize_batch():
    """
    Expect JSON: { "articles": [ {"title":"", "content":"", "url":""}, ... ] }
    Returns: {
       "summaries": [{"title":"", "summary":"", "url":""}, ...],
       "digest": "..."
    }
    """
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

    # create digest from the one-sentence summaries
    digest = ai_digest(all_summaries)

    return jsonify({"summaries": results, "digest": digest})


# Light-weight endpoint to test prompt behavior without article content
@app.route("/debug_prompt", methods=["POST"])
def debug_prompt():
    payload = request.get_json(force=True, silent=True) or {}
    prompt = payload.get("prompt", "")
    which = payload.get("which", "article")
    if which == "digest":
        try:
            resp = model_digest.generate_content(prompt)
            return jsonify({"raw": safe_extract_text_from_response(resp)})
        except Exception as e:
            log.exception("Error debug digest")
            return jsonify({"error": str(e)}), 500
    else:
        try:
            resp = model_article.generate_content(prompt)
            return jsonify({"raw": safe_extract_text_from_response(resp)})
        except Exception as e:
            log.exception("Error debug article")
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
