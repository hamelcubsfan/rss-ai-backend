from flask import Flask, request, jsonify
import google.generativeai as genai
import os, re, threading, time
import feedparser
from flask_cors import CORS
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

# simple keyword bucket for talent‑movement tagging
MOVE_PAT = re.compile(
    r"(layoff|hiring|acqui|merge|ipo|fundrais|restructur|expan|headcount)",
    re.IGNORECASE,
)

def ai_summary(title, content, article_prompt=None):
    if article_prompt:
        prompt = article_prompt.format(title=title, content=content[:2000])
    else:
        prompt = (
            "You are an expert recruiting‑market analyst. "
            "Write ONE clear sentence summarizing this article, "
            "focusing on hiring, layoffs, acquisitions, funding, or notable org changes.\n\n"
            f"TITLE: {title}\n\nCONTENT: {content[:2000]}"
        )
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


def generate_digest(feeds, article_prompt=None, digest_prompt=None):
    """Summarize a list of feed URLs and return a digest payload."""
    all_sentences = []
    out = []

    for url in feeds[:10]:
        parsed = feedparser.parse(url)
        if not parsed.entries:
            continue

        feed_block = {"source": parsed.feed.get("title", url), "articles": []}

        for entry in parsed.entries[:5]:
            title = entry.get("title", "")
            body = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")

            summary = ai_summary(title, body, article_prompt)
            flag = bool(MOVE_PAT.search(title + " " + body))

            feed_block["articles"].append(
                {"title": title, "summary": summary, "link": link, "movement": flag}
            )
            all_sentences.append(summary)

        out.append(feed_block)

    if digest_prompt:
        prompt = digest_prompt + "\n" + "\n".join(all_sentences[:40])
    else:
        prompt = (
            "You are a recruiting-market analyst. "
            "Given these one-sentence article summaries, produce:\n"
            "1) A 2-sentence high-level overview.\n"
            "2) Five bullet points of the most relevant talent-movement stories.\n\n"
            + "\n".join(all_sentences[:40])
        )
    digest_resp = model.generate_content(prompt).text.strip()

    return {"digest": digest_resp, "feeds": out}

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    feeds = data.get("feedUrls", [])
    article_prompt = data.get("articlePrompt")
    digest_prompt = data.get("digestPrompt")

    if not isinstance(feeds, list) or not feeds:
        return jsonify({"error": "feedUrls should be a non-empty list"}), 400

    payload = generate_digest(feeds, article_prompt, digest_prompt)

    return jsonify(payload)


def background_worker():
    """Periodically fetch feeds and emit digests via Socket.IO."""
    feed_env = os.getenv("BACKGROUND_FEEDS", "")
    if not feed_env:
        return
    feeds = [f.strip() for f in feed_env.split(",") if f.strip()]
    if not feeds:
        return
    interval = int(os.getenv("FETCH_INTERVAL", "600"))
    while True:
        payload = generate_digest(feeds)
        socketio.emit("digest", payload)
        time.sleep(interval)

if __name__ == "__main__":
    if os.getenv("BACKGROUND_FEEDS"):
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
