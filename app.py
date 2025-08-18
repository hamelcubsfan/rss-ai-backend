from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os, re
import feedparser

app = Flask(__name__)
CORS(app)  # <-- This is the most permissive and safe for debugging!

# Catch-all logger for every request
@app.before_request
def log_request():
    print(f"==> LOG: {request.method} {request.path} | headers: {dict(request.headers)}")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

MOVE_PAT = re.compile(r"(layoff|hiring|acqui|merge|ipo|fundrais|restructur|expan|headcount)", re.IGNORECASE)

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

@app.route("/summarize", methods=["POST"])
def summarize():
    print("Received POST to /summarize")
    data = request.json
    feeds = data.get("feedUrls", [])
    article_prompt = data.get("articlePrompt")
    digest_prompt = data.get("digestPrompt")

    if not isinstance(feeds, list) or not feeds:
        print("feedUrls missing or not a list")
        return jsonify({"error": "feedUrls should be a non‑empty list"}), 400

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

            feed_block["articles"].append({
                "title": title,
                "summary": summary,
                "link": link,
                "movement": flag
            })
            all_sentences.append(summary)

        out.append(feed_block)

    if digest_prompt:
        prompt = digest_prompt + "\n" + "\n".join(all_sentences[:40])
    else:
        prompt = (
            "You are a recruiting‑market analyst. "
            "Given these one‑sentence article summaries, produce:\n"
            "1) A 2‑sentence high‑level overview.\n"
            "2) Five bullet points of the most relevant talent‑movement stories.\n\n"
            + "\n".join(all_sentences[:40])
        )
    digest_resp = model.generate_content(prompt).text.strip()
    print("Finished processing /summarize")

    return jsonify({"digest": digest_resp, "feeds": out})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
