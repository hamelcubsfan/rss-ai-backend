from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import feedparser
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")

@app.route("/")
def index():
    return jsonify({"message": "RSS Summarizer API is running."})

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    feed_url = data.get("feedUrl")

    if not feed_url:
        return jsonify({"error": "Missing feedUrl"}), 400

    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return jsonify({"error": "No articles found in feed"}), 404

    summaries = []
    for entry in feed.entries[:5]:
        title = entry.get("title", "")
        content = entry.get("summary", "") or entry.get("description", "")
        url = entry.get("link", "")

        try:
            prompt = f"Summarize the following news article clearly and concisely:\n\nTitle: {title}\n\n{content}"
            response = model.generate_content(prompt)
            summaries.append({
                "title": title,
                "summary": response.text.strip(),
                "link": url
            })
        except Exception as e:
            summaries.append({
                "title": title,
                "summary": f"[Error: {str(e)}]",
                "link": url
            })

    return jsonify({"summaries": summaries})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
