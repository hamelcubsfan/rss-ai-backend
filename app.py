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
    try:
        data = request.json
        feed_urls = data.get("feedUrls", [])

        if not feed_urls or not isinstance(feed_urls, list):
            return jsonify({"error": "feedUrls must be a list of URLs"}), 400

        result = []
        for url in feed_urls[:10]:
            print(f"⏳ Parsing feed: {url}")
            feed = feedparser.parse(url)
            if not feed.entries:
                continue

            feed_summaries = []
            for entry in feed.entries[:5]:
                title = entry.get("title", "")
                content = entry.get("summary", "") or entry.get("description", "")
                link = entry.get("link", "")

                try:
                    prompt = f"Summarize the following news article clearly and concisely:\n\nTitle: {title}\n\n{content}"
                    response = model.generate_content(prompt)
                    summary = response.text.strip()
                except Exception as e:
                    summary = f"[Gemini error: {str(e)}]"

                feed_summaries.append({
                    "title": title,
                    "summary": summary,
                    "link": link
                })

            result.append({
                "source": feed.feed.get("title", url),
                "summaries": feed_summaries
            })

        return jsonify({ "feeds": result })
    except Exception as e:
        print(f"❌ Uncaught error in summarize(): {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
