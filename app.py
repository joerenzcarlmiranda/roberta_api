from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load CardiffNLP Twitter RoBERTa model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

@app.route("/")
def home():
    return "RoBERTa Sentiment API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Please provide text in JSON payload"}), 400

    text = data["text"]
    result = sentiment_pipeline(text)

    # Format result as label + score
    response = [{"label": r["label"], "score": float(r["score"])} for r in result]
    return jsonify(response)

if __name__ == "__main__":
    # Railway uses PORT environment variable
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
