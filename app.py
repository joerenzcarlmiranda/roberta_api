from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline

app = Flask(__name__)

# Load the pre-trained CardiffNLP Twitter RoBERTa model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@app.route("/")
def home():
    return "Roberta Sentiment API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    result = sentiment_pipeline(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
