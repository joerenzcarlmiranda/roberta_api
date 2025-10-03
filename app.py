from flask import Flask, request, jsonify
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
from deep_translator import GoogleTranslator

# Initialize Flask
app = Flask(__name__)

# Load CardiffNLP RoBERTa model (sentiment)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route("/sentiment", methods=["POST"])
def sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Translate Tagalog (or any language) to English
    translated_text = GoogleTranslator(source='auto', target='en').translate(text)

    # Run sentiment analysis
    result = nlp_pipeline(translated_text)[0]

    # Map CardiffNLP labels
    labels_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    sentiment_result = {
        "label": labels_map.get(result["label"], result["label"]),
        "score": float(result["score"]),
        "original_text": text,
        "translated_text": translated_text
    }
    return jsonify(sentiment_result)

# Health check
@app.route("/", methods=["GET"])
def index():
    return "CardiffNLP Sentiment API with Tagalog Support!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
