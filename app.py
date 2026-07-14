import os
import json
import random
import pickle
import threading
import webbrowser

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

ERROR_THRESHOLD = 0.25

# Load data and model
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)


# === Chatbot logic ===
def clean_up_sentence(sentence):
    sentence = sentence.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "")
    return sentence.split()


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]


def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I'm sorry, I didn't understand that. Could you rephrase?"
    tag = ints[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["intent"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."


# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "input" not in data or not str(data["input"]).strip():
        return jsonify({"error": "Missing or empty 'input' field"}), 400

    user_input = str(data["input"]).strip()
    ints = predict_class(user_input)
    result = get_response(ints, intents)

    return jsonify({
        "response": result,
        "intent": ints[0]["intent"] if ints else None,
        "confidence": round(ints[0]["probability"], 4) if ints else None,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": len(classes), "vocab_size": len(words)})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    shutdown_func = request.environ.get("werkzeug.server.shutdown")
    if shutdown_func is None:
        return "Server shutdown not available (not running werkzeug dev server)."
    shutdown_func()
    return "Chatbot server is shutting down..."


# === Auto-open browser on local run ===
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    if not debug_mode:
        threading.Timer(1.5, open_browser).start()
    app.run(debug=debug_mode)
