# CUST University Chatbot

An intent-classification chatbot that answers common questions about Capital University of Science and Technology (CUST) — admissions, fees by department, scholarships, hostel costs, and university leadership — served through a Flask web app with a live chat UI.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00)
![Flask](https://img.shields.io/badge/Flask-2.3-000000)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Instead of relying on a large pretrained LLM, this project builds a lightweight intent classifier from scratch:

- **Bag-of-words** text representation over a 164-word vocabulary (no NLTK dependency — a custom tokenizer)
- A small **feed-forward neural network** (Dense 128 → Dropout → Dense 64 → Dropout → Softmax) trained to classify user input into one of 25 intents
- **25 intents** covering admissions, per-department fee structures, scholarships, hostel fees, HODs, and university leadership
- A **Flask backend** exposing a `/predict` API, with a browser-based chat interface for interacting with it live

## Results

Evaluated with a **stratified train/validation split** (ensuring all 25 intent classes are represented in both sets, rather than a naive random split):

| Metric | Score |
|---|---|
| Training accuracy | 100% |
| Validation accuracy | 70% |

**On the generalization gap:** several intents (e.g. `fee_structure_computer_science` vs. `fee_structure_english`) differ only by department name, which is genuinely difficult for a bag-of-words model to separate with a small number of examples per class. 70% validation accuracy is the honest, measured result — not tuned or cherry-picked — and reflects a known limitation of this architecture on a small dataset rather than a bug. See [Future Work](#future-work) for how this could be improved.

Spot-checked against realistic phrasing not seen verbatim in training:

| Query | Predicted intent | Confidence |
|---|---|---|
| "What is the fee for BS Artificial Intelligence?" | `fee_structure_artificial_intelligence` | 85.9% |
| "Who is the vice chancellor?" | `vice_chancellor` | 99.6% |
| "Where is CUST located?" | `university_location` | 99.8% |
| "Does CUST offer scholarships?" | `scholarships` | 100.0% |

## Demo

The chat UI shows the bot's response along with its predicted intent and confidence score for transparency:

```
You: What is the fee for BS Artificial Intelligence?
Bot: The tuition fee for BS Artificial Intelligence at CUST is PKR 292,500 per
     semester for Fall 2025, excluding the one-time admission fee of PKR 20,000...
     (fee_structure_artificial_intelligence, 85.9%)
```

## Project structure

```
├── app.py                      # Flask app: /predict, /health, /shutdown routes
├── index.html                  # Chat UI
├── train_model.py              # Trains the model with a stratified val split
├── generate_pickle_files.py    # Builds vocabulary (words.pkl) and intent labels (classes.pkl)
├── intents.json                # Training data: 25 intents, ~150 example phrases
├── model.h5                    # Trained model weights
├── words.pkl                   # Vocabulary used for bag-of-words encoding
├── classes.pkl                 # Intent label list
├── requirements.txt

```

## Running locally

```bash
git clone https://github.com/SyedKashaaf/cust-university-chatbot.git
cd cust-university-chatbot
pip install -r requirements.txt
python app.py
```

This opens `http://127.0.0.1:5000` in your browser with the chat interface.

**To retrain the model from scratch** (e.g. after editing `intents.json`):
```bash
python generate_pickle_files.py   # rebuild vocabulary + labels
python train_model.py              # retrain and save model.h5
```

## API

`POST /predict`
```json
{ "input": "What is the fee for BS Computer Science?" }
```
Response:
```json
{
  "response": "The tuition fee for BS Computer Science at CUST is PKR 292,500...",
  "intent": "fee_structure_computer_science",
  "confidence": 0.94
}
```

`GET /health` — returns model status and vocabulary/class counts.

## Tech stack

Python · TensorFlow/Keras · Flask · NumPy · scikit-learn (stratified split)

## Limitations

- Trained on ~150 example phrases across 25 intents — sufficient for a course project, not production scale
- No entity extraction — each department has its own dedicated intent rather than a single "fee lookup" intent with a department slot, which limits scalability as more departments are added
- No conversational memory — each message is classified independently, with no multi-turn context
- Fee, admission-date, and personnel data are hardcoded for Fall 2025 and will need periodic manual updates

## Future work

- Replace bag-of-words with sentence embeddings (e.g. sentence-transformers) to better generalize across similarly-worded intents and close the train/validation gap
- Add entity extraction (department name) to collapse the 15 near-duplicate `fee_structure_X` intents into a single parametrized intent
- Add conversation history / multi-turn context
- Deploy publicly (Render/Railway) with a scheduled job to refresh fee and admission data

## Author

**Syed Muhammad Kashaaf Haider** — BS Artificial Intelligence, Capital University of Science and Technology (CUST), Islamabad
[GitHub](https://github.com/SyedKashaaf) · [LinkedIn](https://linkedin.com/in/syed-kashaaf-haider-10552833a)

## License

MIT
