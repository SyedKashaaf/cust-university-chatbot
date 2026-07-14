import json
import pickle

# Load intents
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

# Basic tokenizer (no NLTK)
def simple_tokenizer(text):
    for ch in ignore_chars:
        text = text.replace(ch, "")
    return text.lower().split()

for intent in data['intents']:
    for pattern in intent['text']:
        word_list = simple_tokenizer(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['intent']))
        if intent['intent'] not in classes:
            classes.append(intent['intent'])

words = sorted(set(words))
classes = sorted(set(classes))

# Save to pickle files
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

print("✅ Pickle files created without NLTK: words.pkl & classes.pkl")
