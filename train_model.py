import json
import numpy as np
import random
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 200
BATCH_SIZE = 5
TEST_SIZE = 0.2

# Load processed files
with open("words.pkl", "rb") as f:
    words = pickle.load(f)
with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


def simple_tokenizer(text):
    return text.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").split()


# Build training data
X, y = [], []
for intent in intents['intents']:
    for pattern in intent['text']:
        tokenized = simple_tokenizer(pattern)
        bag = [1 if w in tokenized else 0 for w in words]
        X.append(bag)
        y.append(classes.index(intent['intent']))

X = np.array(X)
y_idx = np.array(y)
y = tf.keras.utils.to_categorical(y_idx, num_classes=len(classes))

# Stratified split so every class is represented in both train and validation sets
# (a plain random split can leave an entire class out of validation with this few
#  examples per class, which produces a misleadingly noisy accuracy number)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y_idx
)

print(f"Total samples: {len(X)}  |  Train: {len(X_train)}  |  Val: {len(X_val)}  |  "
      f"Vocabulary: {len(words)}  |  Classes: {len(classes)}")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(words),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=25, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# Report real, held-out performance
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"\nFinal training accuracy:   {train_acc * 100:.2f}%")
print(f"Final validation accuracy: {val_acc * 100:.2f}%")

# Save the model
model.save("model.h5")
print("\n✅ model.h5 has been created successfully!")
