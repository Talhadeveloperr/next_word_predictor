import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer once
model = load_model("model/model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "?"
