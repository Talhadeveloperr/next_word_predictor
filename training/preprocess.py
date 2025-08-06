import os
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === CONFIG ===
SEQ_LEN = 5
USE_ONE_HOT = False  # Change to True if you have enough RAM (>10GB)

def load_and_clean_text(filepath):
    print("📖 Reading text file...")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r'\s+', ' ', text)
    print("✅ Text cleaned.")
    return text.strip()

def create_sequences(text, seq_len=SEQ_LEN):
    print("🔠 Tokenizing text...")
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    words = text.split()

    print(f"🧠 Creating {len(words)-seq_len} sequences...")
    for i in range(seq_len, len(words)):
        seq = words[i-seq_len:i+1]
        encoded = tokenizer.texts_to_sequences([" ".join(seq)])[0]
        if len(encoded) == seq_len + 1:
            input_sequences.append(encoded)

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=seq_len+1, padding='pre'))
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    if USE_ONE_HOT:
        print("⚙️ One-hot encoding target variable...")
        try:
            y = np.eye(total_words)[y]
        except MemoryError:
            print("❌ MemoryError: Reduce dataset or turn off one-hot encoding.")
            raise

    return X, y, tokenizer, total_words

if __name__ == '__main__':
    print("🚀 Starting preprocessing...")
    input_path = "../data/shakespeare.txt"
    output_dir = "../model"
    os.makedirs(output_dir, exist_ok=True)

    try:
        text = load_and_clean_text(input_path)
        X, y, tokenizer, total_words = create_sequences(text)
        print(f"✅ Data shapes: X = {X.shape}, y = {y.shape}, vocab = {total_words}")

        # Save files
        x_path = os.path.join(output_dir, "X.npy")
        y_path = os.path.join(output_dir, "y.npy")
        tokenizer_path = os.path.join(output_dir, "tokenizer.pkl")

        np.save(x_path, X)
        print(f"✅ Saved X to {x_path} ({X.nbytes / (1024**2):.2f} MB)")

        np.save(y_path, y)
        print(f"✅ Saved y to {y_path} ({y.nbytes / (1024**2):.2f} MB)")

        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"✅ Saved tokenizer to {tokenizer_path}")

        print("🎉 All preprocessing complete!")

    except Exception as e:
        print(f"❌ ERROR: {e}")
