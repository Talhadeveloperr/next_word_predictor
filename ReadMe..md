
---

```markdown
# 🧠 Next Word Predictor

A full-stack Machine Learning application that predicts the **next word** in a given sentence using a deep learning model trained on Shakespeare's text.  
This project includes:

- 🔄 Data preprocessing and sequence generation with TensorFlow
- 🧠 Model training using LSTM + Bidirectional RNN
- 🚀 FastAPI backend for real-time prediction
- 🎨 Streamlit frontend for user interaction

---

## 📁 Project Structure

```

next\_word\_predictor/
├── api/                    # FastAPI backend
│   └── app.py              # API with POST /predict
├── model/                  # Trained model & tokenizer
│   ├── model.h5
│   ├── tokenizer.pkl
│   ├── X.npy
│   └── y.npy
├── training/               # Training pipeline
│   ├── preprocess.py       # Text preprocessing & dataset creation
│   └── train.py            # LSTM model training script
├── predict.py              # Next word prediction logic
├── streamlit\_app.py        # Frontend for interacting with the API
├── data/
│   └── shakespeare.txt     # Raw dataset used for training
└── README.md               # Project documentation

````

---

## 🧪 Model Architecture

- 🔠 Embedding Layer
- 🔁 Bidirectional LSTM (150 units)
- 💧 Dropout (0.2)
- 🔁 LSTM (100 units)
- 🧮 Dense Softmax output layer

Loss function: `categorical_crossentropy`  
Optimizer: `adam`

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/next_word_predictor.git
cd next_word_predictor
````

2. **Install the required packages:**

```bash
pip install -r requirements.txt
```

> Create `requirements.txt` with:

```txt
tensorflow
numpy
pickle-mixin
streamlit
fastapi
uvicorn
requests
```

---

## ⚙️ Usage

### 1. Train the Model

```bash
cd training
python preprocess.py     # Prepares X.npy, y.npy, tokenizer
python train.py          # Trains and saves the model
```

### 2. Start the FastAPI Backend

```bash
uvicorn api.app:app --reload
```

API will be live at: [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

### 3. Run the Streamlit Frontend

In a new terminal:

```bash
streamlit run streamlit_app.py
```

Streamlit UI will open in the browser: [http://localhost:8501](http://localhost:8501)

---

## 📌 Sample Input

> Input:

```
to be or not
```

> Output:

```
Predicted next word: to
```

---

## 🚀 Deployment Suggestions

* **API**: Deploy FastAPI backend on [Render](https://render.com) or [Railway](https://railway.app)
* **Frontend**: Deploy Streamlit app on [Streamlit Cloud](https://streamlit.io/cloud)
* **Storage**: Use AWS S3 or GitHub for storing model assets (`model.h5`, `tokenizer.pkl`)

---

## 📚 Dataset

The dataset used is [Shakespeare’s Complete Works](https://ocw.mit.edu/ans7870/6/6.005/s16/psets/ps1/shakespeare.txt), cleaned and tokenized for training.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss your idea.



---

