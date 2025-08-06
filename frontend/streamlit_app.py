import streamlit as st
import requests

st.title("📚 Next Word Predictor")
st.markdown("Enter a sentence below to predict the next word based on Shakespearean text.")

user_input = st.text_input("Your sentence:")

if st.button("Predict"):
    if user_input.strip():
        response = requests.post("http://localhost:8000/predict", json={"text": user_input})
        st.success(f"Next word might be: **{response.json()['next_word']}**")
    else:
        st.warning("Please enter some text.")