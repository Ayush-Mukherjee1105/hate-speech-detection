# src/app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from langdetect import detect

# Always force CPU
device = torch.device("cpu")

# Set page config first
st.set_page_config(page_title="Hate Speech Detection", page_icon="🔩", layout="centered")

# Hide anchor link icons (those annoying 🔗 next to headers)
hide_streamlit_style = """
    <style>
    [data-testid="stMarkdownContainer"] a {
        display: none;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Class labels
class_labels = ["Non-Offensive", "Hate/Offensive"]

# Load models and tokenizers
@st.cache_resource
def load_models():
    mbert_tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned/mbert")
    mbert_model = AutoModelForSequenceClassification.from_pretrained("models/fine_tuned/mbert").to(device)

    xlmr_tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned/xlmr")
    xlmr_model = AutoModelForSequenceClassification.from_pretrained("models/fine_tuned/xlmr").to(device)

    return mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model

# Load once
mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model = load_models()

# Prediction function
def predict(text, model_choice):
    if model_choice == "mBERT":
        inputs = mbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = mbert_model(**inputs)
    else:
        inputs = xlmr_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = xlmr_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy().flatten()
    prediction = probs.argmax()
    confidence = probs[prediction]
    return class_labels[prediction], confidence, probs


# Language detection function
def detect_language(text):
    try:
        lang = detect(text)
        lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
        return lang_map.get(lang, "Unknown")
    except:
        return "Unknown"

# Streamlit UI
st.title("🔩 Multilingual Hate Speech Detection")

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ["mBERT", "XLM-R"])

st.markdown("### Enter Text to Analyze")
text_input = st.text_area("Type your comment/post here:", height=150, key="input_text")

# Only show Predict button if text is entered
if text_input.strip():
    if st.button("Predict"):
        with st.spinner('Analyzing...'):
            label, confidence, probs = predict(text_input, model_choice)
            language = detect_language(text_input)

        st.success(f"**Prediction:** {label}  \n**Confidence:** {confidence:.2%}  \n**Detected Language:** {language}")

        # Plot pie chart
        st.markdown("### 🔍 Prediction Confidence Breakdown")
        fig, ax = plt.subplots()
        ax.pie(probs, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
        ax.axis('equal')
        st.pyplot(fig)
else:
    st.info("🔍 Please enter some text above to enable prediction.")
