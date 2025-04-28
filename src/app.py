import streamlit as st
import torch 
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
@st.cache_resource
def load_models():
    mbert_tokenizer = BertTokenizer.from_pretrained("models/fine_tuned/mbert")
    mbert_model = BertForSequenceClassification.from_pretrained("models/fine_tuned/mbert").to(device)
    
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("models/fine_tuned/xlm-r")
    xlmr_model = XLMRobertaForSequenceClassification.from_pretrained("models/fine_tuned/xlm-r").to(device)
    
    return mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model

mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model = load_models()

# Class labels (match your training labels)
class_labels = ["Normal", "Offensive", "HateSpeech"]

# Prediction function
def predict(text, model_name):
    if model_name == "mBERT":
        inputs = mbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = mbert_model(**inputs)
    else:
        inputs = xlmr_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = xlmr_model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = probs.argmax(dim=-1).item()
    confidence = probs.max().item()
    return class_labels[prediction], confidence

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detection", page_icon="🛡️", layout="centered")
st.title("🛡️ Multilingual Hate Speech Detection")

with st.sidebar:
    st.header("Settings ⚙️")
    model_choice = st.selectbox("Select Model", ["mBERT", "XLM-R"])
    st.markdown("---")
    st.caption("Developed for multilingual hate speech research.")

st.write("Enter a sentence to detect if it is **Normal**, **Offensive**, or **Hate Speech**.")

text = st.text_area("Text Input", height=150, placeholder="Type something here...")

if st.button("Predict 🚀"):
    if not text.strip():
        st.warning("⚠️ Please enter some text for prediction!")
    else:
        with st.spinner("Analyzing..."):
            prediction, confidence = predict(text, model_choice)
        if confidence < 0.6:
            st.error(f"⚠️ Low confidence prediction: {prediction} ({confidence:.2f})")
        else:
            st.success(f"✅ Prediction ({model_choice}): **{prediction}** with confidence {confidence:.2f}")

st.markdown("---")
st.caption("Made with ❤️ using BERT and XLM-RoBERTa models.")
