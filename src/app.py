import streamlit as st
import torch 
from transformers import (
    BertTokenizer, BertForSequenceClassification,  # mBERT
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification  # XLM-R
)

# Load both models
@st.cache_resource
def load_models():
    mbert_tokenizer = BertTokenizer.from_pretrained("models/fine_tuned/mbert")
    mbert_model = BertForSequenceClassification.from_pretrained("models/fine_tuned/mbert")
    
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained("models/fine_tuned/xlm-r")
    xlmr_model = XLMRobertaForSequenceClassification.from_pretrained("models/fine_tuned/xlm-r")
    
    return mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model

mbert_tokenizer, mbert_model, xlmr_tokenizer, xlmr_model = load_models()

# Prediction function
def predict(text, model_name):
    if model_name == "mBERT":
        inputs = mbert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = mbert_model(**inputs)
    else:
        inputs = xlmr_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = xlmr_model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = probs.argmax().item()
    return ["Hateful", "Non-Hateful", "Neutral"][prediction], probs.max().item()

# Streamlit UI
st.title("Multilingual Hate Speech Detection")
text = st.text_area("Enter text:")
model_choice = st.selectbox("Select Model", ["mBERT", "XLM-R"])

if st.button("Predict"):
    prediction, confidence = predict(text, model_choice)
    st.write(f"**Prediction ({model_choice}):** {prediction} (Confidence: {confidence:.2f})")