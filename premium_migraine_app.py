import streamlit as st
import pandas as pd
import numpy as np

# Simple configuration
st.set_page_config(page_title="Migraine Diagnosis", page_icon="🧠", layout="wide")

st.title("🧠 Migraine Diagnosis AI")
st.write("Simple and effective migraine diagnosis tool")

# Load data and train simple model
@st.cache_resource
def load_model():
    df = pd.read_csv('migraine_data.csv')
    
    # Simple encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    X = df.drop(['Type', 'Type_encoded'], axis=1)
    y = df['Type_encoded']
    
    # Simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, le

model, le = load_model()

# Simple interface
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 80, 35)
    duration = st.selectbox("Duration", [1, 2, 3])
    frequency = st.selectbox("Frequency", [1, 2, 3, 4, 5])

with col2:
    nausea = st.radio("Nausea", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    visual = st.selectbox("Visual Aura", [0, 1, 2])
    vertigo = st.radio("Vertigo", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("Diagnose"):
    features = [age, duration, frequency, 1, 1, 2, nausea, 0, 1, 1, visual, 0, 0, 0, vertigo, 0, 0, 0, 0, 0, 0, 0, 0]
    
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    diagnosis = le.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    st.success(f"**Diagnosis:** {diagnosis}")
    st.info(f"**Confidence:** {confidence:.1%}")

st.warning("Consult a doctor for medical advice.")
