
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ===== PREMIUM CONFIGURATION =====
st.set_page_config(
    page_title="NeuroScan Pro • Migraine Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS FOR PREMIUM LOOK =====
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
    }
    .premium-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    .emergency-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .symptom-score {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ===== AI MODEL LOADING =====
@st.cache_resource
def load_advanced_model():
    df = pd.read_csv('migraine_data.csv')
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    X = df.drop(['Type', 'Type_encoded'], axis=1)
    y = df['Type_encoded']
    
    # Simple model training without SMOTE
    model = RandomForestClassifier(
        n_estimators=100,  # Reduced for faster deployment
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model, le

model, le = load_advanced_model()

# REST OF YOUR PREMIUM CODE REMAINS EXACTLY THE SAME...
# [Keep all the beautiful interface, tabs, and features]
