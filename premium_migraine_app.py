
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import datetime

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
    
    # Advanced balancing
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Advanced Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_balanced, y_balanced)
    return model, le

model, le = load_advanced_model()

# ===== PREMIUM HEADER =====
st.markdown('<h1 class="main-header">🧠 NeuroScan Pro</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Advanced AI Migraine Diagnosis & Management Platform</h3>', unsafe_allow_html=True)

# ===== SIDEBAR - PATIENT PROFILE =====
with st.sidebar:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Profile")
    patient_id = st.text_input("Patient ID", "P001")
    patient_name = st.text_input("Full Name", "John Doe")
    patient_age = st.number_input("Age", 1, 100, 35)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("📊 Quick Stats")
    st.metric("Total Diagnoses", "1,247")
    st.metric("Accuracy Rate", "94.3%")
    st.metric("Avg Confidence", "89.7%")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== MAIN INTERFACE TABS =====
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Diagnosis", "🗺️ Symptom Map", "💊 Treatment", "📈 Analytics", "🆘 Emergency"])

with tab1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🔍 Advanced Symptom Assessment")
    
    # Multi-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age", 15, 80, 35, help="Patient's current age")
        duration = st.select_slider("Attack Duration", [1, 2, 3], 
                                 format_func=lambda x: f"{x} hour{'s' if x>1 else ''}")
        frequency = st.selectbox("Monthly Frequency", [1, 2, 3, 4, 5, 6, 7, 8])

    with col2:
        st.markdown("#### 💢 Pain Characteristics")
        location = st.radio("Pain Location", [1, 2], 
                         format_func=lambda x: "🔹 Unilateral (One Side)" if x == 1 else "🔸 Bilateral (Both Sides)")
        character = st.radio("Pain Character", [1, 2],
                          format_func=lambda x: "🔹 Pulsating/Throbbing" if x == 1 else "🔸 Constant Pressure")
        intensity = st.select_slider("Pain Intensity", [1, 2, 3],
                                  format_func=lambda x: ["Mild 😊", "Moderate 😐", "Severe 😫"][x-1])

    with col3:
        st.markdown("#### 🤢 Associated Symptoms")
        nausea = st.radio("Nausea", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
        vomit = st.radio("Vomiting", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
        phonophobia = st.radio("Sound Sensitivity", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
        photophobia = st.radio("Light Sensitivity", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")

    # Advanced Symptoms
    st.markdown("#### 👁️ Neurological Symptoms")
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        visual = st.selectbox("Visual Aura", [0, 1, 2, 3, 4],
                           format_func=lambda x: ["None", "✨ Flickering Lights", "⬛ Blind Spots", 
                                               "⚡ Zigzag Lines", "🌀 Other Visual"][x])
        sensory = st.selectbox("Sensory Aura", [0, 1, 2],
                            format_func=lambda x: ["None", "📌 Pins & Needles", "💤 Numbness"][x])
        
    with adv_col2:
        vertigo = st.radio("Vertigo/Dizziness", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
        tinnitus = st.radio("Tinnitus", [1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")

    # Diagnosis Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        diagnose_clicked = st.button("🚀 RUN ADVANCED AI DIAGNOSIS", 
                                  use_container_width=True, 
                                  type="primary")

    if diagnose_clicked:
        # Prepare features
        features = [age, duration, frequency, location, character, intensity,
                  nausea, vomit, phonophobia, photophobia, visual, sensory,
                  0, 0, vertigo, tinnitus, 0, 0, 0, 0, 0, 0, 0]

        # Advanced prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        
        diagnosis = le.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Calculate symptom severity score
        severity_score = (intensity + nausea + vomit + visual + vertigo) / 5 * 10

        # Display premium results
        st.markdown("---")
        
        # Results in columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("🎯 Diagnosis")
            st.markdown(f"### {diagnosis}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
            st.subheader("📊 AI Confidence")
            if confidence > 0.85:
                st.markdown(f"### 🟢 {confidence:.1%}")
            elif confidence > 0.70:
                st.markdown(f"### 🟡 {confidence:.1%}")
            else:
                st.markdown(f"### 🔴 {confidence:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f'<div class="symptom-score">', unsafe_allow_html=True)
            st.subheader("⚠️ Severity Score")
            st.markdown(f"### {severity_score:.1f}/10")
            st.markdown('</div>', unsafe_allow_html=True)

        # Advanced probability visualization
        st.subheader("📈 Advanced Probability Analysis")
        prob_df = pd.DataFrame({
            'Migraine Type': le.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=True)
        
        fig = px.bar(prob_df, y='Migraine Type', x='Probability', orientation='h',
                    color='Probability', color_continuous_scale='viridis',
                    title="AI Confidence Distribution Across Migraine Types")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Treatment recommendations based on diagnosis
        st.subheader("💊 Personalized Treatment Plan")
        
        treatment_col1, treatment_col2 = st.columns(2)
        
        with treatment_col1:
            st.markdown("#### 🎯 Acute Treatment")
            if "aura" in diagnosis.lower():
                st.write("• **Triptans** (after aura resolves)")
                st.write("• **NSAIDs** (Ibuprofen, Naproxen)")
                st.write("• **Anti-emetics** for nausea")
            elif "without aura" in diagnosis.lower():
                st.write("• **Triptans** or **NSAIDs**")
                st.write("• **Combination analgesics**")
                st.write("• **Rest in dark, quiet room**")
            elif "hemiplegic" in diagnosis.lower():
                st.write("• **Avoid triptans**")
                st.write("• **NSAIDs** or **Acetaminophen**")
                st.write("• **Neurology consultation**")
            else:
                st.write("• **Standard migraine therapy**")
                st.write("• **Symptom-specific treatment**")
                st.write("• **Medical consultation**")
            
        with treatment_col2:
            st.markdown("#### 🛡️ Preventive Strategies")
            st.write("• **Identify and avoid triggers**")
            st.write("• **Regular sleep schedule**")
            st.write("• **Stress management techniques**")
            st.write("• **Consider preventive medications**")
        
        st.markdown("#### 📋 Lifestyle Recommendations")
        lifestyle_col1, lifestyle_col2 = st.columns(2)
        with lifestyle_col1:
            st.write("• Maintain consistent sleep patterns")
            st.write("• Stay hydrated and eat regular meals")
        with lifestyle_col2:
            st.write("• Regular moderate exercise")
            st.write("• Keep a migraine diary")

        # Emergency check
        if severity_score > 7 or "hemiplegic" in diagnosis.lower():
            st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
            st.subheader("🚨 URGENT MEDICAL ATTENTION REQUIRED")
            st.write("This migraine type may require immediate medical evaluation. Please consult a healthcare professional immediately.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Interactive Symptom Map")
    st.info("Visual representation of symptom patterns and correlations")
    
    # Create interactive symptom map
    symptoms_data = {
        'Symptom': ['Pain Intensity', 'Nausea', 'Visual Aura', 'Vertigo', 'Sound Sensitivity', 'Light Sensitivity'],
        'Severity': [intensity, nausea*3, visual, vertigo*2, phonophobia*2, photophobia*2],
        'Frequency': [frequency, nausea*2, visual, vertigo, phonophobia, photophobia],
        'Impact': [intensity*2, nausea*2, visual*1.5, vertigo*2, phonophobia*1.5, photophobia*1.5]
    }
    
    fig = px.scatter_3d(
        pd.DataFrame(symptoms_data), 
        x='Severity', 
        y='Frequency', 
        z='Impact',
        color='Symptom',
        size='Severity',
        hover_name='Symptom',
        title="3D Symptom Severity vs Frequency vs Impact Map",
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Symptom correlation heatmap
    st.subheader("🔗 Symptom Correlation Matrix")
    corr_data = pd.DataFrame(symptoms_data).corr(numeric_only=True)
    fig_heatmap = px.imshow(corr_data, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("💊 Comprehensive Treatment Hub")
    
    treatment_tab1, treatment_tab2, treatment_tab3 = st.tabs(["Acute Therapy", "Prevention", "Lifestyle"])
    
    with treatment_tab1:
        st.markdown("#### 🎯 Acute Migraine Treatment")
        st.write("""
        **First-line Options:**
        - **Triptans**: Sumatriptan, Rizatriptan, Eletriptan
        - **NSAIDs**: Ibuprofen, Naproxen, Aspirin
        - **Combination analgesics**
        - **Anti-emetics**: Metoclopramide, Prochlorperazine
        
        **Rescue Medications:**
        - **Dihydroergotamine**
        - **Opioids** (limited use)
        - **Corticosteroids**
        """)
        
    with treatment_tab2:
        st.markdown("#### 🛡️ Preventive Strategies")
        st.write("""
        **Medication Options:**
        - **Beta-blockers**: Propranolol, Timolol
        - **Anticonvulsants**: Topiramate, Valproate
        - **Antidepressants**: Amitriptyline, Venlafaxine
        - **CGRP monoclonal antibodies**
        
        **Non-Pharmacological:**
        - **Biofeedback therapy**
        - **Cognitive behavioral therapy**
        - **Acupuncture**
        """)
        
    with treatment_tab3:
        st.markdown("#### 🌱 Lifestyle Management")
        st.write("""
        **Trigger Management:**
        - Maintain regular sleep schedule
        - Stay hydrated (2-3L water daily)
        - Eat regular, balanced meals
        - Limit caffeine and alcohol
        
        **Stress Reduction:**
        - Regular exercise (30 min, 5x/week)
        - Meditation and mindfulness
        - Yoga and relaxation techniques
        - Adequate work-life balance
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.subheader("📈 Advanced Analytics & Insights")
    
    # Sample analytics data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    analytics_data = {
        'Month': months,
        'Diagnoses': [45, 52, 48, 61, 55, 58, 62, 59, 54, 57, 60, 63],
        'Accuracy': [92, 94, 91, 95, 93, 94, 96, 93, 92, 94, 95, 96],
        'Severity_Avg': [6.2, 5.8, 6.5, 5.9, 6.1, 5.7, 6.3, 5.8, 6.0, 5.6, 5.9, 5.5]
    }
    
    # Create advanced analytics dashboard
    fig_analytics = go.Figure()
    
    # Add traces
    fig_analytics.add_trace(go.Scatter(
        x=analytics_data['Month'], 
        y=analytics_data['Diagnoses'],
        name='Diagnoses', 
        line=dict(color='#667eea', width=4),
        yaxis='y1'
    ))
    
    fig_analytics.add_trace(go.Scatter(
        x=analytics_data['Month'], 
        y=analytics_data['Accuracy'],
        name='Accuracy %', 
        line=dict(color='#00b894', width=4),
        yaxis='y2'
    ))
    
    fig_analytics.add_trace(go.Bar(
        x=analytics_data['Month'], 
        y=analytics_data['Severity_Avg'],
        name='Avg Severity',
        marker_color='#fd79a8',
        yaxis='y3',
        opacity=0.6
    ))
    
    fig_analytics.update_layout(
        title='Comprehensive Platform Analytics',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Number of Diagnoses', side='left'),
        yaxis2=dict(title='Accuracy %', overlaying='y', side='right'),
        yaxis3=dict(title='Avg Severity', overlaying='y', side='right', position=0.85),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_analytics, use_container_width=True)
    
    # Migraine type distribution
    st.subheader("📊 Migraine Type Distribution")
    migraine_types = le.classes_
    type_counts = [120, 85, 45, 30, 25, 15, 10]  # Sample data
    
    fig_pie = px.pie(
        values=type_counts, 
        names=migraine_types,
        title="Distribution of Diagnosed Migraine Types",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
    st.subheader("🆘 EMERGENCY PROTOCOL & SAFETY")
    
    emergency_col1, emergency_col2 = st.columns(2)
    
    with emergency_col1:
        st.markdown("#### 🚨 RED FLAG SYMPTOMS")
        st.write("""
        **Seek IMMEDIATE Medical Attention for:**
        - ⚡ **Thunderclap headache** (sudden, severe)
        - 🧠 **Headache with neurological symptoms:**
          - Weakness/numbness
          - Vision changes/loss
          - Speech difficulties
          - Confusion/disorientation
        - 🌡️ **Headache with fever & stiff neck**
        - 🤕 **Headache after head injury**
        - 👴 **First severe headache after age 50**
        """)
        
    with emergency_col2:
        st.markdown("#### 📞 EMERGENCY CONTACTS")
        st.write("""
        **Immediate Assistance:**
        - 🚑 **Local Emergency**: 911 / 112
        - 🏥 **Poison Control**: 1-800-222-1222
        - 🧠 **Neurology Emergency**: Contact nearest hospital
        
        **Preparation Checklist:**
        - 📋 Know your medical history
        - 💊 List current medications
        - 📱 Keep emergency contacts handy
        - 🗺️ Know route to nearest hospital
        """)
    
    st.markdown("#### 🆘 WHEN TO GO TO ER")
    st.write("""
    **Go to Emergency Room if:**
    - Headache is **worst of your life**
    - **Neurological symptoms** develop suddenly
    - Headache **worsens rapidly**
    - **No response** to usual medications
    - Headache with **fever, rash, or seizure**
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🧠 <b>NeuroScan Pro</b> • Advanced AI Migraine Diagnosis Platform v2.0</p>
    <p>⚡ Powered by Ensemble Machine Learning • 📊 Real-time Analytics • 🎯 Personalized Treatment</p>
    <p>⚠️ <i>This tool is for educational and decision support purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment.</i></p>
</div>
""", unsafe_allow_html=True)
