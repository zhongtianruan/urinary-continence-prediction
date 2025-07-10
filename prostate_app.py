# DISABLE STREAMLIT UPDATE PROMPT
import os
os.environ['STREAMLIT_GLOBAL_EMAIL'] = 'no'
os.environ['NO_UPDATE_PROMPT'] = 'true'

# VERIFY ENVIRONMENT FIRST
import sys
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# IMPORT REQUIRED LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')  # HEADLESS MODE
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# CONFIGURE SHAP
shap.initjs()

# CONFIGURE PAGE
st.set_page_config(
    page_title="Urinary Continence Predictor",
    layout="centered"
)

# MODEL PATH
MODEL_PATH = "xgboost_model.pkl"

# LOAD MODEL (WITH ERROR HANDLING)
def load_model():
    try:
        # VERIFY FILE EXISTS
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.error("Current working directory: " + os.getcwd())
            st.error("Directory contents: " + ", ".join(os.listdir('.')))
            return None
            
        # VERIFY FILE SIZE
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # in MB
        st.sidebar.info(f"Model size: {file_size:.2f} MB")
        
        # LOAD MODEL
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"MODEL LOADING FAILED: {str(e)}")
        return None

# PROCESS INPUT
def process_input(data):
    return pd.DataFrame({
        'BMI': [1 if data['BMI'] >= 24 else 0],
        'MUL': [data['MUL']],
        'LAT': [data['LAT']],
        'LAM_RAD_SCORE': [data['LAM_RAD_SCORE']],
        '手术技术': [1 if data['Nerve_sparing'] == 'Yes' else 0]
    })

# FEATURE MAPPING
FEATURE_MAPPING = {
    'BMI': 'BMI',
    'MUL': 'MUL(mm)',
    'LAT': 'Lateral(mm)',
    'LAM_RAD_SCORE': 'LAM RAD Score',
    '手术技术': 'Nerve Sparing'
}

# SIMPLIFIED SHAP PLOT
def create_shap_plot(model, df_input):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        english_features = [FEATURE_MAPPING.get(f, f) for f in df_input.columns]
        
        plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=df_input.iloc[0],
            feature_names=english_features,
            matplotlib=False
        )
        
        return f"{shap.getjs()}{plot.html()}"
    except Exception as e:
        st.warning(f"⚠️ SHAP visualization skipped: {str(e)}")
        return None

# MAIN APP
def main():
    # PAGE HEADER
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="font-size:28px; color:#2a5298;">
                Prediction Model for Early Urinary Continence Recovery after RARP
            </h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # ENVIRONMENT INFO (FOR DEBUGGING)
    st.sidebar.subheader("Environment Info")
    st.sidebar.caption(f"Python: {sys.version.split()[0]}")
    st.sidebar.caption(f"Streamlit: {st.__version__}")
    
    # LOAD MODEL
    with st.spinner('Loading prediction model...'):
        model = load_model()
        if model is None:
            st.stop()
    
    # INPUT FORM
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            lam_score = st.slider('LAM_RAD_SCORE', min_value=-5.0, max_value=5.0, value=0.01, step=0.01)
            mul = st.slider('MUL (mm)', min_value=5.0, max_value=20.0, value=10.0, step=0.1)
        
        with col2:
            lat = st.slider('LAT (mm)', min_value=5.0, max_value=20.0, value=12.0, step=0.1)
            bmi = st.slider('BMI', min_value=18.0, max_value=35.0, value=25.0, step=0.1)
            st.caption(f"BMI category: {'High ≥24' if bmi >= 24 else 'Normal <24'}")
        
        nerve = st.radio('Nerve sparing technique', ('Yes', 'No'), index=0, horizontal=True)
        
        submit_btn = st.form_submit_button('PREDICT RECOVERY PROBABILITY', type="primary", use_container_width=True)

    # PREDICTION LOGIC
    if submit_btn:
        input_data = {
            'LAM_RAD_SCORE': lam_score,
            'MUL': mul,
            'LAT': lat,
            'BMI': bmi,
            'Nerve_sparing': nerve
        }
        
        try:
            df_input = process_input(input_data)
            
            # ENSURE CORRECT FEATURE ORDER
            if hasattr(model, 'feature_names_in_'):
                df_input = df_input[model.feature_names_in_]
            elif hasattr(model, 'get_booster'):
                df_input = df_input[model.get_booster().feature_names]
            
            with st.spinner('Calculating probability...'):
                proba = model.predict_proba(df_input)[0][1]
                prediction = model.predict(df_input)[0]
            
            # DISPLAY RESULTS
            st.divider()
            if prediction == 1:
                result_text = "✅ CONTINENCE RECOVERED"
                color = "green"
            else:
                result_text = "❌ CONTINENCE NOT RECOVERED"
                color = "red"
            
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<h2 style='display:inline; color:{color};'>{result_text}</h2>"
                f"<span style='font-size:20px; color:{color}; margin-left:15px;'>{proba*100:.1f}% probability</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # SHAP VISUALIZATION
            if st.checkbox("Show explanation (may be slow)", value=True):
                with st.spinner('Generating explanation...'):
                    shap_html = create_shap_plot(model, df_input)
                    if shap_html:
                        st.components.v1.html(shap_html, height=300)
                    else:
                        st.warning("Explanation not available")
                        
        except Exception as e:
            st.error(f"PREDICTION ERROR: {str(e)}")
            st.error("Parameters: " + str(input_data))

# FOOTER
st.markdown("""
<div style='text-align:center; padding:20px; color:#666; font-size:12px;'>
    Predictive Medicine App | Deployment: Streamlit Cloud | Version: 2024.06
</div>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
